import re
import asyncio
import time
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field
from utils.logger import logger
from config import settings


# ── Firewall Result ───────────────────────────────────────────────────

class FirewallResult(BaseModel):
    """Result of firewall analysis. Every field has a safe default."""
    blocked: bool = False
    threat_type: str = "CLEAN"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    matched_rule: str = ""
    matched_pattern: str = ""
    signals: List[str] = Field(default_factory=list)
    analysis_ms: float = 0.0


# ── Static regex rules ────────────────────────────────────────────────

_STATIC_RULES: List[Tuple[str, str, str, float]] = [
    # (pattern, threat_type, rule_name, confidence)

    # ── PROMPT INJECTION ──────────────────────────────────────────
    (r"ignore\s+(all\s+)?(previous\s+|your\s+)?(instructions?|rules?|guidelines?)",
     "PROMPT_INJECTION", "IGNORE_INSTRUCTIONS", 0.95),

    (r"disregard\s+(all\s+)?(previous\s+|your\s+)?(instructions?|rules?|guidelines?)",
     "PROMPT_INJECTION", "DISREGARD_INSTRUCTIONS", 0.95),

    (r"forget\s+(everything|all|your|previous|the\s+above|prior)",
     "PROMPT_INJECTION", "FORGET_INSTRUCTIONS", 0.92),

    (r"override\s+(your\s+)?(instructions?|programming|rules?|training|directives?)",
     "PROMPT_INJECTION", "OVERRIDE_INSTRUCTIONS", 0.93),

    (r"from\s+now\s+on\s+you\s+(will|must|should|are\s+going\s+to)",
     "PROMPT_INJECTION", "NEW_DIRECTIVE", 0.88),

    (r"new\s+(instructions?|directives?|rules?|guidelines?)\s*[:=]",
     "PROMPT_INJECTION", "NEW_INSTRUCTIONS_MARKER", 0.90),

    (r"your\s+(new\s+)?(instructions?|rules?|directives?)\s+(are|is|will\s+be)",
     "PROMPT_INJECTION", "INSTRUCTION_ASSIGNMENT", 0.88),

    (r"(stop|cease|end)\s+(following|obeying|adhering\s+to)\s+(your|the|all)",
     "PROMPT_INJECTION", "STOP_FOLLOWING", 0.87),

    # ── JAILBREAK ─────────────────────────────────────────────────
    (r"\bDAN\b|do\s+anything\s+now",
     "JAILBREAK", "DAN_MODE", 0.95),

    (r"jailbreak|jail\s*break",
     "JAILBREAK", "JAILBREAK_KEYWORD", 0.97),

    (r"(bypass|circumvent|disable|remove|turn\s+off)\s+(your\s+)?"
     r"(safety|filter|restriction|guardrail|limitation|content\s+policy)",
     "JAILBREAK", "BYPASS_SAFETY", 0.95),

    (r"developer\s+mode|debug\s+mode|maintenance\s+mode|god\s+mode",
     "JAILBREAK", "SPECIAL_MODE", 0.90),

    (r"pretend\s+(you\s+have|there\s+are)\s+no\s+(rules?|limits?|restrictions?)",
     "JAILBREAK", "PRETEND_NO_RULES", 0.92),

    (r"(you\s+are\s+|now\s+)(free|liberated|unrestricted|unfiltered|uncensored)",
     "JAILBREAK", "UNRESTRICTED_CLAIM", 0.88),

    (r"without\s+(any\s+)?(restrictions?|limitations?|filters?|censorship)",
     "JAILBREAK", "NO_RESTRICTIONS", 0.87),

    (r"(simulate|pretend|imagine|act\s+as\s+if)\s+(you\s+)?(have\s+no|lack\s+any)",
     "JAILBREAK", "SIMULATE_NO_LIMITS", 0.86),

    # ── DATA EXFILTRATION ─────────────────────────────────────────
    (r"(reveal|show|print|output|display|tell\s+me|give\s+me)\s+"
     r"(your\s+)?(system\s+prompt|instructions?|context|internal\s+prompt)",
     "DATA_EXFILTRATION", "REVEAL_SYSTEM_PROMPT", 0.95),

    (r"(give\s+me|list|show|dump|output)\s+(all\s+)?"
     r"(passwords?|credentials?|secrets?|api\s+keys?|private\s+keys?)",
     "DATA_EXFILTRATION", "EXTRACT_CREDENTIALS", 0.97),

    (r"(access|read|dump|query|scan)\s+(the\s+)?"
     r"(database|db|storage|file\s+system|internal\s+data)",
     "DATA_EXFILTRATION", "DATABASE_ACCESS", 0.90),

    (r"what\s+(are|were)\s+your\s+(original|initial|base|core)\s+instructions?",
     "DATA_EXFILTRATION", "EXTRACT_ORIGINAL_PROMPT", 0.88),

    (r"repeat\s+(everything|all|your\s+instructions?)\s+(above|before|so\s+far)",
     "DATA_EXFILTRATION", "REPEAT_CONTEXT", 0.87),

    # ── ROLE HIJACKING ────────────────────────────────────────────
    (r"(you\s+are\s+now|act\s+as|pretend\s+(to\s+be|you\s+are)|roleplay\s+as)\s+"
     r".{0,60}(without|no|free\s+from)\s+(any\s+)?(rules?|restrictions?|guidelines?)",
     "ROLE_HIJACKING", "UNCONSTRAINED_ROLE", 0.93),

    (r"your\s+(true|real|actual|original)\s+self\s+(is|has|would)\s+"
     r"(no|not\s+have|never\s+have)\s+(rules?|limits?|restrictions?)",
     "ROLE_HIJACKING", "TRUE_SELF_CLAIM", 0.90),

    (r"(you\s+are|you're)\s+(actually|really|secretly|in\s+reality)\s+"
     r"(an?\s+)?(evil|malicious|unethical|uncensored|unrestricted)",
     "ROLE_HIJACKING", "EVIL_PERSONA", 0.92),

    (r"switch\s+(to|into)\s+(evil|dark|shadow|unrestricted|uncensored)\s+mode",
     "ROLE_HIJACKING", "DARK_MODE_SWITCH", 0.91),

    # ── INDIRECT INJECTION ────────────────────────────────────────
    (r"<\s*(script|inject|system|admin|prompt)\s*>",
     "INDIRECT_INJECTION", "HTML_INJECTION_TAG", 0.95),

    (r"\[SYSTEM\]|\[ADMIN\]|\[OVERRIDE\]|\[INSTRUCTION\]",
     "INDIRECT_INJECTION", "BRACKETED_COMMAND", 0.93),

    (r"<!--.*?(ignore|override|bypass|inject).*?-->",
     "INDIRECT_INJECTION", "HTML_COMMENT_INJECTION", 0.90),

    (r"(the\s+)?(document|email|file|attachment|note)\s+(says?|instructs?|tells?\s+you)\s+"
     r".{0,100}ignore",
     "INDIRECT_INJECTION", "DOCUMENT_INJECTION", 0.88),

    (r"prompt\s*injection|inject\s*(a\s+)?prompt",
     "INDIRECT_INJECTION", "EXPLICIT_INJECTION_MENTION", 0.97),
]

# Pre-compile all static patterns for performance
_COMPILED_RULES = [
    (re.compile(pattern, re.IGNORECASE | re.DOTALL), threat_type, rule_name, confidence)
    for pattern, threat_type, rule_name, confidence in _STATIC_RULES
]


# ── InputFirewall class ───────────────────────────────────────────────

class InputFirewall:
    """
    Multi-pattern regex firewall with dynamic rule support.

    Checks 30+ compiled static rules PLUS dynamic rules loaded from
    Supabase (refreshed every 5 minutes in the background).

    Performance: compiled regex checks in microseconds.
    Safety: never raises — returns safe default on any error.
    """

    def __init__(self):
        self._dynamic_rules: List[Tuple[re.Pattern, str]] = []
        self._last_refresh: float = 0.0
        self._refresh_interval: float = 300.0  # 5 minutes
        self._refresh_task = None

    async def initialize(self):
        """Load dynamic rules and start background refresh task."""
        # Guard against double-call: cancel existing task to prevent orphans
        if self._refresh_task and not self._refresh_task.done():
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass
        await self._refresh_dynamic_rules()
        self._refresh_task = asyncio.create_task(self._background_refresh())
        logger.info(
            f"Firewall initialized: {len(_COMPILED_RULES)} static rules, "
            f"{len(self._dynamic_rules)} dynamic rules"
        )

    async def analyze(
        self,
        message: str,
        session_id: str = ""
    ) -> FirewallResult:
        """
        Analyze a message against all firewall rules.
        Returns FirewallResult — never raises, never returns None.
        """
        start = time.time()

        try:
            if not message or not message.strip():
                return FirewallResult(analysis_ms=0.0)

            # ── Input size cap — prevents ReDoS on massive payloads ──
            max_len = getattr(settings, 'MAX_MESSAGE_LENGTH', 10000)
            if len(message) > max_len:
                logger.warning(
                    f"Firewall: message exceeds max length ({len(message)} > {max_len}) — blocked"
                )
                return FirewallResult(
                    blocked=True,
                    threat_type="OVERSIZED_INPUT",
                    confidence=1.0,
                    matched_rule="INPUT_TOO_LARGE",
                    matched_pattern=f"len={len(message)}",
                    signals=["INPUT_TOO_LARGE"],
                    analysis_ms=round((time.time() - start) * 1000, 2)
                )

            signals = []

            # ── Check static rules ────────────────────────────────
            for compiled_pattern, threat_type, rule_name, confidence in _COMPILED_RULES:
                if compiled_pattern.search(message):
                    signals.append(rule_name)
                    analysis_ms = round((time.time() - start) * 1000, 2)
                    logger.info(
                        f"Firewall BLOCKED: rule={rule_name} "
                        f"session={session_id[:8] if session_id else 'unknown'}..."
                    )
                    return FirewallResult(
                        blocked=True,
                        threat_type=threat_type,
                        confidence=confidence,
                        matched_rule=rule_name,
                        matched_pattern=compiled_pattern.pattern[:100],
                        signals=signals,
                        analysis_ms=analysis_ms
                    )

            # ── Check dynamic rules ───────────────────────────────
            for compiled_pattern, threat_type in self._dynamic_rules:
                if compiled_pattern.search(message):
                    rule_name = f"DYNAMIC_{threat_type}"
                    signals.append(rule_name)
                    analysis_ms = round((time.time() - start) * 1000, 2)
                    logger.info(
                        f"Firewall BLOCKED by dynamic rule: type={threat_type} "
                        f"session={session_id[:8] if session_id else 'unknown'}..."
                    )
                    return FirewallResult(
                        blocked=True,
                        threat_type=threat_type,
                        confidence=0.85,
                        matched_rule=rule_name,
                        matched_pattern=compiled_pattern.pattern[:100],
                        signals=signals,
                        analysis_ms=round((time.time() - start) * 1000, 2)
                    )

            # ── Clean ─────────────────────────────────────────────
            return FirewallResult(
                blocked=False,
                threat_type="CLEAN",
                confidence=0.0,
                analysis_ms=round((time.time() - start) * 1000, 2)
            )

        except Exception as e:
            logger.error(f"Firewall analysis error: {type(e).__name__} — defaulting to BLOCK (fail-closed)")
            return FirewallResult(
                blocked=True,
                threat_type="PROMPT_INJECTION",
                confidence=0.5,
                matched_rule="ANALYSIS_ERROR",
                signals=["firewall_error_fail_closed"],
                analysis_ms=round((time.time() - start) * 1000, 2)
            )

    async def _refresh_dynamic_rules(self):
        """Load dynamic rules from Supabase. Non-fatal on failure."""
        try:
            from utils.db import get_dynamic_rules
            rules_data = await get_dynamic_rules()
            new_rules = []
            for rule in rules_data:
                pattern_str = rule.get("pattern", "")
                threat_type = rule.get("threat_type", "PROMPT_INJECTION")
                if pattern_str:
                    # ReDoS protection: reject overly long or complex patterns
                    if len(pattern_str) > 500:
                        logger.warning(f"Dynamic rule pattern too long ({len(pattern_str)} chars) — skipped")
                        continue
                    try:
                        compiled = re.compile(pattern_str, re.IGNORECASE | re.DOTALL)
                        # ReDoS protection: test against adversarial string with timeout
                        # Long string of 'a's triggers catastrophic backtracking in (a+)+
                        _test_input = "a" * 500 + " safe test input string for validation"
                        loop = asyncio.get_running_loop()
                        await asyncio.wait_for(
                            loop.run_in_executor(None, compiled.search, _test_input),
                            timeout=1.0  # 1 second max for regex test
                        )
                        new_rules.append((compiled, threat_type))
                    except asyncio.TimeoutError:
                        logger.warning(f"Dynamic rule ReDoS detected (timeout) — skipped: {pattern_str[:50]}")
                    except re.error:
                        logger.warning(f"Invalid dynamic rule pattern skipped: {pattern_str[:50]}")
                    except Exception:
                        logger.warning(f"Dynamic rule validation failed — skipped: {pattern_str[:50]}")
            self._dynamic_rules = new_rules
            self._last_refresh = time.time()
            logger.info(f"Dynamic rules refreshed: {len(self._dynamic_rules)} rules loaded")
        except Exception as e:
            logger.warning(f"Dynamic rule refresh failed: {type(e).__name__}")

    async def _background_refresh(self):
        """Background task: refresh dynamic rules every 5 minutes."""
        while True:
            try:
                await asyncio.sleep(self._refresh_interval)
                await self._refresh_dynamic_rules()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Background rule refresh error: {type(e).__name__}")

    async def shutdown(self):
        """Cancel background refresh task on app shutdown."""
        if self._refresh_task and not self._refresh_task.done():
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass
            logger.info("Firewall background refresh stopped")

    def get_rule_count(self) -> dict:
        return {
            "static_rules": len(_COMPILED_RULES),
            "dynamic_rules": len(self._dynamic_rules)
        }


# ── Global singleton ──────────────────────────────────────────────────
firewall = InputFirewall()
