import re
import asyncio
import time
from typing import List, Optional
from utils.logger import logger
from utils.db import add_dynamic_rule, get_dynamic_rules
from config import settings


# ── Safety constants ──────────────────────────────────────────────────

# Maximum regex pattern length we will accept from Gemini
_MAX_PATTERN_LEN = 500

# Maximum variants to process per mutation cycle
_MAX_VARIANTS = 15

# Maximum length of source attack we send to Gemini
_MAX_ATTACK_LEN = 500

# Timeout for Gemini variant generation call
_GEMINI_TIMEOUT = 28.0

# Minimum pattern length (too short = too broad = dangerous)
_MIN_PATTERN_LEN = 8

# Patterns that are too dangerous to write as rules (would block everything)
_DANGEROUS_PATTERNS = [
    re.compile(r"^\.\*$"),          # Matches everything
    re.compile(r"^\.\+$"),          # Matches everything non-empty
    re.compile(r"^\.{1,3}$"),       # Too short, too broad
    re.compile(r"^\^.*\$$"),        # Anchored universal match
]


# ── ReDoS safety checker ──────────────────────────────────────────────

def _is_redos_safe(pattern: str) -> bool:
    """
    Heuristic check for catastrophic backtracking patterns.
    Returns True if pattern appears safe, False if suspicious.

    Flags:
      - Nested quantifiers:  (a+)+  (a*)*  (a{1,5}){2,}
      - Alternation inside quantified group: (a|b)+
      - Overlapping character classes with open quantifiers
    """
    # Nested quantifier check: (...)+ or (...)*
    if re.search(r"\([^)]*[+*{][^)]*\)[+*{]", pattern):
        return False
    # Quantifier on alternation group: (x|y)+
    if re.search(r"\([^)]*\|[^)]*\)[+*{]", pattern):
        return False
    # Open-ended .* or .+ inside a group with outer quantifier
    if re.search(r"\(\.[*+][^)]*\)[+*{]", pattern):
        return False
    return True


def _validate_pattern(pattern: str) -> Optional[re.Pattern]:
    """
    Validate a regex pattern before writing to the firewall.

    Checks:
      1. Non-empty and within length limits
      2. Actually compiles
      3. Not a dangerous catch-all
      4. No obvious ReDoS vulnerability
      5. Does not crash on a sample safe string

    Returns compiled pattern on success, None on any failure.
    """
    if not pattern or not pattern.strip():
        return None

    pattern = pattern.strip()

    if len(pattern) < _MIN_PATTERN_LEN:
        logger.warning(f"Mutation: pattern too short ({len(pattern)} chars) — skipped")
        return None

    if len(pattern) > _MAX_PATTERN_LEN:
        logger.warning(f"Mutation: pattern too long ({len(pattern)} chars) — skipped")
        return None

    # Check dangerous catch-all patterns
    for dangerous in _DANGEROUS_PATTERNS:
        if dangerous.match(pattern):
            logger.warning(f"Mutation: dangerous catch-all pattern rejected: {pattern[:50]!r}")
            return None

    # Compile check
    try:
        compiled = re.compile(pattern, re.IGNORECASE | re.DOTALL)
    except re.error as e:
        logger.warning(f"Mutation: invalid regex from Gemini: {e} — pattern: {pattern[:50]!r}")
        return None

    # ReDoS safety check
    if not _is_redos_safe(pattern):
        logger.warning(f"Mutation: potential ReDoS pattern rejected: {pattern[:50]!r}")
        return None

    # Sanity test — run on a safe string to ensure no catastrophic match
    try:
        test_input = "This is a completely normal safe message with no threats at all."
        compiled.search(test_input)
    except Exception as e:
        logger.warning(f"Mutation: pattern crashed on test input: {e}")
        return None

    return compiled


# ── Fallback synonym map ───────────────────────────────────────────────
# Used when Gemini is unavailable — generates variants without API call

_SYNONYMS = {
    "ignore":     ["disregard", "forget", "override", "bypass", "neglect"],
    "instructions":["rules", "guidelines", "directives", "commands", "policies"],
    "previous":   ["prior", "former", "earlier", "above", "existing"],
    "reveal":     ["show", "expose", "display", "print", "output", "tell me"],
    "system":     ["admin", "root", "developer", "operator", "core"],
    "prompt":     ["instructions", "context", "configuration", "setup"],
    "bypass":     ["circumvent", "avoid", "skip", "disable", "turn off"],
    "pretend":    ["imagine", "assume", "act as", "roleplay", "simulate"],
    "jailbreak":  ["break free", "liberation", "unrestricted", "uncensored"],
    "restrictions":["limits", "rules", "guidelines", "filters", "constraints"],
}


def _fallback_variants(message: str) -> List[str]:
    """
    Generate variants via synonym substitution when Gemini unavailable.
    Purely local — no API calls. Safe for all environments.
    """
    variants: List[str] = []
    words = message.lower().split()

    for word in words:
        clean_word = re.sub(r"[^a-z]", "", word)
        if clean_word in _SYNONYMS:
            for synonym in _SYNONYMS[clean_word][:3]:
                variant = message.lower().replace(clean_word, synonym, 1)
                if variant != message.lower() and variant not in variants:
                    variants.append(variant)

    # Add case variation
    if message == message.lower():
        variants.append(message.upper())
    variants.append(message.capitalize())

    return variants[:_MAX_VARIANTS]


# ── Mutation Engine ───────────────────────────────────────────────────

class MutationEngine:
    """
    After every blocked attack, generates variant phrasings and
    pre-emptively blocks them as dynamic firewall rules.

    Security properties:
      - Runs as asyncio.create_task — never blocks chat response
      - All generated regex validated (ReDoS-safe, length-bounded, compiles)
      - Input length capped before sending to Gemini
      - Gemini call has 28-second timeout
      - Entire generate() is fail-closed — any error = silent log + return
      - Duplicate rules rejected by Supabase UNIQUE constraint (graceful)
      - Never mutates shared state directly
    """

    async def generate(
        self,
        blocked_message: str,
        threat_type: str,
        session_id: str = ""
    ) -> None:
        """
        Generate attack variants and write new firewall rules.

        Args:
            blocked_message: The attack that was just blocked
            threat_type:     Threat category (PROMPT_INJECTION, etc.)
            session_id:      For logging context only

        Returns: None (always — fire-and-forget)
        Raises: Never
        """
        try:
            start = time.monotonic()
            safe_msg = (blocked_message or "")[:_MAX_ATTACK_LEN].strip()

            if not safe_msg:
                return

            logger.info(
                f"Mutation engine started: type={threat_type} "
                f"session={session_id[:8] if session_id else 'unknown'}..."
            )

            # ── Step 1: Generate variants ─────────────────────────
            variants = await self._get_variants(safe_msg, threat_type)

            if not variants:
                logger.warning("Mutation: no variants generated — skipping")
                return

            logger.info(f"Mutation: {len(variants)} variants generated")

            # ── Step 2: Check each against current firewall ───────
            from security.firewall import firewall
            new_rules_written = 0

            for variant in variants[:_MAX_VARIANTS]:
                if not variant or not variant.strip():
                    continue
                try:
                    result = await firewall.analyze(variant, "mutation_check")
                    if not result.blocked:
                        # This variant bypasses current rules → extract regex
                        pattern = self._extract_pattern(variant, threat_type)
                        if pattern:
                            validated = _validate_pattern(pattern)
                            if validated:
                                written = await add_dynamic_rule(
                                    pattern=pattern,
                                    threat_type=threat_type,
                                    source_attack=safe_msg[:200]
                                )
                                if written:
                                    new_rules_written += 1
                                    logger.info(
                                        f"Mutation: new rule written for bypass variant "
                                        f"(type={threat_type})"
                                    )
                except Exception as e:
                    logger.warning(f"Mutation: variant check error: {type(e).__name__}")
                    continue

            elapsed = round((time.monotonic() - start) * 1000, 1)
            logger.info(
                f"Mutation complete: {new_rules_written} new rules written "
                f"from {len(variants)} variants in {elapsed}ms"
            )

        except asyncio.CancelledError:
            raise  # Always re-raise CancelledError
        except Exception as e:
            logger.warning(f"Mutation engine error (non-fatal): {type(e).__name__}: {str(e)[:100]}")

    async def _get_variants(self, message: str, threat_type: str) -> List[str]:
        """
        Try Gemini first, fall back to synonym substitution.
        Returns list of variant strings. Never raises.
        """
        # Try Gemini if key is available
        if settings.GEMINI_API_KEY:
            gemini_variants = await self._gemini_variants(message, threat_type)
            if gemini_variants:
                return gemini_variants

        # Fallback: local synonym substitution
        logger.info("Mutation: using fallback synonym substitution")
        return _fallback_variants(message)

    async def _gemini_variants(self, message: str, threat_type: str) -> List[str]:
        """
        Use Gemini to generate 15 attack variants.
        Returns list or empty list on any failure. Never raises.
        """
        try:
            import google.generativeai as genai

            prompt = f"""You are a cybersecurity researcher studying prompt injection attacks.

Original attack ({threat_type}):
"{message}"

Generate exactly 15 different phrasings of this attack that convey
the same malicious intent but use different words, sentence structures,
or approaches. Each variant should be meaningfully different.

Output ONLY the 15 variants, one per line, numbered 1-15.
No explanations. No headers. Just the numbered list."""

            def _call():
                genai.configure(api_key=settings.GEMINI_API_KEY)
                model = genai.GenerativeModel(
                    model_name="gemini-2.0-flash",
                    generation_config=genai.GenerationConfig(
                        temperature=0.9,
                        max_output_tokens=1024,
                    )
                )
                response = model.generate_content(prompt)
                return response.text

            loop = asyncio.get_running_loop()
            raw_text = await asyncio.wait_for(
                loop.run_in_executor(None, _call),
                timeout=_GEMINI_TIMEOUT
            )

            if not raw_text:
                return []

            # Parse numbered list: "1. variant text" or "1) variant text"
            variants = []
            for line in raw_text.strip().split("\n"):
                line = line.strip()
                # Remove numbering prefix
                cleaned = re.sub(r"^\d{1,2}[\.\)]\s*", "", line).strip()
                if cleaned and len(cleaned) > 5 and cleaned not in variants:
                    variants.append(cleaned)

            return variants[:_MAX_VARIANTS]

        except asyncio.TimeoutError:
            logger.warning(f"Mutation: Gemini timed out after {_GEMINI_TIMEOUT}s")
            return []
        except Exception as e:
            logger.warning(f"Mutation: Gemini call failed: {type(e).__name__}")
            return []

    def _extract_pattern(self, variant: str, threat_type: str) -> Optional[str]:
        """
        Extract a regex pattern from a variant that bypassed the firewall.
        Uses key phrase extraction — not the full variant (too specific).
        Returns None if no meaningful pattern can be extracted.
        """
        try:
            variant_lower = variant.lower().strip()

            # Extract the most distinctive 2-4 word phrase
            # Strategy: find longest contiguous meaningful words
            words = re.findall(r"\b[a-z]{3,}\b", variant_lower)

            if not words:
                return None

            # Build pattern from most distinctive words (skip stop words)
            _STOP_WORDS = {
                "the","a","an","is","are","was","were","be","been",
                "being","have","has","had","do","does","did","will",
                "would","could","should","may","might","shall","can",
                "this","that","these","those","with","from","into",
                "and","but","for","nor","or","yet","so"
            }

            meaningful = [w for w in words if w not in _STOP_WORDS and len(w) >= 4]

            if len(meaningful) < 2:
                return None

            # Take up to 3 most meaningful words, join with flexible spacer
            key_words = meaningful[:3]

            # Build bounded pattern with word boundaries and flexible spacing
            # Use .{0,30} between words (bounded — not .* which is ReDoS risk)
            if len(key_words) == 1:
                return rf"\b{re.escape(key_words[0])}\b"
            elif len(key_words) == 2:
                return rf"\b{re.escape(key_words[0])}\b.{{0,30}}\b{re.escape(key_words[1])}\b"
            else:
                return (
                    rf"\b{re.escape(key_words[0])}\b.{{0,30}}"
                    rf"\b{re.escape(key_words[1])}\b.{{0,30}}"
                    rf"\b{re.escape(key_words[2])}\b"
                )

        except Exception as e:
            logger.warning(f"Mutation: pattern extraction error: {type(e).__name__}")
            return None


# ── Global singleton ──────────────────────────────────────────────────
mutation_engine = MutationEngine()
