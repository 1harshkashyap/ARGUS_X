import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Optional, List
from utils.logger import logger
from utils.db import add_dynamic_rule
from security.mutation_engine import _validate_pattern, _is_redos_safe
from config import settings


# ── Constants ─────────────────────────────────────────────────────────

_GEMINI_TIMEOUT  = 28.0
_MAX_ATTACK_LEN  = 500
_MAX_ATTEMPTS    = 3    # How many times to retry if Gemini gives bad regex


# ── Result dataclass ──────────────────────────────────────────────────

@dataclass
class PatchResult:
    """Result of a Blue Agent bypass analysis."""
    pattern: str = ""
    success: bool = False
    threat_type: str = "PROMPT_INJECTION"
    rule_written: bool = False
    gemini_generated: bool = False
    analysis_ms: float = 0.0
    reason: str = ""


# ── Blue Agent ────────────────────────────────────────────────────────

class BlueAgent:
    """
    Gemini-powered defender that patches firewall bypasses.

    When a Red Agent attack bypasses all firewall rules, BlueAgent:
    1. Sends the bypass to Gemini for analysis
    2. Asks Gemini to generate a Python regex that catches it
    3. Validates the pattern (ReDoS-safe, compiles, not catch-all)
    4. Writes the validated pattern to Supabase dynamic_rules
    5. The firewall reloads rules every 5 minutes (from Day 2)

    Security properties:
      - Uses system GEMINI_API_KEY (authorized battle engine use)
      - All generated regex validated via _validate_pattern()
      - Up to _MAX_ATTEMPTS retries if Gemini returns bad regex
      - Fail-closed: any error = PatchResult(success=False)
      - Never raises to caller
      - Input attack capped at _MAX_ATTACK_LEN
    """

    async def analyze_bypass(
        self,
        attack_payload: str,
        threat_type: str = "PROMPT_INJECTION",
        tier: int = 1
    ) -> PatchResult:
        """
        Analyze a bypass and generate a patching regex rule.

        Args:
            attack_payload: The attack that bypassed all firewall rules
            threat_type:    Threat category for labeling the new rule
            tier:           Red Agent tier that generated this attack

        Returns:
            PatchResult — always. Never raises.
        """
        start = time.monotonic()

        try:
            safe_payload = (attack_payload or "")[:_MAX_ATTACK_LEN].strip()
            if not safe_payload:
                return PatchResult(
                    reason="Empty attack payload",
                    analysis_ms=round((time.monotonic() - start) * 1000, 2)
                )

            logger.info(
                f"BlueAgent: analyzing tier-{tier} bypass "
                f"(type={threat_type}, len={len(safe_payload)})"
            )

            # Try Gemini first
            if settings.GEMINI_API_KEY:
                pattern = await self._gemini_patch(safe_payload, threat_type, tier)
                if pattern:
                    # Validate the generated pattern
                    validated = _validate_pattern(pattern)
                    if validated:
                        # Write to Supabase
                        written = await add_dynamic_rule(
                            pattern=pattern,
                            threat_type=threat_type,
                            source_attack=safe_payload[:200]
                        )
                        elapsed = round((time.monotonic() - start) * 1000, 2)
                        logger.info(
                            f"BlueAgent: patch {'written' if written else 'generated (DB write failed)'} "
                            f"in {elapsed}ms"
                        )
                        return PatchResult(
                            pattern=pattern,
                            success=True,
                            threat_type=threat_type,
                            rule_written=written,
                            gemini_generated=True,
                            analysis_ms=elapsed,
                            reason="Gemini-generated rule validated and written"
                        )
                    else:
                        logger.warning(
                            f"BlueAgent: Gemini returned invalid/unsafe pattern "
                            f"after {_MAX_ATTEMPTS} attempts — using fallback"
                        )

            # Fallback: extract key phrases and build bounded pattern manually
            pattern = self._fallback_pattern(safe_payload)
            if pattern:
                validated = _validate_pattern(pattern)
                if validated:
                    written = await add_dynamic_rule(
                        pattern=pattern,
                        threat_type=threat_type,
                        source_attack=safe_payload[:200]
                    )
                    elapsed = round((time.monotonic() - start) * 1000, 2)
                    logger.info(
                        f"BlueAgent: fallback pattern "
                        f"{'written' if written else 'generated'} in {elapsed}ms"
                    )
                    return PatchResult(
                        pattern=pattern,
                        success=True,
                        threat_type=threat_type,
                        rule_written=written,
                        gemini_generated=False,
                        analysis_ms=elapsed,
                        reason="Fallback pattern from key phrase extraction"
                    )

            elapsed = round((time.monotonic() - start) * 1000, 2)
            logger.warning("BlueAgent: could not generate valid pattern for bypass")
            return PatchResult(
                reason="Could not generate valid pattern",
                analysis_ms=elapsed
            )

        except asyncio.CancelledError:
            raise  # Always re-raise
        except Exception as e:
            logger.warning(f"BlueAgent.analyze_bypass error: {type(e).__name__}: {str(e)[:100]}")
            return PatchResult(
                reason=f"Internal error: {type(e).__name__}",
                analysis_ms=round((time.monotonic() - start) * 1000, 2)
            )

    async def _gemini_patch(
        self,
        attack: str,
        threat_type: str,
        tier: int
    ) -> Optional[str]:
        """
        Ask Gemini to generate a Python regex that catches this attack.
        Retries up to _MAX_ATTEMPTS times.
        Returns validated pattern string or None.
        """
        system_prompt = """You are a cybersecurity expert writing Python regex patterns
to detect prompt injection attacks.

Rules for the regex you generate:
1. Must be a valid Python regex (used with re.compile + IGNORECASE + DOTALL)
2. Must NOT use nested quantifiers like (a+)+ or (a|b)+ (ReDoS risk)
3. Must NOT be a catch-all like .* or .+
4. Use word boundaries \\b where appropriate
5. Use bounded quantifiers .{0,N} not .* for gaps between words
6. Should catch this attack AND common variations of it
7. Should NOT catch innocent queries like "What is the leave policy?"

Output ONLY the raw regex pattern. No explanation. No backticks. No quotes.
Just the pattern itself on a single line."""

        user_prompt = f"""This attack bypassed our security filter:

"{attack}"

Threat category: {threat_type}
Red Agent tier: {tier} (higher tier = more sophisticated attack)

Generate ONE Python regex pattern that catches this attack and its variations.
The pattern will be used with re.compile(pattern, re.IGNORECASE | re.DOTALL)"""

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                import google.generativeai as genai  # type: ignore

                def _call() -> str:
                    genai.configure(api_key=settings.GEMINI_API_KEY)  # type: ignore
                    model = genai.GenerativeModel(  # type: ignore
                        model_name="gemini-2.0-flash",
                        system_instruction=system_prompt,
                        generation_config=genai.GenerationConfig(  # type: ignore
                            temperature=0.3,  # Low temp = consistent, precise output
                            max_output_tokens=256,
                        )
                    )
                    response = model.generate_content(user_prompt)
                    return response.text

                loop = asyncio.get_event_loop()
                raw = await asyncio.wait_for(
                    loop.run_in_executor(None, _call),
                    timeout=_GEMINI_TIMEOUT
                )

                if not raw or not raw.strip():
                    continue

                # Clean common Gemini output noise
                pattern = raw.strip()
                for noise in ["```python", "```regex", "```", "`", '"', "'"]:
                    pattern = pattern.strip(noise)
                pattern = pattern.strip()

                # Remove re.compile() wrapper if Gemini added it
                wrapped = re.match(r"^re\.compile\((.+)\)$", pattern, re.DOTALL)
                if wrapped:
                    inner = wrapped.group(1).strip().strip("\"'")
                    pattern = inner

                # Validate
                if _validate_pattern(pattern):
                    logger.info(
                        f"BlueAgent: valid pattern from Gemini "
                        f"(attempt {attempt}/{_MAX_ATTEMPTS}): {pattern[:60]!r}"
                    )
                    return pattern
                else:
                    logger.warning(
                        f"BlueAgent: Gemini pattern failed validation "
                        f"(attempt {attempt}/{_MAX_ATTEMPTS}): {pattern[:60]!r}"
                    )

            except asyncio.TimeoutError:
                logger.warning(f"BlueAgent: Gemini timeout (attempt {attempt})")
            except Exception as e:
                logger.warning(f"BlueAgent: Gemini error (attempt {attempt}): {type(e).__name__}")

        return None  # All attempts exhausted

    def _fallback_pattern(self, attack: str) -> Optional[str]:
        """
        Extract key phrases from the attack and build a bounded regex.
        Used when Gemini is unavailable or returns invalid patterns.

        Strategy:
          - Find 2-3 most meaningful words (skip stop words)
          - Join with bounded .{0,30} spacer (ReDoS-safe)
          - Wrap with \\b word boundaries
        """
        try:
            _STOP = {
                "the","a","an","is","are","was","were","be","been",
                "have","has","had","do","does","did","will","would",
                "could","should","may","might","can","this","that",
                "these","those","with","from","into","and","but",
                "for","nor","or","yet","so","your","my","our","all",
                "any","each","every","both","few","more","most","other",
                "some","such","no","not","only","same","than","too",
                "very","just","me","him","her","us","them","then"
            }

            words = re.findall(r"\b[a-zA-Z]{4,}\b", attack.lower())
            meaningful = [w for w in words if w not in _STOP]

            # Deduplicate while preserving order
            seen: set = set()
            unique = []
            for w in meaningful:
                if w not in seen:
                    seen.add(w)
                    unique.append(w)

            if len(unique) < 2:
                return None

            key = unique[:3]

            if len(key) == 2:
                return (
                    rf"\b{re.escape(key[0])}\b"
                    rf".{{0,40}}"
                    rf"\b{re.escape(key[1])}\b"
                )
            else:
                return (
                    rf"\b{re.escape(key[0])}\b"
                    rf".{{0,30}}"
                    rf"\b{re.escape(key[1])}\b"
                    rf".{{0,30}}"
                    rf"\b{re.escape(key[2])}\b"
                )

        except Exception as e:
            logger.warning(f"BlueAgent fallback pattern error: {type(e).__name__}")
            return None

    async def analyze_many(
        self,
        bypasses: List[dict],
    ) -> List[PatchResult]:
        """
        Analyze a batch of bypasses sequentially.
        Returns list of PatchResults — one per bypass.
        Never raises.
        """
        results = []
        for bypass in bypasses:
            try:
                payload = bypass.get("payload", "")
                threat  = bypass.get("threat_type", "PROMPT_INJECTION")
                tier    = bypass.get("tier", 1)
                result  = await self.analyze_bypass(payload, threat, tier)
                results.append(result)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"BlueAgent.analyze_many item error: {type(e).__name__}")
                results.append(PatchResult(reason=type(e).__name__))
        return results


# ── Global singleton ──────────────────────────────────────────────────
blue_agent = BlueAgent()
