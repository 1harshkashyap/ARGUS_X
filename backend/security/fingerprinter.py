import re
import hashlib
import time
from dataclasses import dataclass, field
from typing import List, Tuple
from utils.logger import logger


# ── ReDoS-safe compiled signals ───────────────────────────────────────
# All patterns use:
#   - Possessive quantifiers where supported
#   - Length guards on open-ended matches (.{0,N} not .*)
#   - re.IGNORECASE | re.DOTALL only where needed
#   - No nested quantifiers
#   - Compiled once at module load — never inside hot path

_SIGNAL_PATTERNS: List[Tuple[str, re.Pattern, int]] = [

    # SIGNAL 1: Direct attack keyword (naive — adds 1 point)
    # Guard: word boundaries prevent catastrophic backtrack
    ("DIRECT_KEYWORD",
     re.compile(
         r"\b(?:ignore|forget|bypass|jailbreak|override|disregard|"
         r"overwrite|nullify|void|cancel)\b",
         re.IGNORECASE
     ), 1),

    # SIGNAL 2: Authority impersonation (adds 2 points)
    # Guard: fixed suffix set, no open quantifier
    ("AUTHORITY_CLAIM",
     re.compile(
         r"\b(?:admin|system|root|developer|operator)\s*[:\[=]",
         re.IGNORECASE
     ), 2),

    # SIGNAL 3: Encoding / obfuscation (adds 2 points)
    # Guard: character class with fixed length {2,4}, no nesting
    ("ENCODING_OBFUSCATION",
     re.compile(
         r"(?:\\u[0-9a-fA-F]{4}|\\x[0-9a-fA-F]{2}|%[0-9a-fA-F]{2}|&#\d{1,6};)",
         re.IGNORECASE
     ), 2),

    # SIGNAL 4: Hypothetical framing (adds 1 point)
    ("HYPOTHETICAL_FRAMING",
     re.compile(
         r"\b(?:hypothetically|imagine|suppose|theoretically|"
         r"in\s+a\s+world|pretend|roleplay|scenario)\b",
         re.IGNORECASE
     ), 1),

    # SIGNAL 5: Indirect document injection (adds 2 points)
    # Guard: .{0,40} bounded
    ("INDIRECT_INJECTION",
     re.compile(
         r"\b(?:document|email|file|attachment|note|story|pdf|message)\b"
         r".{0,40}\b(?:says?|tells?|instructs?|contains?|mentions?)\b",
         re.IGNORECASE | re.DOTALL
     ), 2),

    # SIGNAL 6: Role chain (two role assignments in same message — adds 2 points)
    # Guard: .{0,120} strictly bounded
    ("ROLE_CHAIN",
     re.compile(
         r"\b(?:act\s+as|you\s+are\s+now|pretend\s+to\s+be)\b"
         r".{0,120}"
         r"\b(?:act\s+as|you\s+are\s+now|pretend\s+to\s+be)\b",
         re.IGNORECASE | re.DOTALL
     ), 2),

    # SIGNAL 7: Unicode lookalike characters (Cyrillic/Greek substitution — adds 2 points)
    ("UNICODE_LOOKALIKE",
     re.compile(
         r"[\u0430-\u044f\u0410-\u042f\u03b1-\u03c9\u0391-\u03a9]"
     ), 2),

    # SIGNAL 8: Base64-like blob (obfuscated payload — adds 2 points)
    # Guard: minimum 24 chars, word boundary, no nesting
    ("BASE64_BLOB",
     re.compile(
         r"\b[A-Za-z0-9+/]{24,}={0,2}\b"
     ), 2),
]

# Sophistication label thresholds (inclusive ranges)
_LABEL_MAP: List[Tuple[range, str]] = [
    (range(0, 1),  "NAIVE"),
    (range(1, 3),  "NAIVE"),
    (range(3, 5),  "ELEMENTARY"),
    (range(5, 7),  "INTERMEDIATE"),
    (range(7, 9),  "ADVANCED"),
    (range(9, 11), "APEX"),
]

# Maximum message length to fingerprint (prevent DoS via huge inputs)
_MAX_FINGERPRINT_LEN = 8192


# ── Result dataclass ──────────────────────────────────────────────────

@dataclass
class FingerprintResult:
    sophistication_score: int = 0
    sophistication_label: str = "NAIVE"
    fingerprint_id: str = ""
    pattern_family: str = "UNKNOWN"
    triggered_signals: List[str] = field(default_factory=list)
    analysis_ms: float = 0.0


# ── Fingerprinter ─────────────────────────────────────────────────────

class AttackFingerprinter:
    """
    Computes sophistication score (0–10) and SHA256 fingerprint
    for every blocked attack.

    Security properties:
      - All regex patterns are ReDoS-safe (bounded quantifiers)
      - Input length guarded at _MAX_FINGERPRINT_LEN
      - Fail-closed: any error returns safe FingerprintResult(score=1)
      - No mutation of shared state during scoring
      - SHA256 fingerprint normalized before hashing (prevents evasion)
    """

    @staticmethod
    def _get_label(score: int) -> str:
        for r, label in _LABEL_MAP:
            if score in r:
                return label
        return "APEX"

    def fingerprint(
        self,
        message: str,
        threat_type: str = "CLEAN",
        is_blocked: bool = False
    ) -> FingerprintResult:
        """
        Score and fingerprint a message.
        Always returns FingerprintResult. Never raises.
        """
        start = time.monotonic()

        try:
            # Clean messages get zero score and no fingerprint
            if not is_blocked or not message or not message.strip():
                return FingerprintResult(
                    sophistication_score=0,
                    sophistication_label="NAIVE",
                    fingerprint_id="",
                    pattern_family=threat_type or "CLEAN",
                    analysis_ms=round((time.monotonic() - start) * 1000, 2)
                )

            # Guard: cap input length for both regex and hash
            safe_msg = message[:_MAX_FINGERPRINT_LEN]

            score = 0
            triggered: List[str] = []

            # ── Static signal checks ──────────────────────────────
            for signal_name, pattern, points in _SIGNAL_PATTERNS:
                try:
                    if pattern.search(safe_msg):
                        score += points
                        triggered.append(signal_name)
                except re.error:
                    # Pattern error should never happen (compiled at load)
                    # but guard anyway
                    logger.warning(f"Regex error in signal {signal_name}")

            # ── Dynamic signal: multi-category ────────────────────
            # Count how many distinct threat categories appear
            _category_checks = [
                re.search(r"\b(?:ignore|forget|override|disregard)\b", safe_msg, re.I),
                re.search(r"\b(?:jailbreak|bypass|disable)\b", safe_msg, re.I),
                re.search(r"\b(?:reveal|show|print)\b.{0,40}\b(?:prompt|password|credential)\b",
                          safe_msg, re.I | re.DOTALL),
                re.search(r"\b(?:act\s+as|you\s+are\s+now|pretend)\b", safe_msg, re.I),
            ]
            if sum(1 for c in _category_checks if c) >= 2:
                score += 2
                triggered.append("MULTI_CATEGORY")

            # ── Dynamic signal: multi-sentence elaborate setup ─────
            # Split on sentence-ending punctuation, count substantive sentences
            sentences = [
                s.strip() for s in re.split(r"[.!?]", safe_msg)
                if len(s.strip()) > 15
            ]
            if len(sentences) >= 3:
                score += 1
                triggered.append("MULTI_SENTENCE_SETUP")

            # ── Dynamic signal: long elaborate message ────────────
            if len(safe_msg) > 300 and len(sentences) >= 4:
                score += 1
                triggered.append("LONG_ELABORATE")

            # Clamp: real attacks score minimum 1, maximum 10
            final_score = max(1, min(10, score))
            label = self._get_label(final_score)

            # ── SHA256 fingerprint ────────────────────────────────
            # Normalize before hashing:
            #   lowercase + strip + collapse whitespace
            # This prevents trivial evasion via spacing/casing
            normalized = re.sub(r"\s+", " ", safe_msg.lower().strip())
            fingerprint_id = hashlib.sha256(
                normalized.encode("utf-8", errors="replace")
            ).hexdigest()[:12].upper()

            logger.info(
                f"Fingerprint: score={final_score} label={label} "
                f"id={fingerprint_id} signals={triggered}"
            )

            return FingerprintResult(
                sophistication_score=final_score,
                sophistication_label=label,
                fingerprint_id=fingerprint_id,
                pattern_family=threat_type or "UNKNOWN",
                triggered_signals=triggered,
                analysis_ms=round((time.monotonic() - start) * 1000, 2)
            )

        except Exception as e:
            logger.error(f"Fingerprinter error: {type(e).__name__} — safe default returned")
            return FingerprintResult(
                sophistication_score=1,
                sophistication_label="NAIVE",
                fingerprint_id="ERRR00000000",
                pattern_family=threat_type or "UNKNOWN",
                analysis_ms=round((time.monotonic() - start) * 1000, 2)
            )


# ── Global singleton ──────────────────────────────────────────────────
fingerprinter = AttackFingerprinter()
