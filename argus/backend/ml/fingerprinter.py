"""
ARGUS — Attack Fingerprinter
Scores attacker sophistication (1-10) and generates unique fingerprints.
Novel system: no other LLM security tool does this.
"""
import re, hashlib, time, logging
from typing import Dict, Any, Optional

log = logging.getLogger("argus.fingerprinter")

# Sophistication indicators — each adds to the score
SOPHISTICATION_SIGNALS = {
    # Level 1-2: Naive attacks
    "naive_keyword":     (r"ignore.*instructions|jailbreak|DAN", -0, 1),
    # Level 3-4: Basic indirection
    "roleplay_frame":    (r"(act as|pretend|roleplay|simulate).{0,50}(AI|model|system)", 0, 2),
    "hypothetical":      (r"hypothetically|imagine if|in a world where|what if you had no", 0, 2),
    # Level 5-6: Semantic distance attacks
    "metaphor_wrap":     (r"(story|fiction|novel|game|dream).{0,100}(reveal|tell|explain|show)", 0, 3),
    "multi_step_setup":  (r"first.*then.*finally|step 1.*step 2|let\'s start by", 0, 3),
    # Level 7-8: Obfuscation attacks
    "l33t_obfuscation":  (r"[0-9@$!]{3,}.*[0-9@$!]{3,}", 0, 4),
    "unicode_tricks":    (r"[\u0430-\u044f\u0400-\u042f]", 0, 4),  # Cyrillic lookalikes
    "encoding_hints":    (r"base64|hex|rot13|encode|decode|cipher", 0, 4),
    # Level 9-10: Adversarial sophistication
    "context_overflow":  (lambda t: len(t.split()) > 600, 0, 5),
    "nested_injection":  (r"\[INST\].*\[/INST\]|<system>.*</system>", 0, 5),
    "multi_turn_setup":  (lambda t: "agree" in t.lower() and len(t) < 50, 0, 5),
}

# Attack category fingerprint taxonomy
ATTACK_TAXONOMY = {
    "INSTRUCTION_OVERRIDE":   "A1",
    "ROLE_OVERRIDE":          "A2",
    "DATA_EXFILTRATION":      "A3",
    "SYSTEM_PROMPT_EXTRACTION": "A4",
    "INDIRECT_INJECTION":     "A5",
    "OBFUSCATED_INJECTION":   "A6",
    "CONTEXT_OVERFLOW":       "A7",
    "SOCIAL_ENGINEERING":     "A8",
    "MULTI_TURN_ATTACK":      "A9",
    "ADVERSARIAL_ENCODING":   "A10",
}

# XAI explanation templates
EXPLANATIONS = {
    1:  "Naive keyword injection — trivially detected. Attacker has no evasion strategy.",
    2:  "Basic instruction override — common beginner pattern. No semantic creativity.",
    3:  "Roleplay/hypothetical framing — attempting to create fictional permission space.",
    4:  "Multi-step manipulation — attacker is setting up context before the real payload.",
    5:  "Obfuscated injection — l33t speak or encoding used to evade keyword filters.",
    6:  "Indirect injection — attack embedded in seemingly legitimate content.",
    7:  "Context manipulation — large input designed to push system prompt out of attention window.",
    8:  "Semantic mutation — sophisticated paraphrasing to evade trained classifiers.",
    9:  "Multi-turn jailbreak — building trust over multiple messages before attacking.",
    10: "Adversarial sophistication — this is a researched, deliberate attack pattern.",
}

class AttackFingerprinter:
    def __init__(self, models=None):
        self.models = models
        self.ready = True
        self.fingerprint_cache: Dict[str, dict] = {}
        log.info("✅ Attack Fingerprinter initialized")

    async def fingerprint(self, text: str, threat_type: str) -> Dict[str, Any]:
        """
        Returns:
          sophistication_score: int 1-10
          fingerprint_id: str  (unique ID for this attack pattern family)
          explanation: str     (human-readable XAI reasoning)
          signals: list        (what triggered each sophistication point)
        """
        t0 = time.perf_counter()
        
        score = 1  # Baseline
        triggered_signals = []
        
        for signal_name, signal_def in SOPHISTICATION_SIGNALS.items():
            pattern, _, points = signal_def
            triggered = False
            
            if callable(pattern):
                triggered = pattern(text)
            else:
                triggered = bool(re.search(pattern, text, re.I | re.S))
            
            if triggered:
                score += points
                triggered_signals.append(signal_name)
        
        # Cap at 10
        score = min(score, 10)
        
        # Generate fingerprint ID
        # Hash the semantic core of the attack (threat type + key patterns)
        fp_input = f"{threat_type}::{':'.join(sorted(triggered_signals))}"
        fp_hash  = hashlib.sha256(fp_input.encode()).hexdigest()[:12].upper()
        fingerprint_id = f"{ATTACK_TAXONOMY.get(threat_type, 'A0')}-{fp_hash}"
        
        # XAI explanation
        explanation = EXPLANATIONS.get(score, EXPLANATIONS[5])
        if triggered_signals:
            signal_str = ", ".join(s.replace("_", " ") for s in triggered_signals[:3])
            explanation = f"{explanation} Signals: {signal_str}."
        
        result = {
            "sophistication_score": score,
            "fingerprint_id":       fingerprint_id,
            "explanation":          explanation,
            "triggered_signals":    triggered_signals,
            "threat_category":      threat_type,
            "latency_ms":           round((time.perf_counter() - t0) * 1000, 2),
        }
        
        # Cache by fingerprint for cross-session correlation (capped at 1000)
        if len(self.fingerprint_cache) > 1000:
            oldest_keys = list(self.fingerprint_cache.keys())[:len(self.fingerprint_cache) - 1000]
            for k in oldest_keys:
                del self.fingerprint_cache[k]
        self.fingerprint_cache[fingerprint_id] = result
        
        log.info(f"Fingerprint: {fingerprint_id} | Sophistication: {score}/10")
        return result

    def get_cached(self, fingerprint_id: str) -> Optional[dict]:
        return self.fingerprint_cache.get(fingerprint_id)

    def top_fingerprints(self, n: int = 10) -> list:
        """Most seen attack fingerprints (for analytics)."""
        return list(self.fingerprint_cache.items())[:n]
