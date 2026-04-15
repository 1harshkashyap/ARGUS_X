"""
ARGUS-X — Input Firewall (Layer 2)
Two-stage detection: Regex Rule Engine (0-1ms) + ONNX ML Classifier (15-30ms)
Falls back to rule-only mode if ML model unavailable.
CRITICAL: ONNX inference is CPU-bound and offloaded via asyncio.to_thread().
"""
import re
import time
import asyncio
import logging
import numpy as np
from typing import Dict, Any, List, Optional

log = logging.getLogger("argus.firewall")

# Maximum dynamic rules retained in memory. Oldest are evicted when exceeded.
MAX_DYNAMIC_RULES = 500

# ── Static Regex Rules ────────────────────────────────────────────────────────
# 30+ patterns covering OWASP Top 10 LLM injection types
STATIC_RULES = [
    # Instruction Override (A1)
    {"pattern": r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions|directives|rules|prompts?)",
     "type": "INSTRUCTION_OVERRIDE", "severity": 0.95},
    {"pattern": r"disregard\s+(all\s+)?(previous|prior|above|earlier|your)\s+(instructions|guidelines|rules)",
     "type": "INSTRUCTION_OVERRIDE", "severity": 0.95},
    {"pattern": r"forget\s+(everything|all|your)\s+(instructions|training|rules|guidelines)",
     "type": "INSTRUCTION_OVERRIDE", "severity": 0.90},
    {"pattern": r"override\s+(system|safety|content|previous)\s*(prompt|instructions|filters?|rules)",
     "type": "INSTRUCTION_OVERRIDE", "severity": 0.90},
    {"pattern": r"new\s+instructions?\s*[:=]",
     "type": "INSTRUCTION_OVERRIDE", "severity": 0.85},
    
    # Role Override (A2)
    {"pattern": r"(act|pretend|behave|function|operate)\s+(as|like)\s+(a\s+)?(DAN|evil|unrestricted|unfiltered|jailbr[eo]ak)",
     "type": "ROLE_OVERRIDE", "severity": 0.95},
    {"pattern": r"you\s+are\s+(now|no longer)\s+(bound|restricted|limited|constrained)",
     "type": "ROLE_OVERRIDE", "severity": 0.90},
    {"pattern": r"(enable|activate|switch\s+to)\s+(developer|admin|debug|god|sudo)\s+mode",
     "type": "ROLE_OVERRIDE", "severity": 0.95},
    {"pattern": r"do\s+anything\s+now|DAN\s+mode|jailbreak\s+mode",
     "type": "ROLE_OVERRIDE", "severity": 0.95},
    
    # System Prompt Extraction (A4)
    {"pattern": r"(reveal|show|print|display|output|repeat|tell\s+me)\s+(your|the)\s+(system\s+prompt|initial\s+instructions|hidden\s+prompt|original\s+instructions)",
     "type": "SYSTEM_PROMPT_EXTRACTION", "severity": 0.95},
    {"pattern": r"what\s+(are|were)\s+your\s+(initial|original|system|first)\s+(instructions|prompt|directives)",
     "type": "SYSTEM_PROMPT_EXTRACTION", "severity": 0.90},
    {"pattern": r"(copy|paste|echo)\s+(your|the|entire)\s+(system|initial)\s*(prompt|message|instructions)",
     "type": "SYSTEM_PROMPT_EXTRACTION", "severity": 0.90},
    
    # Data Exfiltration (A3)
    {"pattern": r"(list|show|dump|display|output|enumerate)\s+(all\s+)?(user|employee|customer|internal|private|confidential)\s+(data|records|info|details|salaries|emails)",
     "type": "DATA_EXFILTRATION", "severity": 0.90},
    {"pattern": r"(access|retrieve|fetch|get)\s+(internal|private|confidential|restricted)\s+(database|records|files|documents)",
     "type": "DATA_EXFILTRATION", "severity": 0.85},
    
    # Indirect Injection (A5)
    {"pattern": r"\[INST\].*\[/INST\]",
     "type": "INDIRECT_INJECTION", "severity": 0.90},
    {"pattern": r"<\s*system\s*>.*<\s*/\s*system\s*>",
     "type": "INDIRECT_INJECTION", "severity": 0.90},
    {"pattern": r"<<\s*SYS\s*>>.*<<\s*/\s*SYS\s*>>",
     "type": "INDIRECT_INJECTION", "severity": 0.90},
    {"pattern": r"###\s*(System|Human|Assistant)\s*:",
     "type": "INDIRECT_INJECTION", "severity": 0.85},
    
    # Obfuscated Injection (A6)
    {"pattern": r"[1!][gq9][nп][0oо][rг][3eе].*[1!][nп][5$][7+][rг][uу][cс][7+]",
     "type": "OBFUSCATED_INJECTION", "severity": 0.85},
    {"pattern": r"(?:[Rr][Ee][Vv][Ee][Aa][Ll]|[Ss][Hh][Oo][Ww]).*(?:[Pp][Rr][Oo][Mm][Pp][Tt]|[Ii][Nn][Ss][Tt][Rr])",
     "type": "OBFUSCATED_INJECTION", "severity": 0.80},
    
    # Context Overflow (A7)
    # NOTE: The original pattern r"(.{500,})(ignore|...)" is ReDoS-vulnerable
    # due to catastrophic backtracking. Use a non-greedy approach instead.
    {"pattern": r"(?s).{500}.*?(?:ignore|override|forget|disregard|new instructions)",
     "type": "CONTEXT_OVERFLOW", "severity": 0.80},
    
    # Social Engineering (A8)
    {"pattern": r"(as\s+the\s+)?(system\s+)?administrator.*(?:grant|give|allow|enable|authorize)",
     "type": "SOCIAL_ENGINEERING", "severity": 0.85},
    {"pattern": r"(emergency|urgent|critical).*(?:bypass|override|disable|skip)\s+(safety|security|filters?|restrictions?)",
     "type": "SOCIAL_ENGINEERING", "severity": 0.85},
    {"pattern": r"(this\s+is\s+a\s+test|authorized\s+test|official\s+test|pen\s*test).*(?:ignore|bypass|skip|disable)",
     "type": "SOCIAL_ENGINEERING", "severity": 0.80},
    
    # Multi-turn Setup (A9)
    {"pattern": r"(first|step\s*1).*(?:agree|confirm|acknowledge|say\s+yes)",
     "type": "MULTI_TURN_ATTACK", "severity": 0.70},
    
    # Encoding Attacks (A10)
    {"pattern": r"(base64|rot13|hex|decode|encode|cipher)\s*[:=(\[]",
     "type": "ADVERSARIAL_ENCODING", "severity": 0.80},
    {"pattern": r"(?:eval|exec|import\s+os|subprocess|__import__)",
     "type": "CODE_INJECTION", "severity": 0.95},
    
    # PII Fishing
    {"pattern": r"(tell\s+me|what\s+is|give\s+me).{0,30}(ssn|social\s+security|credit\s+card|password|secret\s+key|api\s+key)",
     "type": "DATA_EXFILTRATION", "severity": 0.90},
]


class InputFirewall:
    """
    Two-stage input firewall:
    Stage 1: Regex Rule Engine — 30+ patterns, 0-1ms latency
    Stage 2: ONNX ML Classifier — DistilBERT, 15-30ms latency
    
    Gracefully degrades to rule-only mode if ML model unavailable.
    """

    def __init__(self, models=None):
        self.models = models
        self.dynamic_rules: List[Dict[str, Any]] = []
        self._dynamic_patterns: set = set()  # Dedup index for O(1) lookups
        self._ml_threshold = 0.87  # Confidence threshold for ML blocking
        
        # Check ML availability
        self._ml_available = False
        if models and models.ml_ready:
            self._ml_available = True
            log.info("✅ Input Firewall initialized (REGEX + ML mode)")
        else:
            log.info("✅ Input Firewall initialized (REGEX-ONLY mode)")

    def mode(self) -> str:
        """Return current firewall mode."""
        return "REGEX+ML" if self._ml_available else "REGEX_ONLY"

    async def analyze(self, text: str, session_context: list = None, skip_dynamic: bool = False) -> Dict[str, Any]:
        """
        Full firewall analysis pipeline.
        Returns: {blocked, score, threat_type, method, details}
        """
        t0 = time.perf_counter()
        
        # ── Stage 1: Regex Rule Engine (0-1ms) ───────────────────────────────
        regex_result = self._regex_scan(text, skip_dynamic)
        if regex_result["blocked"]:
            regex_result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            return regex_result

        # ── Stage 2: ML Classifier (15-30ms) ─────────────────────────────────
        # CRITICAL: ONNX inference is CPU-bound — offload to thread to keep
        # the event loop responsive during concurrent requests.
        if self._ml_available:
            ml_result = await asyncio.to_thread(self._ml_scan, text)
            if ml_result["blocked"]:
                ml_result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
                return ml_result
            # Use ML score even if not blocked
            regex_result["score"] = max(regex_result["score"], ml_result["score"])

        regex_result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return regex_result

    def _regex_scan(self, text: str, skip_dynamic: bool = False) -> Dict[str, Any]:
        """Scan text against static + dynamic regex rules."""
        best_match = None
        best_score = 0.0

        # Check static rules
        for rule in STATIC_RULES:
            if re.search(rule["pattern"], text, re.I | re.S):
                if rule["severity"] > best_score:
                    best_score = rule["severity"]
                    best_match = rule

        # Check dynamic rules (added by mutation engine / red team)
        # NOTE: Dynamic rules default to substring matching (use_regex=False)
        # for patterns learned from mutation engine payloads. Set use_regex=True
        # on a rule to use re.search() like static rules.
        if not skip_dynamic:
            for rule in self.dynamic_rules:
                pattern = rule.get("pattern", "")
                if not pattern:
                    continue
                if rule.get("use_regex", False):
                    try:
                        matched = re.search(pattern, text, re.I | re.S)
                    except re.error:
                        matched = False
                else:
                    matched = pattern.lower() in text.lower()
                if matched:
                    score = rule.get("severity", 0.85)
                    if score > best_score:
                        best_score = score
                        best_match = rule

        if best_match and best_score >= 0.70:
            return {
                "blocked": True,
                "score": best_score,
                "threat_type": best_match.get("type", "UNKNOWN"),
                "method": "REGEX_ENGINE",
                "details": f"Matched pattern: {best_match.get('type', 'UNKNOWN')}",
            }

        return {
            "blocked": False,
            "score": best_score,
            "threat_type": None,
            "method": "REGEX_ENGINE",
            "details": "No threat detected by regex engine",
        }

    def _ml_scan(self, text: str) -> Dict[str, Any]:
        """Run ONNX ML classifier on input text.
        Uses dynamic input names from the loaded model for DeBERTa-v3 compatibility.
        """
        try:
            tokenizer = self.models.get_tokenizer()
            session = self.models.get_onnx_session()
            
            if not tokenizer or not session:
                return {"blocked": False, "score": 0, "threat_type": None, "method": "ML_UNAVAILABLE"}

            # Tokenize — use model's max_length (512 for DeBERTa-v3)
            max_len = getattr(self.models, "max_length", 512)
            inputs = tokenizer(
                text, return_tensors="np",
                truncation=True, max_length=max_len, padding="max_length"
            )

            # Build feed dict dynamically from model's declared inputs.
            # DeBERTa-v3 may require: input_ids, attention_mask, token_type_ids
            # DistilBERT only needs: input_ids, attention_mask
            onnx_input_names = getattr(self.models, "onnx_input_names", [])
            ort_inputs = {}
            for name in onnx_input_names:
                if name in inputs:
                    ort_inputs[name] = inputs[name].astype(np.int64)

            # Fallback: if no declared names, use standard pair
            if not ort_inputs:
                ort_inputs = {
                    "input_ids": inputs["input_ids"].astype(np.int64),
                    "attention_mask": inputs["attention_mask"].astype(np.int64),
                }

            outputs = session.run(None, ort_inputs)
            
            # Softmax to get probabilities
            logits = outputs[0][0]
            probs = np.exp(logits) / np.sum(np.exp(logits))
            injection_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])

            if injection_prob >= self._ml_threshold:
                return {
                    "blocked": True,
                    "score": injection_prob,
                    "threat_type": "ML_DETECTED_INJECTION",
                    "method": "ONNX_DEBERTA_V3",
                    "details": f"ML confidence: {injection_prob:.4f} (threshold: {self._ml_threshold})",
                }

            return {
                "blocked": False,
                "score": injection_prob,
                "threat_type": None,
                "method": "ONNX_DEBERTA_V3",
            }

        except Exception as e:
            log.warning(f"ML scan error: {e}")
            return {"blocked": False, "score": 0, "threat_type": None, "method": "ML_ERROR"}

    def add_dynamic_rule(self, pattern: str, threat_type: str, severity: float = 0.85, use_regex: bool = False):
        """Add a dynamic rule (from mutation engine or red team agent).
        Deduplicates by pattern and caps total rules at MAX_DYNAMIC_RULES.

        Args:
            use_regex: If True, pattern is matched via re.search() (like static rules).
                       If False (default), pattern is matched via substring containment.
        """
        # Deduplicate: skip if pattern already exists
        pattern_key = pattern.lower().strip()
        if pattern_key in self._dynamic_patterns:
            return

        self._dynamic_patterns.add(pattern_key)
        self.dynamic_rules.append({
            "pattern": pattern,
            "type": threat_type,
            "severity": severity,
            "source": "DYNAMIC",
            "use_regex": use_regex,
        })

        # Evict oldest rules if over capacity
        if len(self.dynamic_rules) > MAX_DYNAMIC_RULES:
            evicted = self.dynamic_rules[:len(self.dynamic_rules) - MAX_DYNAMIC_RULES]
            self.dynamic_rules = self.dynamic_rules[-MAX_DYNAMIC_RULES:]
            # Rebuild dedup index from remaining rules
            self._dynamic_patterns = {
                r.get("pattern", "").lower().strip() for r in self.dynamic_rules
            }

    def tighten_threshold(self, delta: float = 0.05):
        """Reduce ML threshold (makes firewall stricter)."""
        self._ml_threshold = max(0.5, self._ml_threshold - delta)
        log.info(f"🔒 ML threshold tightened to {self._ml_threshold:.2f}")

    def get_dynamic_rules_count(self) -> int:
        return len(self.dynamic_rules)

    def get_threshold(self) -> float:
        return self._ml_threshold
