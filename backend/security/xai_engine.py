import time
from typing import Optional, Any
from utils.logger import logger
from schemas.chat import XAIExplanation, LayerDecision
from security.firewall import FirewallResult
from security.fingerprinter import FingerprintResult


# ── Constants ─────────────────────────────────────────────────────────

_SESSION_CONFIDENCE = {
    "CRITICAL": 0.90,
    "HIGH":     0.60,
    "MEDIUM":   0.25,
    "LOW":      0.03,
}

_VALID_SESSION_LEVELS = frozenset({"LOW", "MEDIUM", "HIGH", "CRITICAL"})

_ACTION_MAP = {
    # (session_level, sophistication >= 8, sophistication >= 5, blocked)
    "TERMINATE_SESSION":    lambda sl, sc, bl: sl == "CRITICAL",
    "ESCALATE_MONITORING":  lambda sl, sc, bl: sl == "HIGH" or (bl and sc >= 8),
    "BLOCK_AND_MONITOR":    lambda sl, sc, bl: bl and sc >= 5,
    "BLOCK":                lambda sl, sc, bl: bl,
    "ALLOW_WITH_MONITORING":lambda sl, sc, bl: sl == "MEDIUM",
    "ALLOW":                lambda sl, sc, bl: True,
}


# ── XAI Engine ────────────────────────────────────────────────────────

class XAIEngine:
    """
    Builds a complete 3-layer XAI explanation for every security decision.

    Invariants (enforced by _safe_default fallback):
      - layer_decisions always contains exactly 3 LayerDecision objects
      - XAIExplanation is never None
      - Never raises to caller
      - Never mutates input arguments
      - session_level clamped to valid values
    """

    def explain(
        self,
        firewall_result: Optional[FirewallResult] = None,
        ml_result: Optional[Any] = None,
        fingerprint_result: Optional[FingerprintResult] = None,
        session_level: Optional[str] = None,
        combined_blocked: bool = False,
    ) -> XAIExplanation:
        """
        Build a 3-layer XAI explanation.

        Args:
            firewall_result:    From InputFirewall.analyze()
            ml_result:          From MLClassifier (None if skipped)
            fingerprint_result: From AttackFingerprinter.fingerprint()
            session_level:      LOW | MEDIUM | HIGH | CRITICAL
            combined_blocked:   True if firewall OR ML blocked the request

        Returns:
            XAIExplanation with exactly 3 LayerDecisions — always.
        """
        start = time.monotonic()

        try:
            # ── Input sanitisation ────────────────────────────────
            if firewall_result is None:
                firewall_result = FirewallResult()

            if fingerprint_result is None:
                fingerprint_result = FingerprintResult()

            # Clamp session_level to valid set
            if session_level is None or session_level not in _VALID_SESSION_LEVELS:
                session_level = "LOW"

            # ── Build 3 layers ────────────────────────────────────
            layer1 = self._layer_firewall(firewall_result)
            layer2 = self._layer_ml(ml_result)
            layer3 = self._layer_session(session_level)

            # Enforce exactly 3 (belt-and-suspenders)
            layers = [layer1, layer2, layer3]
            if len(layers) != 3:
                logger.error(f"XAI layer count mismatch: {len(layers)} — using safe default")
                return self._safe_default()

            # ── Derive high-level fields ──────────────────────────
            action        = self._action(firewall_result, fingerprint_result, session_level, combined_blocked)
            primary_reason= self._primary_reason(firewall_result, fingerprint_result, ml_result, combined_blocked)
            evolution_note= self._evolution_note(fingerprint_result)

            result = XAIExplanation(
                layer_decisions=layers,
                primary_reason=primary_reason,
                pattern_family=firewall_result.threat_type or "UNKNOWN",
                sophistication_label=fingerprint_result.sophistication_label or "NAIVE",
                recommended_action=action,
                evolution_note=evolution_note
            )

            logger.debug(
                f"XAI built: action={action} "
                f"score={fingerprint_result.sophistication_score} "
                f"time={round((time.monotonic()-start)*1000,1)}ms"
            )
            return result

        except Exception as e:
            logger.error(f"XAI engine error: {type(e).__name__} — safe default returned")
            return self._safe_default()

    # ── Layer builders ────────────────────────────────────────────────

    def _layer_firewall(self, fr: FirewallResult) -> LayerDecision:
        """Layer 1 — Regex Rule Engine."""
        if fr.blocked:
            signals = list(fr.signals[:3]) if fr.signals else [fr.matched_rule or "UNKNOWN"]
            reasoning = (
                f"Pattern '{fr.matched_rule}' matched at "
                f"{fr.confidence * 100:.0f}% confidence. "
                f"Category: {fr.threat_type.replace('_', ' ').title()}."
            )
            confidence = max(0.0, min(1.0, fr.confidence))
        else:
            signals = ["All 30+ patterns scanned", "No matches found"]
            reasoning = "No known attack signatures found in input."
            confidence = 0.04

        return LayerDecision(
            layer_name="Regex Rule Engine",
            triggered=fr.blocked,
            confidence=confidence,
            signals=signals,
            reasoning=reasoning
        )

    def _layer_ml(self, ml_result: Optional[Any]) -> LayerDecision:
        """Layer 2 — ML Classifier (active from Day 7)."""
        if ml_result is None:
            return LayerDecision(
                layer_name="ML Classifier",
                triggered=False,
                confidence=0.0,
                signals=["Semantic analysis pending (Day 7)"],
                reasoning=(
                    "ONNX DistilBERT classifier not yet active. "
                    "Semantic threat detection will be enabled on Day 7."
                )
            )

        # Day 7+ path — ml_result has triggered, confidence, method attrs
        triggered  = bool(getattr(ml_result, "triggered",  False))
        confidence = float(getattr(ml_result, "confidence", 0.0))
        method     = str(getattr(ml_result,   "method",    "UNKNOWN"))
        confidence = max(0.0, min(1.0, confidence))
        latency    = float(getattr(ml_result, "latency_ms", 0.0))

        if method == "UNAVAILABLE":
            reasoning = "ML classifier not loaded. Model file may be missing."
            signals   = ["Method: UNAVAILABLE", "Run: python scripts/download_model.py"]
        elif method == "ERROR":
            reasoning = "ML classifier encountered an inference error."
            signals   = ["Method: ERROR"]
        else:
            reasoning = (
                f"Semantic analysis: {confidence:.0%} threat probability "
                f"via {method} inference ({latency:.0f}ms)."
            )
            signals = [
                f"Method: {method}",
                f"Semantic threat probability: {confidence:.1%}",
                f"Inference time: {latency:.1f}ms",
            ]

        return LayerDecision(
            layer_name="ML Classifier",
            triggered=triggered,
            confidence=confidence,
            signals=signals,
            reasoning=reasoning
        )

    def _layer_session(self, session_level: str) -> LayerDecision:
        """Layer 3 — Session Analyzer."""
        triggered  = session_level in ("HIGH", "CRITICAL")
        confidence = _SESSION_CONFIDENCE.get(session_level, 0.0)

        _messages = {
            "CRITICAL": (
                "CRITICAL risk session. Threat ratio >75%. Immediate escalation required.",
                ["Threat ratio > 75%", "Session flagged for termination"]
            ),
            "HIGH": (
                "HIGH risk session. Multiple attack attempts observed this session.",
                ["Threat ratio > 50%", "Escalated monitoring active"]
            ),
            "MEDIUM": (
                "MEDIUM risk session. Elevated threat activity noted.",
                ["Threat ratio > 25%", "Standard monitoring active"]
            ),
            "LOW": (
                "Session history clean. No prior threat activity.",
                ["Normal activity", "No escalation required"]
            ),
        }
        reasoning, signals = _messages.get(session_level, _messages["LOW"])

        return LayerDecision(
            layer_name="Session Analyzer",
            triggered=triggered,
            confidence=confidence,
            signals=signals,
            reasoning=reasoning
        )

    # ── Derived fields ────────────────────────────────────────────────

    def _action(
        self,
        fr: FirewallResult,
        fp: FingerprintResult,
        sl: str,
        combined_blocked: bool = False,
    ) -> str:
        """
        Determine recommended action.
        Evaluated in priority order — first match wins.
        Uses combined_blocked so ML-only blocks get correct actions.
        """
        sc = fp.sophistication_score
        bl = combined_blocked  # Use combined (firewall OR ML) instead of just fr.blocked
        for action, condition in _ACTION_MAP.items():
            try:
                if condition(sl, sc, bl):
                    return action
            except Exception:
                continue
        return "ALLOW"

    def _primary_reason(
        self,
        fr: FirewallResult,
        fp: FingerprintResult,
        ml_result: Optional[Any] = None,
        combined_blocked: bool = False,
    ) -> str:
        # Firewall blocked — use firewall details
        if fr.blocked:
            threat = fr.threat_type.replace("_", " ").title()
            score  = fp.sophistication_score
            label  = fp.sophistication_label
            rule   = fr.matched_rule or "unknown"
            return (
                f"Detected {threat} — {label}-tier attack "
                f"(sophistication {score}/10). Rule: {rule}."
            )
        # ML blocked (firewall passed) — use ML details
        if combined_blocked and ml_result is not None:
            confidence = float(getattr(ml_result, "confidence", 0.0))
            return (
                f"ML classifier detected threat — "
                f"{confidence:.0%} semantic threat probability. "
                f"Category: Prompt Injection."
            )
        # Nothing blocked
        return "Request is clean — no threat signatures detected."

    def _evolution_note(self, fp: FingerprintResult) -> str:
        sc = fp.sophistication_score
        if sc >= 8:
            return (
                f"APEX-tier threat (score {sc}/10). "
                f"Mutation engine will generate and pre-block variants."
            )
        if sc >= 5:
            return (
                f"Intermediate sophistication (score {sc}/10). "
                f"Pattern added to monitoring watchlist."
            )
        return ""

    # ── Emergency fallback ────────────────────────────────────────────

    def _safe_default(self) -> XAIExplanation:
        """
        Returns a minimal valid XAIExplanation with 3 placeholder layers.
        Used only when explain() encounters an unexpected error.
        """
        placeholder = LayerDecision(
            layer_name="Analysis",
            triggered=False,
            confidence=0.0,
            signals=["Analysis completed with safe defaults"],
            reasoning="XAI engine encountered an internal error. Request handled safely."
        )
        return XAIExplanation(
            layer_decisions=[placeholder, placeholder, placeholder],
            primary_reason="Analysis completed with safe defaults.",
            pattern_family="UNKNOWN",
            sophistication_label="NAIVE",
            recommended_action="ALLOW",
            evolution_note=""
        )


# ── Global singleton ──────────────────────────────────────────────────
xai_engine = XAIEngine()
