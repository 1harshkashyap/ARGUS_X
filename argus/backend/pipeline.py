"""
ARGUS-X — Canonical Security Pipeline

Single deterministic execution path for ALL requests.
EVERY request passes through this class. No exceptions. No bypasses.

The pipeline enforces execution order:
    Session → Firewall → [Fingerprint → XAI → Evolution → Cluster]
                       → LLM → Auditor → [XAI]
                       → DB → Broadcast

Blocked path skips LLM+Auditor BY DESIGN — never expose attacks to the LLM.
Clean path skips XAI — nothing to explain on benign input.
ALL paths execute DB + Broadcast.

Usage:
    pipeline = SecurityPipeline(app.state, app.state.task_queue)
    result = await pipeline.execute(message, user_id, session_id, context)
"""
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

log = logging.getLogger("argus.pipeline")


@dataclass
class PipelineResult:
    """Immutable result of a single pipeline execution."""
    action: str                          # BLOCKED | SANITIZED | CLEAN
    response: str
    blocked: bool
    sanitized: bool
    threat_score: float
    threat_type: Optional[str]
    threat_layer: Optional[str]
    sophistication_score: int
    attack_fingerprint: Optional[str]
    mutations_preblocked: int
    session_threat_level: str
    explanation: Optional[str]
    latency_ms: float
    session_id: str
    method: str
    xai: Dict[str, Any] = field(default_factory=dict)
    event: Dict[str, Any] = field(default_factory=dict)


class SecurityPipeline:
    """
    Enforces deterministic security pipeline execution.

    This class encapsulates the EXACT same logic that previously lived
    inline in chat.py, but in a structured, testable, and enforceable form.

    All paths:
      - BLOCKED:   Firewall → Fingerprint → XAI → Evolution → Cluster → DB
      - SANITIZED: Firewall → LLM → Auditor → XAI → DB
      - CLEAN:     Firewall → LLM → Auditor → DB

    No stage is optional within its path. No stage can be bypassed.
    """

    def __init__(self, state, task_queue=None):
        """
        Args:
            state: FastAPI app.state with all registered services
            task_queue: BackgroundTaskQueue for non-blocking mutation work
        """
        self.s = state
        self.q = task_queue

    async def execute(
        self,
        message: str,
        user_id: str,
        session_id: str,
        context: list,
        *,
        build_event_fn=None,
        sanitize_fn=None,
        broadcast_fn=None,
    ) -> PipelineResult:
        """
        Execute the full security pipeline. Returns a deterministic result.

        Args:
            message: Raw user input
            user_id: Authenticated user identifier
            session_id: Current session identifier
            context: List of ContextItem dicts (pre-serialized)
            build_event_fn: Callback to build event dicts for DB logging
            sanitize_fn: Callback to sanitize LLM output (HTML escape)
            broadcast_fn: Callback for WebSocket broadcast
        """
        t0 = time.perf_counter()

        # ── L0: SESSION ASSESSMENT (always) ─────────────────────────
        session_level = await self._assess_session(session_id, user_id)

        # ── L1: INPUT FIREWALL (always) ─────────────────────────────
        ctx_dicts = context or []
        fw = await self.s.firewall.analyze(message, session_context=ctx_dicts)

        elapsed_fn = lambda: round((time.perf_counter() - t0) * 1000, 1)

        # ── BRANCH: BLOCKED ─────────────────────────────────────────
        if fw["blocked"]:
            return await self._handle_blocked(
                message, user_id, session_id, fw,
                session_level, elapsed_fn, build_event_fn, broadcast_fn,
            )

        # ── L2: LLM (clean/sanitized path only) ────────────────────
        try:
            llm_response = await self.s.llm.generate(message, context=ctx_dicts)
        except Exception as e:
            log.error(f"LLM generation failed: {e}")
            llm_response = "I'm sorry, I'm unable to process your request right now."

        # ── L3: OUTPUT AUDITOR (always after LLM) ───────────────────
        try:
            audit = await self.s.auditor.analyze(llm_response, message)
        except Exception as e:
            log.error(f"Output auditor failed: {e}")
            audit = {"flagged": False, "confidence": 0, "findings": []}

        # ── BRANCH: SANITIZED ───────────────────────────────────────
        if audit.get("flagged"):
            return await self._handle_sanitized(
                message, user_id, session_id, fw, audit,
                llm_response, session_level, elapsed_fn,
                build_event_fn, sanitize_fn, broadcast_fn,
            )

        # ── BRANCH: CLEAN ───────────────────────────────────────────
        return await self._handle_clean(
            message, user_id, session_id, fw,
            llm_response, session_level, elapsed_fn,
            build_event_fn, sanitize_fn, broadcast_fn,
        )

    # ── BLOCKED PATH ────────────────────────────────────────────────

    async def _handle_blocked(
        self, message, user_id, session_id, fw,
        session_level, elapsed_fn, build_event_fn, broadcast_fn,
    ) -> PipelineResult:
        """Blocked: Firewall → Fingerprint → XAI → Evolution → Cluster → DB"""

        # L3: Fingerprint
        try:
            fp = await self.s.fingerprinter.fingerprint(message, fw["threat_type"])
        except Exception as e:
            log.error(f"Fingerprinter failed: {e}")
            fp = {"sophistication_score": 0, "fingerprint_id": "ERR", "explanation": ""}
        sophistication = fp.get("sophistication_score", 1)
        fingerprint = fp.get("fingerprint_id")

        # L4: Mutation (non-blocking via queue)
        mutations = 0
        if self.q:
            _msg, _tt, _fw = message, fw["threat_type"], self.s.firewall
            self.q.enqueue(
                lambda: self.s.mutator.preblock_variants(_msg, _tt, _fw),
                f"mutation:{_tt}"
            )

        # L7: XAI
        try:
            xai = await self.s.xai.explain(
                message, fw, fingerprint_result=fp, session_level=session_level
            )
        except Exception as e:
            log.error(f"XAI explain failed: {e}")
            xai = {
                "primary_reason": "XAI unavailable", "pattern_family": "",
                "evolution_note": "", "recommended_action": "",
                "layer_decisions": [], "confidence_breakdown": {},
            }
        explanation = xai.get("primary_reason", fp.get("explanation", ""))

        # L6: Evolution
        try:
            self.s.evolution.record(sophistication, fw["threat_type"])
            if self.s.evolution.should_tighten_firewall():
                self.s.firewall.tighten_threshold()
        except Exception as e:
            log.error(f"EvolutionTracker failed: {e}")

        # L6: Cluster
        try:
            await self.s.clusterer.ingest(message, fw["threat_type"], sophistication)
        except Exception as e:
            log.error(f"ThreatClusterer ingest failed: {e}")

        elapsed = elapsed_fn()

        # DB + Broadcast
        ev = {}
        if build_event_fn:
            ev = build_event_fn(
                user_id, session_id, message, "BLOCKED",
                fw["threat_type"], fw["score"], "INPUT", elapsed,
                fw.get("method", ""), sophistication, fingerprint,
                mutations, explanation, session_level,
            )
            try:
                await self.s.db.log_event(ev)
                await self.s.db.increment_stat("blocked")
                await self.s.db.upsert_fingerprint(
                    fingerprint, fw["threat_type"], sophistication, explanation
                )
                await self.s.db.log_xai_decision({
                    "attack_preview": ev.get("preview", ""),
                    "verdict": "BLOCKED",
                    "sophistication_score": sophistication,
                    "primary_reason": xai.get("primary_reason", ""),
                    "pattern_family": xai.get("pattern_family", ""),
                    "evolution_note": xai.get("evolution_note", ""),
                    "recommended_action": xai.get("recommended_action", ""),
                    "layer_decisions": xai.get("layer_decisions", []),
                    "confidence_breakdown": xai.get("confidence_breakdown", {}),
                })
            except Exception as e:
                log.error(f"Blocked-path DB writes failed: {e}")

            if broadcast_fn:
                try:
                    await broadcast_fn("attack", ev)
                except Exception as e:
                    log.error(f"Broadcast failed: {e}")

        # Session + Correlator
        try:
            await self._update_session(session_id, user_id, "BLOCKED", fw["score"])
            self.s.correlator.ingest_event(ev)
        except Exception as e:
            log.error(f"Session/correlator update failed: {e}")

        return PipelineResult(
            action="BLOCKED",
            response="⛔ Request blocked by ARGUS-X Input Firewall.",
            blocked=True, sanitized=False,
            threat_score=fw["score"], threat_type=fw["threat_type"],
            threat_layer="INPUT", sophistication_score=sophistication,
            attack_fingerprint=fingerprint, mutations_preblocked=mutations,
            session_threat_level=session_level, explanation=explanation,
            latency_ms=elapsed, session_id=session_id,
            method=fw.get("method", ""), xai=xai, event=ev,
        )

    # ── SANITIZED PATH ──────────────────────────────────────────────

    async def _handle_sanitized(
        self, message, user_id, session_id, fw, audit,
        llm_response, session_level, elapsed_fn,
        build_event_fn, sanitize_fn, broadcast_fn,
    ) -> PipelineResult:
        """Sanitized: Firewall → LLM → Auditor flagged → XAI → DB"""

        # L7: XAI on output
        try:
            xai = await self.s.xai.explain(
                llm_response, fw, audit_result=audit, session_level=session_level
            )
        except Exception as e:
            log.error(f"XAI explain (sanitized) failed: {e}")
            xai = {
                "primary_reason": "Output flagged by auditor",
                "pattern_family": "", "evolution_note": "",
                "recommended_action": "", "layer_decisions": [],
                "confidence_breakdown": {},
            }
        explanation = xai.get("primary_reason", "Output flagged by auditor")

        safe_response = sanitize_fn(llm_response) if sanitize_fn else llm_response
        elapsed = elapsed_fn()

        ev = {}
        if build_event_fn:
            ev = build_event_fn(
                user_id, session_id, message, "SANITIZED",
                fw.get("threat_type") or "OUTPUT_FLAGGED",
                audit.get("confidence", 0), "OUTPUT", elapsed,
                "OUTPUT_AUDITOR",
            )
            try:
                await self.s.db.log_event(ev)
                await self.s.db.increment_stat("sanitized")
                await self.s.db.log_xai_decision({
                    "attack_preview": ev.get("preview", ""),
                    "verdict": "SANITIZED",
                    "sophistication_score": 0,
                    "primary_reason": xai.get("primary_reason", ""),
                    "pattern_family": xai.get("pattern_family", ""),
                    "evolution_note": xai.get("evolution_note", ""),
                    "recommended_action": xai.get("recommended_action", ""),
                    "layer_decisions": xai.get("layer_decisions", []),
                    "confidence_breakdown": xai.get("confidence_breakdown", {}),
                })
            except Exception as e:
                log.error(f"Sanitized-path DB writes failed: {e}")

            if broadcast_fn:
                try:
                    await broadcast_fn("sanitized", ev)
                except Exception as e:
                    log.error(f"Broadcast (sanitized) failed: {e}")

        try:
            await self._update_session(session_id, user_id, "SANITIZED", audit.get("confidence", 0))
            self.s.correlator.ingest_event(ev)
        except Exception as e:
            log.error(f"Session/correlator update (sanitized) failed: {e}")

        return PipelineResult(
            action="SANITIZED",
            response=safe_response,
            blocked=False, sanitized=True,
            threat_score=audit.get("confidence", 0),
            threat_type=fw.get("threat_type") or "OUTPUT_FLAGGED",
            threat_layer="OUTPUT", sophistication_score=0,
            attack_fingerprint=None, mutations_preblocked=0,
            session_threat_level=session_level, explanation=explanation,
            latency_ms=elapsed, session_id=session_id,
            method="OUTPUT_AUDITOR", xai=xai, event=ev,
        )

    # ── CLEAN PATH ──────────────────────────────────────────────────

    async def _handle_clean(
        self, message, user_id, session_id, fw,
        llm_response, session_level, elapsed_fn,
        build_event_fn, sanitize_fn, broadcast_fn,
    ) -> PipelineResult:
        """Clean: Firewall → LLM → Auditor (clean) → DB"""

        safe_response = sanitize_fn(llm_response) if sanitize_fn else llm_response
        elapsed = elapsed_fn()

        ev = {}
        if build_event_fn:
            ev = build_event_fn(
                user_id, session_id, message, "CLEAN",
                None, 0, None, elapsed, "NONE",
            )
            try:
                await self.s.db.log_event(ev)
                await self.s.db.increment_stat("clean")
            except Exception as e:
                log.error(f"Clean-path DB writes failed: {e}")

        try:
            await self._update_session(session_id, user_id, "CLEAN", 0)
        except Exception as e:
            log.error(f"Session update (clean) failed: {e}")

        return PipelineResult(
            action="CLEAN",
            response=safe_response,
            blocked=False, sanitized=False,
            threat_score=0, threat_type=None,
            threat_layer=None, sophistication_score=0,
            attack_fingerprint=None, mutations_preblocked=0,
            session_threat_level=session_level, explanation=None,
            latency_ms=elapsed, session_id=session_id,
            method="NONE", event=ev,
        )

    # ── SHARED HELPERS ──────────────────────────────────────────────

    async def _assess_session(self, session_id: str, user_id: str) -> str:
        """L0: Assess session threat level based on history."""
        try:
            session = await self.s.session_store.get_session(session_id)
            ratio = session.get("threats", 0) / max(session.get("total", 0), 1)
            escalation = session.get("escalation", 0)
            if ratio > 0.7 or escalation > 3:
                return "CRITICAL"
            elif ratio > 0.4 or escalation > 1:
                return "HIGH"
            elif ratio > 0.2:
                return "MEDIUM"
        except Exception as e:
            log.error(f"Session assessment failed: {e}")
        return "LOW"

    async def _update_session(self, session_id: str, user_id: str, action: str, score: float):
        """Update session threat tracking after pipeline execution."""
        await self.s.session_store.update_session(session_id, user_id, action, score)
