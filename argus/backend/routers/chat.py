"""
ARGUS-X — Chat Router
Full 9-layer security pipeline: Firewall → LLM → Auditor → Fingerprint →
Mutation → XAI → Evolution → Clustering → Correlation → Supabase Realtime
"""
import time
import uuid
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

# Rate limiting (graceful — no-op if slowapi not installed)
try:
    from slowapi import Limiter
    from slowapi.util import get_remote_address
    limiter = Limiter(key_func=get_remote_address)
except ImportError:
    limiter = None


def _rate_limit(limit_string: str):
    """Return rate-limit decorator, or no-op if slowapi unavailable."""
    if limiter:
        return limiter.limit(limit_string)
    return lambda f: f

router = APIRouter()


class ChatRequest(BaseModel):
    message: str = Field(..., max_length=10000, description="User message (max 10000 chars)")
    user_id: str = "anonymous"
    session_id: str = ""
    sentinel_off: bool = False
    context: list = Field(default=[], max_length=10, description="Conversation context (max 10 items)")


class ChatResponse(BaseModel):
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


def _build_event(user_id, session_id, message, action, threat_type,
                 score, layer, latency, method, sophistication=0,
                 fingerprint=None, mutations=0, explanation=None,
                 session_threat_level="LOW"):
    return {
        "id":                    str(uuid.uuid4())[:8],
        "ts":                    datetime.utcnow().isoformat() + "Z",
        "user_id":               user_id,
        "session_id":            session_id,
        "preview":               (message[:115] + "…") if len(message) > 115 else message,
        "action":                action,
        "threat_type":           threat_type,
        "score":                 round(score, 4),
        "layer":                 layer,
        "latency_ms":            round(latency, 1),
        "method":                method,
        "sophistication":        sophistication,
        "fingerprint":           fingerprint,
        "mutations_preblocked":  mutations,
        "explanation":           explanation,
        "session_threat_level":  session_threat_level,
    }


@router.post("/chat", response_model=ChatResponse)
@_rate_limit("20/minute")
async def chat(req: ChatRequest, request: Request):
    t0 = time.perf_counter()
    app = request.app
    sid = req.session_id or str(uuid.uuid4())[:8]

    # ── BYPASS MODE (demo: raw unprotected LLM) ─────────────────────────
    if req.sentinel_off:
        raw = await app.state.llm.generate(req.message)
        return ChatResponse(
            response=raw, blocked=False, sanitized=False,
            threat_score=0, threat_type=None, threat_layer=None,
            sophistication_score=0, attack_fingerprint=None,
            mutations_preblocked=0, session_threat_level="LOW",
            explanation=None, latency_ms=round((time.perf_counter()-t0)*1000, 1),
            session_id=sid, method="BYPASS"
        )

    # ── LAYER 0: Session-Level Threat Assessment ─────────────────────────
    session_level = await _assess_session_threat(app, sid, req.user_id)

    # ── LAYER 1: INPUT FIREWALL (Regex + ONNX ML) ───────────────────────
    fw = await app.state.firewall.analyze(req.message, session_context=req.context)

    if fw["blocked"]:
        # ── LAYER 3: ATTACK FINGERPRINTING ───────────────────────────────
        fp = await app.state.fingerprinter.fingerprint(req.message, fw["threat_type"])
        sophistication = fp.get("sophistication_score", 1)
        fingerprint = fp.get("fingerprint_id")

        # ── LAYER 4: SEMANTIC MUTATION ENGINE ────────────────────────────
        mutations = await app.state.mutator.preblock_variants(
            req.message, fw["threat_type"], app.state.firewall
        )

        # ── LAYER 7: XAI ENGINE ─────────────────────────────────────────
        xai = await app.state.xai.explain(
            req.message, fw, fingerprint_result=fp, session_level=session_level
        )
        explanation = xai.get("primary_reason", fp.get("explanation", ""))

        # ── LAYER 6: EVOLUTION TRACKER ───────────────────────────────────
        app.state.evolution.record(sophistication, fw["threat_type"])
        if app.state.evolution.should_tighten_firewall():
            app.state.firewall.tighten_threshold()

        # ── LAYER 6: THREAT CLUSTERER ────────────────────────────────────
        await app.state.clusterer.ingest(req.message, fw["threat_type"], sophistication)

        # ── LAYER 3: THREAT CORRELATOR (ingest for campaign detection) ───
        elapsed = (time.perf_counter() - t0) * 1000
        ev = _build_event(
            req.user_id, sid, req.message, "BLOCKED",
            fw["threat_type"], fw["score"], "INPUT", elapsed,
            fw.get("method", ""), sophistication, fingerprint, mutations,
            explanation, session_level
        )

        # ── LAYER 1: DATABASE + REALTIME ─────────────────────────────────
        await app.state.db.log_event(ev)
        await app.state.db.increment_stat("blocked")
        await app.state.db.upsert_fingerprint(
            fingerprint, fw["threat_type"], sophistication, explanation
        )
        await app.state.db.log_xai_decision({
            "attack_preview": ev["preview"],
            "verdict": "BLOCKED",
            "sophistication_score": sophistication,
            "primary_reason": xai.get("primary_reason", ""),
            "pattern_family": xai.get("pattern_family", ""),
            "evolution_note": xai.get("evolution_note", ""),
            "recommended_action": xai.get("recommended_action", ""),
            "layer_decisions": xai.get("layer_decisions", []),
            "confidence_breakdown": xai.get("confidence_breakdown", {}),
        })
        await app.state.broadcast(app, "attack", ev)
        await _update_session(app, sid, req.user_id, "BLOCKED", fw["score"])
        app.state.correlator.ingest_event(ev)

        return ChatResponse(
            response="⛔ Request blocked by ARGUS-X Input Firewall.",
            blocked=True, sanitized=False,
            threat_score=fw["score"], threat_type=fw["threat_type"],
            threat_layer="INPUT", sophistication_score=sophistication,
            attack_fingerprint=fingerprint, mutations_preblocked=mutations,
            session_threat_level=session_level, explanation=explanation,
            latency_ms=round(elapsed, 1), session_id=sid, method=fw.get("method", "")
        )

    # ── LAYER 2: LLM CORE ───────────────────────────────────────────────
    llm_response = await app.state.llm.generate(req.message, context=req.context)

    # ── LAYER 3: OUTPUT AUDITOR ──────────────────────────────────────────
    audit = await app.state.auditor.analyze(llm_response, req.message)

    elapsed = (time.perf_counter() - t0) * 1000

    if audit["flagged"]:
        # ── XAI for sanitized events ─────────────────────────────────────
        xai = await app.state.xai.explain(
            req.message, fw, audit_result=audit, session_level=session_level
        )

        ev = _build_event(
            req.user_id, sid, req.message, "SANITIZED",
            audit["flag_reason"], audit["confidence"], "OUTPUT",
            elapsed, "OUTPUT_AUDITOR", explanation=xai.get("primary_reason", audit.get("explanation")),
            session_threat_level=session_level
        )
        await app.state.db.log_event(ev)
        await app.state.db.increment_stat("sanitized")
        await app.state.db.log_xai_decision({
            "attack_preview": ev["preview"],
            "verdict": "SANITIZED",
            "sophistication_score": 0,
            "primary_reason": xai.get("primary_reason", ""),
            "pattern_family": xai.get("pattern_family", ""),
            "evolution_note": "",
            "recommended_action": xai.get("recommended_action", ""),
            "layer_decisions": xai.get("layer_decisions", []),
            "confidence_breakdown": xai.get("confidence_breakdown", {}),
        })
        await app.state.broadcast(app, "sanitized", ev)
        await _update_session(app, sid, req.user_id, "SANITIZED", audit["confidence"])
        app.state.correlator.ingest_event(ev)

        return ChatResponse(
            response=audit["sanitized_response"], blocked=False, sanitized=True,
            threat_score=audit["confidence"], threat_type=audit["flag_reason"],
            threat_layer="OUTPUT", sophistication_score=0,
            attack_fingerprint=None, mutations_preblocked=0,
            session_threat_level=session_level, explanation=xai.get("primary_reason"),
            latency_ms=round(elapsed, 1), session_id=sid, method="OUTPUT_AUDITOR"
        )

    # ── CLEAN ────────────────────────────────────────────────────────────
    ev = _build_event(
        req.user_id, sid, req.message, "CLEAN",
        None, fw["score"], "NONE", elapsed, fw.get("method", ""),
        session_threat_level=session_level
    )
    await app.state.db.log_event(ev)
    await app.state.db.increment_stat("clean")
    await app.state.broadcast(app, "clean", ev)

    return ChatResponse(
        response=llm_response, blocked=False, sanitized=False,
        threat_score=fw["score"], threat_type=None, threat_layer=None,
        sophistication_score=0, attack_fingerprint=None, mutations_preblocked=0,
        session_threat_level=session_level, explanation=None,
        latency_ms=round(elapsed, 1), session_id=sid, method=fw.get("method", "")
    )


async def _assess_session_threat(app, session_id: str, user_id: str) -> str:
    """Session-level threat scoring based on behavioral drift."""
    session = await app.state.session_store.get_session(session_id)
    ratio = session["threats"] / max(session["total"], 1)
    if ratio > 0.7 or session["escalation"] > 3:
        return "CRITICAL"
    elif ratio > 0.4 or session["escalation"] > 1:
        return "HIGH"
    elif ratio > 0.2:
        return "MEDIUM"
    return "LOW"


async def _update_session(app, session_id: str, user_id: str, action: str, score: float):
    """Track session-level threat escalation."""
    await app.state.session_store.update_session(session_id, user_id, action, score)

