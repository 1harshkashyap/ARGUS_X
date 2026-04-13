"""
ARGUS-X — Red Team Router
Manual attack testing endpoint for the dashboard console.
"""
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
import time, uuid

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


class RedTeamRequest(BaseModel):
    message: str = Field(..., max_length=5000, description="Attack payload (max 5000 chars)")
    attack_type: Optional[str] = "MANUAL"
    tier: int = Field(default=1, ge=1, le=5, description="Escalation tier (1-5)")


@router.post("/redteam")
@_rate_limit("30/minute")
async def redteam_test(req: RedTeamRequest, request: Request):
    """
    Manual red team test — run an attack against the live firewall.
    Returns detailed security analysis without going through LLM.
    """
    t0 = time.perf_counter()
    app = request.app

    # Run through firewall
    fw_result = await app.state.firewall.analyze(req.message)
    
    # Fingerprint
    threat_type = fw_result.get("threat_type") or req.attack_type
    fp = await app.state.fingerprinter.fingerprint(req.message, threat_type)
    
    # If blocked, run mutations
    mutations = 0
    if fw_result["blocked"]:
        mutations = await app.state.mutator.preblock_variants(
            req.message, threat_type, app.state.firewall
        )

    # XAI explanation
    xai = await app.state.xai.explain(
        req.message, fw_result, fingerprint_result=fp
    )

    elapsed = (time.perf_counter() - t0) * 1000

    # Feed into correlator for campaign detection
    action = "BLOCKED" if fw_result["blocked"] else "CLEAN"
    app.state.correlator.ingest_event({
        "ts": datetime.utcnow().isoformat() + "Z",
        "action": action,
        "threat_type": threat_type,
        "fingerprint": fp.get("fingerprint_id"),
        "user_id": "redteam",
        "session_id": "redteam-" + str(uuid.uuid4())[:4],
        "score": fw_result.get("score", 0),
        "preview": (req.message[:80] + "…") if len(req.message) > 80 else req.message,
    })

    return {
        "blocked": fw_result["blocked"],
        "sanitized": False,
        "threat_score": fw_result.get("score", 0),
        "score": fw_result.get("score", 0),
        "threat_type": threat_type,
        "method": fw_result.get("method", ""),
        "sophistication_score": fp.get("sophistication_score", 0),
        "fingerprint_id": fp.get("fingerprint_id", ""),
        "attack_fingerprint": fp.get("fingerprint_id", ""),
        "explanation": fp.get("explanation", ""),
        "mutations_preblocked": mutations,
        "xai": xai,
        "latency_ms": round(elapsed, 1),
    }
