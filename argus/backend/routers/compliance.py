"""
ARGUS-X — Compliance Export Router
GET /compliance/export — One-click structured report for compliance/audit.
Returns JSON that the frontend renders as a printable HTML page.
"""
import html
import hashlib
from datetime import datetime, timezone
from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/compliance/export")
async def export_compliance_report(request: Request, limit: int = 100):
    """
    Generate a structured compliance report from real ARGUS-X data.

    Includes:
      - Summary statistics (total, blocked, sanitized, clean)
      - Defense timeline with timestamps
      - Top threat types distribution
      - XAI decision samples (redacted)
      - System configuration snapshot

    The frontend renders this as a printable HTML page using window.print().
    """
    app = request.app

    # ── Gather data ──────────────────────────────────────────────────────
    stats = await app.state.db.get_stats()
    events = await app.state.db.get_recent_events(limit)
    xai_decisions = await app.state.db.get_recent_xai_decisions(20)

    # ── Summary ──────────────────────────────────────────────────────────
    blocked = stats.get("blocked", 0)
    sanitized = stats.get("sanitized", 0)
    clean = stats.get("clean", 0)
    total = blocked + sanitized + clean

    # ── Threat distribution ──────────────────────────────────────────────
    type_counts: dict[str, int] = {}
    for ev in events:
        tt = ev.get("threat_type")
        if tt:
            type_counts[tt] = type_counts.get(tt, 0) + 1

    top_threats = sorted(type_counts.items(), key=lambda x: -x[1])[:10]

    # ── Timeline (redacted) ──────────────────────────────────────────────
    timeline = []
    for ev in events[:50]:
        preview = ev.get("preview", "")
        # Redact preview for compliance — hash only
        preview_hash = hashlib.sha256(preview.encode()).hexdigest()[:12] if preview else ""
        timeline.append({
            "timestamp": ev.get("created_at", ""),
            "action": ev.get("action", ""),
            "threat_type": ev.get("threat_type"),
            "score": ev.get("score", 0),
            "layer": ev.get("layer", ""),
            "sophistication": ev.get("sophistication_score", 0),
            "fingerprint": ev.get("attack_fingerprint"),
            "preview_hash": preview_hash,
        })

    # ── XAI samples (redacted) ───────────────────────────────────────────
    xai_samples = []
    for dec in xai_decisions[:10]:
        xai_samples.append({
            "verdict": dec.get("verdict", ""),
            "primary_reason": html.escape(str(dec.get("primary_reason", "")), quote=True),
            "pattern_family": dec.get("pattern_family", ""),
            "sophistication_score": dec.get("sophistication_score", 0),
            "recommended_action": dec.get("recommended_action", ""),
            "timestamp": dec.get("created_at", ""),
        })

    # ── System config snapshot ───────────────────────────────────────────
    config = {
        "firewall_mode": app.state.firewall.mode(),
        "ml_threshold": app.state.firewall.get_threshold(),
        "dynamic_rules_count": app.state.firewall.get_dynamic_rules_count(),
        "database_connected": app.state.db.available,
    }

    return {
        "report_type": "ARGUS-X Compliance Report",
        "generated_at": datetime.now(timezone.utc).isoformat() + "Z",
        "version": "3.0.0",
        "summary": {
            "total_events": total,
            "blocked": blocked,
            "sanitized": sanitized,
            "clean": clean,
            "defense_rate": round(blocked / max(total, 1) * 100, 1),
            "protection_rate": round((blocked + sanitized) / max(total, 1) * 100, 1),
        },
        "top_threats": [{"type": t, "count": c} for t, c in top_threats],
        "timeline": timeline,
        "xai_samples": xai_samples,
        "system_config": config,
    }
