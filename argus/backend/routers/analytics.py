"""
ARGUS-X — Analytics Router
Endpoints: /stats, /logs, /heatmap
"""
from fastapi import APIRouter, Request, Query
from datetime import datetime, timezone

# Rate limiting (graceful — no-op if slowapi not installed)
try:
    from slowapi import Limiter
    from slowapi.util import get_remote_address
    limiter = Limiter(key_func=get_remote_address)
except ImportError:
    limiter = None


def _rate_limit(limit_string: str):
    if limiter:
        return limiter.limit(limit_string)
    return lambda f: f

router = APIRouter()


@router.get("/analytics/stats")
async def get_stats(request: Request):
    """Get all dashboard stats in one call — shaped for frontend consumption."""
    app = request.app
    
    stats = await app.state.db.get_stats()
    
    # Enrich with live data
    red_status = app.state.red_agent.status()
    blue_status = app.state.blue_agent.status()
    correlator_status = app.state.correlator.status()
    
    # Calculate defense rate
    blocked = stats.get("blocked", 0)
    sanitized = stats.get("sanitized", 0)
    clean = stats.get("clean", 0)
    total = blocked + sanitized + clean
    defense_rate = round(blocked / max(total, 1) * 100, 1)
    protect_pct = round((blocked + sanitized) / max(total, 1) * 100, 1)
    
    # Evolution data
    evolution = app.state.evolution.get_evolution_report() if hasattr(app.state, 'evolution') else {}
    
    # Battle state
    battle_state = app.state.battle.get_state() if hasattr(app.state, 'battle') else {}

    # Mutation count from blue agent
    mutations_total = blue_status.get("mutations_spawned", 0)

    return {
        # Top-level stats for direct frontend access
        "blocked": blocked,
        "sanitized": sanitized,
        "clean": clean,
        "total": total,
        "defense_rate": defense_rate,
        "protect_pct": protect_pct,
        "mutations_preblocked": mutations_total + app.state.firewall.get_dynamic_rules_count(),
        "bypasses_found": stats.get("bypasses_found", 0),
        
        # Nested details
        "stats": stats,
        "agent": red_status,
        "blue_agent": blue_status,
        "battle": battle_state,
        "correlator": correlator_status,
        "evolution": evolution,
        "sparkline_data": app.state.evolution.get_sparkline_data(30) if hasattr(app.state, 'evolution') else [],
        "clusters": app.state.clusterer.get_cluster_summary() if hasattr(app.state, 'clusterer') else {},
        "campaigns": app.state.correlator.get_active_campaigns(),
        "threat_velocity": app.state.correlator.get_threat_velocity(),
        "top_fingerprints": app.state.correlator.get_top_fingerprints(),
        "dynamic_rules_count": app.state.firewall.get_dynamic_rules_count(),
        "firewall_mode": app.state.firewall.mode(),
        "ml_threshold": app.state.firewall.get_threshold(),
        "ts": datetime.now(timezone.utc).isoformat() + "Z",
    }


@router.get("/analytics/logs")
@_rate_limit("60/minute")
async def get_logs(request: Request, limit: int = Query(default=50, ge=1, le=500)):
    """Get recent event logs. Capped at 500 to prevent abuse."""
    events = await request.app.state.db.get_recent_events(limit)
    return {"events": events, "count": len(events)}


@router.get("/analytics/heatmap")
async def get_heatmap(request: Request):
    """Get attack heatmap data (hour × day matrix)."""
    data = await request.app.state.db.get_attack_heatmap()
    return {"heatmap": data}


@router.get("/analytics/summary")
async def get_threat_summary(request: Request):
    """Get hourly threat summary."""
    data = await request.app.state.db.get_threat_summary()
    return {"summary": data}


@router.get("/analytics/orgs")
@_rate_limit("30/minute")
async def get_org_stats(request: Request, limit: int = Query(default=50, ge=1, le=200)):
    """
    Multi-Tenant View — Group attacks by org_id.

    Returns per-org threat counts for the dashboard multi-tenant panel.
    Falls back gracefully if org_id column doesn't exist yet.
    """
    events = await request.app.state.db.get_recent_events(limit)

    org_data: dict[str, dict] = {}
    for ev in events:
        oid = ev.get("org_id", "default") or "default"
        if oid not in org_data:
            org_data[oid] = {"org_id": oid, "total": 0, "blocked": 0, "sanitized": 0, "clean": 0, "threats": []}
        org_data[oid]["total"] += 1
        action = ev.get("action", "")
        if action == "BLOCKED":
            org_data[oid]["blocked"] += 1
        elif action == "SANITIZED":
            org_data[oid]["sanitized"] += 1
        else:
            org_data[oid]["clean"] += 1

        tt = ev.get("threat_type")
        if tt and tt not in org_data[oid]["threats"]:
            org_data[oid]["threats"].append(tt)

    orgs = sorted(org_data.values(), key=lambda x: -x["total"])
    return {"orgs": orgs, "count": len(orgs)}

