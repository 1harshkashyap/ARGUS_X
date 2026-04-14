"""
ARGUS-X — Knowledge Router
Endpoints: /clusters, /fingerprints, /campaigns, /evolution
"""
from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/clusters")
async def get_clusters(request: Request):
    """Get current threat cluster summary."""
    app = request.app
    if hasattr(app.state, 'clusterer'):
        return app.state.clusterer.get_cluster_summary()
    return {"total_attacks_ingested": 0, "total_clusters": 0, "clusters": [], "mode": "UNAVAILABLE"}


@router.get("/fingerprints")
async def get_fingerprints(request: Request, limit: int = 20):
    """Get top attack fingerprints."""
    app = request.app
    return {
        "fingerprints": app.state.fingerprinter.top_fingerprints(limit),
        "correlator_fingerprints": app.state.correlator.get_top_fingerprints(limit),
    }


@router.get("/campaigns")
async def get_campaigns(request: Request):
    """Get active campaigns."""
    app = request.app
    return {
        "active": app.state.correlator.get_active_campaigns(),
        "velocity": app.state.correlator.get_threat_velocity(),
    }


@router.get("/evolution")
async def get_evolution(request: Request):
    """Get sophistication evolution report."""
    app = request.app
    if hasattr(app.state, 'evolution'):
        return app.state.evolution.get_evolution_report()
    return {"current_avg": 0, "trend": "STABLE", "threat_level": 1, "data_points": 0}
