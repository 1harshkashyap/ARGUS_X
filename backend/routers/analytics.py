import time
from typing import Optional
from fastapi import APIRouter, Query
from utils.logger import logger
from utils.db import (
    get_stats, get_recent_events,
    get_xai_decisions, get_battle_state
)
from utils.session import session_tracker
from schemas.analytics import StatsResponse, LogsResponse, XAIDecisionsResponse
from schemas.agents import BattleState
from schemas.events import SecurityEvent

router = APIRouter(prefix="/api/v1/analytics", tags=["Analytics"])

# Capture module load time for uptime calculation
_MODULE_START = time.monotonic()


@router.get(
    "/stats",
    response_model=StatsResponse,
    summary="System-wide statistics",
    description=(
        "Running totals, block rate, session stats, battle state. "
        "Always returns HTTP 200. Safe defaults if DB unavailable."
    )
)
async def get_system_stats() -> StatsResponse:
    """
    Aggregate statistics from DB + in-memory session tracker.
    Never raises. Returns zero-value StatsResponse on any error.
    """
    try:
        db_stats       = await get_stats()
        battle_raw     = await get_battle_state()
        session_stats  = session_tracker.get_stats()

        total_events  = int(db_stats.get("total_events",  0))
        total_blocked = int(db_stats.get("total_blocked", 0))
        total_bypasses= int(db_stats.get("total_bypasses",0))
        total_mutations=int(db_stats.get("total_mutations",0))
        total_clean   = max(0, total_events - total_blocked)
        block_rate    = round(total_blocked / total_events, 4) if total_events > 0 else 0.0

        battle: Optional[BattleState] = None
        if battle_raw and isinstance(battle_raw, dict):
            try:
                battle = BattleState(**battle_raw)
            except Exception:
                battle = None

        return StatsResponse(
            total_events=total_events,
            total_blocked=total_blocked,
            total_clean=total_clean,
            total_bypasses=total_bypasses,
            total_mutations=total_mutations,
            block_rate=block_rate,
            avg_sophistication=0.0,
            avg_latency_ms=0.0,
            active_campaigns=0,
            battle_state=battle,
            uptime_seconds=round(time.monotonic() - _MODULE_START, 2)
        )

    except Exception as e:
        logger.error(f"GET /analytics/stats error: {type(e).__name__}")
        return StatsResponse(
            uptime_seconds=round(time.monotonic() - _MODULE_START, 2)
        )


@router.get(
    "/logs",
    response_model=LogsResponse,
    summary="Recent security events",
    description=(
        "Returns the N most recent security events from Supabase. "
        "Always HTTP 200. Empty list if DB unavailable."
    )
)
async def get_logs(
    limit: int = Query(
        default=20, ge=1, le=100,
        description="Number of events to return (1–100)"
    )
) -> LogsResponse:
    """
    Recent security events log.
    Silently returns empty list if DB is down.
    """
    try:
        raw_events = await get_recent_events(limit=limit)
        events: list = []

        for raw in raw_events:
            if not isinstance(raw, dict):
                continue
            try:
                events.append(SecurityEvent(**raw))
            except Exception:
                # Skip malformed rows — never crash on bad DB data
                continue

        return LogsResponse(events=events, count=len(events), limit=limit)

    except Exception as e:
        logger.error(f"GET /analytics/logs error: {type(e).__name__}")
        return LogsResponse(limit=limit)


@router.get(
    "/xai",
    response_model=XAIDecisionsResponse,
    summary="Recent XAI decisions",
    description=(
        "Returns the N most recent XAI layer decisions from Supabase. "
        "Always HTTP 200. Empty list if DB unavailable."
    )
)
async def get_xai(
    limit: int = Query(
        default=10, ge=1, le=50,
        description="Number of XAI decisions to return (1–50)"
    )
) -> XAIDecisionsResponse:
    """
    Recent XAI decisions with full layer breakdowns.
    Silently returns empty list if DB is down.
    """
    try:
        decisions = await get_xai_decisions(limit=limit)
        if not isinstance(decisions, list):
            decisions = []

        return XAIDecisionsResponse(
            decisions=decisions,
            count=len(decisions),
            limit=limit
        )

    except Exception as e:
        logger.error(f"GET /analytics/xai error: {type(e).__name__}")
        return XAIDecisionsResponse(limit=limit)
