import asyncio
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Any, Dict
from utils.logger import logger
from utils.auth import require_dashboard_key
from schemas.agents import AgentStatus

router = APIRouter(prefix="/api/v1/agents", tags=["Agents"])


class SimpleResponse(BaseModel):
    success: bool = True
    message: str = ""


@router.get(
    "/status",
    response_model=AgentStatus,
    summary="Agent system status",
    description="Returns current agent status including pause state and tier.",
    dependencies=[Depends(require_dashboard_key)]
)
async def get_status() -> AgentStatus:
    """Returns battle engine agent status. Always 200."""
    try:
        from agents.battle_engine import battle_engine
        status = battle_engine.get_status()
        return AgentStatus(
            is_paused=status.get("is_paused", False),
            tick=status.get("tick", 0),
            red_tier=status.get("red_tier", 1),
            red_strategy=status.get("red_strategy", "NAIVE"),
            uptime_seconds=status.get("uptime_seconds", 0.0)
        )
    except Exception as e:
        logger.warning(f"GET /agents/status error: {type(e).__name__}")
        return AgentStatus()


@router.post(
    "/pause",
    response_model=SimpleResponse,
    summary="Pause battle engine",
    description="Pauses the autonomous battle loop. Current tick completes first.",
    dependencies=[Depends(require_dashboard_key)]
)
async def pause_engine() -> SimpleResponse:
    """Pause the battle engine loop."""
    try:
        from agents.battle_engine import battle_engine
        battle_engine.pause()
        logger.info("Battle engine paused via API")
        return SimpleResponse(success=True, message="Battle engine paused")
    except Exception as e:
        logger.error(f"POST /agents/pause error: {type(e).__name__}")
        return SimpleResponse(success=False, message=f"Pause failed: {type(e).__name__}")


@router.post(
    "/resume",
    response_model=SimpleResponse,
    summary="Resume battle engine",
    description="Resumes the autonomous battle loop.",
    dependencies=[Depends(require_dashboard_key)]
)
async def resume_engine() -> SimpleResponse:
    """Resume the battle engine loop."""
    try:
        from agents.battle_engine import battle_engine
        battle_engine.resume()
        logger.info("Battle engine resumed via API")
        return SimpleResponse(success=True, message="Battle engine resumed")
    except Exception as e:
        logger.error(f"POST /agents/resume error: {type(e).__name__}")
        return SimpleResponse(success=False, message=f"Resume failed: {type(e).__name__}")


@router.post(
    "/cycle",
    summary="Force one immediate battle tick",
    description=(
        "Triggers one battle tick immediately regardless of pause state. "
        "Returns battle state after the tick. May take up to 35 seconds "
        "(Gemini call). Always HTTP 200."
    ),
    dependencies=[Depends(require_dashboard_key)]
)
async def force_cycle() -> Dict[str, Any]:
    """Force one immediate battle tick. Returns state after tick."""
    try:
        from agents.battle_engine import battle_engine
        logger.info("Forced battle cycle triggered via API")
        result = await battle_engine.cycle()
        return result
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.error(f"POST /agents/cycle error: {type(e).__name__}")
        return {
            "success": False,
            "error": type(e).__name__,
            "message": "Cycle failed -- check server logs"
        }

