import time
from fastapi import APIRouter, Depends
from utils.logger import logger
from utils.db import get_battle_state
from utils.auth import require_dashboard_key
from schemas.agents import BattleState

router = APIRouter(prefix="/api/v1/battle", tags=["Battle"])


@router.get(
    "/state",
    response_model=BattleState,
    summary="Current battle engine state",
    description=(
        "Returns current battle state from in-memory engine + Supabase. "
        "Always HTTP 200. Safe defaults if unavailable."
    ),
    dependencies=[Depends(require_dashboard_key)]
)
async def get_state() -> BattleState:
    """
    Returns the current battle engine state.
    Prefers in-memory state (most current).
    Falls back to Supabase if engine not initialized.
    """
    try:
        from agents.battle_engine import battle_engine
        state = battle_engine.get_state()
        return BattleState(**state)
    except Exception as e:
        logger.warning(f"Battle state from engine failed: {type(e).__name__} — trying DB")
        try:
            raw = await get_battle_state()
            if raw:
                return BattleState(**raw)
        except Exception as db_e:
            logger.warning(f"Battle state from DB failed: {type(db_e).__name__}")
        return BattleState()
