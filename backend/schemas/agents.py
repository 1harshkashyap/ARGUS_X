from pydantic import BaseModel, Field
from typing import Optional


class BattleState(BaseModel):
    """Current state of the autonomous battle engine."""
    tick: int = 0
    red_attacks: int = 0
    red_bypasses: int = 0
    blue_blocks: int = 0
    blue_patches: int = 0
    red_tier: int = Field(default=1, ge=1, le=5)
    red_strategy: str = "NAIVE"
    current_attack_preview: str = ""
    last_attack_result: str = "BLOCKED"
    updated_at: Optional[str] = None


class AgentStatus(BaseModel):
    """Current status of the battle agents."""
    is_paused: bool = False
    tick: int = 0
    red_tier: int = Field(default=1, ge=1, le=5)
    red_strategy: str = "NAIVE"
    uptime_seconds: float = 0.0


