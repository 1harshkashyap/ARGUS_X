from pydantic import BaseModel, Field
from typing import List, Optional
from .events import SecurityEvent
from .agents import BattleState


class StatsResponse(BaseModel):
    """System-wide analytics summary."""
    total_events: int = 0
    total_blocked: int = 0
    total_clean: int = 0
    total_bypasses: int = 0
    total_mutations: int = 0
    block_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    avg_sophistication: float = 0.0
    avg_latency_ms: float = 0.0
    active_campaigns: int = 0
    battle_state: Optional[BattleState] = None
    uptime_seconds: float = 0.0


class LogsResponse(BaseModel):
    """Recent security events log."""
    events: List[SecurityEvent] = Field(default_factory=list)
    count: int = 0
    limit: int = 20


class XAIDecisionsResponse(BaseModel):
    """Recent XAI decisions."""
    decisions: list = Field(default_factory=list)
    count: int = 0
    limit: int = 10
