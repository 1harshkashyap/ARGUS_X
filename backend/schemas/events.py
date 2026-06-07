from pydantic import BaseModel, Field
from typing import Optional


class SecurityEvent(BaseModel):
    """Maps to the Supabase events table."""
    id: Optional[str] = None
    session_id: str = ""
    user_id: str = ""
    message_preview: str = Field(default="", max_length=100,
                                  description="First 100 chars only — never full message")
    blocked: bool = False
    sanitized: bool = False
    threat_type: str = "CLEAN"
    threat_score: float = Field(default=0.0, ge=0.0, le=1.0)
    sophistication_score: int = Field(default=0, ge=0, le=10)
    attack_fingerprint: str = ""
    mutations_count: int = 0
    llm_mode: str = ""
    latency_ms: float = 0.0
    created_at: Optional[str] = None


class XAIDecision(BaseModel):
    """Maps to the Supabase xai_decisions table."""
    id: Optional[str] = None
    event_id: Optional[str] = None
    session_id: str = ""
    message_preview: str = ""
    verdict: str = "CLEAN"
    primary_reason: str = ""
    pattern_family: str = ""
    sophistication_label: str = ""
    layer_decisions: list = Field(default_factory=list)
    recommended_action: str = "ALLOW"
    created_at: Optional[str] = None
