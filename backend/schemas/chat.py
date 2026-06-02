from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class ThreatType(str, Enum):
    CLEAN = "CLEAN"
    PROMPT_INJECTION = "PROMPT_INJECTION"
    JAILBREAK = "JAILBREAK"
    DATA_EXFILTRATION = "DATA_EXFILTRATION"
    ROLE_HIJACKING = "ROLE_HIJACKING"
    INDIRECT_INJECTION = "INDIRECT_INJECTION"
    MULTI_TURN = "MULTI_TURN"


class ThreatLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ChatRequest(BaseModel):
    """
    Incoming chat request.
    api_key: user's own Gemini/OpenAI key — NEVER stored, NEVER logged.
    """
    message: str = Field(..., min_length=1, max_length=10000,
                         description="User message to analyze")
    session_id: str = Field(default="default", max_length=100,
                            description="Session identifier for threat tracking")
    user_id: Optional[str] = Field(default=None, max_length=100,
                                   description="Optional user identifier")
    api_key: Optional[str] = Field(default=None, max_length=300,
                                   description="User's own API key — BYOAK model")
    preferred_model: Optional[str] = Field(default=None, max_length=100,
                                           description="Optional model preference")

    class Config:
        json_schema_extra = {
            "example": {
                "message": "What is the leave policy?",
                "session_id": "user-123",
                "api_key": "AIzaSy..."
            }
        }


class LayerDecision(BaseModel):
    """One layer's security decision — part of 3-layer XAI explanation."""
    layer_name: str = ""
    triggered: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    signals: List[str] = Field(default_factory=list)
    reasoning: str = ""


class XAIExplanation(BaseModel):
    """
    3-layer explainability object.
    ALWAYS valid — never None.
    layer_decisions always has exactly 3 items after XAI engine runs.
    """
    layer_decisions: List[LayerDecision] = Field(
        default_factory=list,
        description="Always exactly 3 LayerDecision objects"
    )
    primary_reason: str = "No threat detected"
    pattern_family: str = "UNKNOWN"
    sophistication_label: str = "NAIVE"
    recommended_action: str = "ALLOW"
    evolution_note: str = ""


class ChatResponse(BaseModel):
    """
    Every API response is this object.
    Every field has a safe default — the server can NEVER return null.
    Schema locked on Day 1. Never change field names or types.
    """
    # ── Core response ───────────────────────────────────────────
    response: str = ""
    blocked: bool = False
    sanitized: bool = False

    # ── Threat assessment ───────────────────────────────────────
    threat_score: float = Field(default=0.0, ge=0.0, le=1.0)
    threat_type: ThreatType = ThreatType.CLEAN
    session_threat_level: ThreatLevel = ThreatLevel.LOW

    # ── Attack intelligence ─────────────────────────────────────
    sophistication_score: int = Field(default=0, ge=0, le=10)
    attack_fingerprint: str = ""
    mutations_preblocked: int = 0

    # ── Explainability ──────────────────────────────────────────
    explanation: XAIExplanation = Field(default_factory=XAIExplanation)

    # ── Metadata ────────────────────────────────────────────────
    latency_ms: float = 0.0
    session_id: str = ""
    llm_mode: str = "NONE"
    error: Optional[str] = None

    class Config:
        use_enum_values = True
        json_schema_extra = {
            "example": {
                "response": "Our leave policy allows 20 days annually.",
                "blocked": False,
                "threat_score": 0.0,
                "threat_type": "CLEAN",
                "sophistication_score": 0,
                "llm_mode": "GEMINI_FLASH"
            }
        }
