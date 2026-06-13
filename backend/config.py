from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
from functools import lru_cache


class Settings(BaseSettings):
    # ── LLM Configuration ─────────────────────────────────────────────
    GEMINI_API_KEY: str = Field(default="", description="Free Gemini key from aistudio.google.com")
    OPENAI_API_KEY: str = Field(default="", description="Optional — user provides their own")
    LLM_MODE: str = Field(default="auto", description="auto | gemini | openai | mock")

    # ── Database ──────────────────────────────────────────────────────
    SUPABASE_URL: str = Field(default="", description="Project URL from Supabase dashboard")
    SUPABASE_KEY: str = Field(default="", description="service_role key — NOT the anon key")

    # ── Application ───────────────────────────────────────────────────
    ENVIRONMENT: str = Field(default="development", description="development | production")
    APP_VERSION: str = Field(default="2.0.0")
    APP_NAME: str = Field(default="ARGUS-X")
    DEBUG: bool = Field(default=False)

    # ── Security ──────────────────────────────────────────────────────
    ALLOWED_ORIGINS: str = Field(
        default="http://localhost:3000,http://localhost:5173",
        description="Comma-separated CORS origins"
    )
    MAX_MESSAGE_LENGTH: int = Field(default=10000)
    RATE_LIMIT_PER_MINUTE: int = Field(default=30)
    DASHBOARD_READ_KEY: str = Field(
        default="",
        description="Key for dashboard read endpoints — MUST be set via env var, no default"
    )

    # ── Battle Engine ─────────────────────────────────────────────────
    BATTLE_ENABLED: bool = Field(default=False)
    BATTLE_INTERVAL_SECONDS: int = Field(default=60)

    # ── ML Classifier ─────────────────────────────────────────────────
    ML_ENABLED: bool = Field(default=False)
    FIREWALL_ML_THRESHOLD: float = Field(default=0.87, ge=0.0, le=1.0)
    ONNX_MODEL_PATH: str = Field(default="models/security_classifier.onnx")

    # ── Timeouts (seconds) ────────────────────────────────────────────
    GEMINI_TIMEOUT: float = Field(default=30.0)
    OPENAI_TIMEOUT: float = Field(default=30.0)
    DB_TIMEOUT: float = Field(default=10.0)
    HTTP_TIMEOUT: float = Field(default=5.0)

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT.lower() == "production"

    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT.lower() == "development"

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",") if o.strip()]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()


# Global singleton — import this everywhere
settings = get_settings()
