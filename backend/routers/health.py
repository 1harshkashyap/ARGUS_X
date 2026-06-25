import asyncio
import time
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict

# Track server start time for uptime calculation
_START_TIME = time.time()

router = APIRouter()


class ServiceStatus(BaseModel):
    """Status of an individual service."""
    status: str = "unknown"
    detail: str = ""


class HealthResponse(BaseModel):
    """Complete system health response. Never returns 500."""
    status: str = "online"
    version: str = "2.0.0"
    app_name: str = "ARGUS-X"
    environment: str = "development"
    uptime_seconds: float = 0.0
    services: Dict[str, str] = {}


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System Health Check",
    description="Returns system status. Always returns 200. Never raises exceptions.",
    tags=["System"]
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    - Always returns HTTP 200
    - Never raises exceptions
    - Reports status of all services
    - Safe to call from monitoring tools (UptimeRobot etc.)
    """
    from config import settings

    uptime = round(time.time() - _START_TIME, 2)

    # Check each service safely — never crash if unavailable
    services = {}

    # Database status — real connectivity check (ARCH-011)
    try:
        if not settings.SUPABASE_URL or not settings.SUPABASE_KEY:
            services["database"] = "not_configured"
        else:
            from utils.db import check_connection
            connected = await asyncio.wait_for(
                check_connection(),
                timeout=3.0
            )
            services["database"] = "online" if connected else "unreachable"
    except asyncio.TimeoutError:
        services["database"] = "timeout"
    except Exception:
        services["database"] = "error"

    # LLM status
    try:
        if settings.GEMINI_API_KEY:
            services["llm"] = "gemini_configured"
        else:
            services["llm"] = "no_key"
    except Exception:
        services["llm"] = "error"

    # ML classifier status
    try:
        from security.ml_classifier import ml_classifier
        if ml_classifier.available:
            services["ml_classifier"] = "online"
        elif settings.ML_ENABLED:
            services["ml_classifier"] = "degraded"
        else:
            services["ml_classifier"] = "disabled"
    except Exception:
        services["ml_classifier"] = "unknown"

    # Battle engine status
    services["battle_engine"] = "disabled" if not settings.BATTLE_ENABLED else "enabled"

    return HealthResponse(
        status="online",
        version=settings.APP_VERSION,
        app_name=settings.APP_NAME,
        environment=settings.ENVIRONMENT,
        uptime_seconds=uptime,
        services=services
    )
