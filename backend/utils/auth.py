"""
Dashboard authentication dependency.

Validates the X-Dashboard-Key header on read-only endpoints that
expose operational data (analytics, battle state, agent status).

NOT used on:
  - /health (must stay public for UptimeRobot)
  - /chat   (uses BYOAK api_key in request body)
"""

from fastapi import Request, HTTPException
from config import settings
from utils.logger import logger


async def require_dashboard_key(request: Request) -> None:
    """
    FastAPI dependency -- validates X-Dashboard-Key header.

    Used on read-only endpoints:
      /analytics/stats, /analytics/logs, /analytics/xai,
      /battle/state, /agents/status, /agents/pause,
      /agents/resume, /agents/cycle

    Returns None on success.
    Raises HTTP 401 on missing or invalid key.
    """
    key = request.headers.get("X-Dashboard-Key", "")

    if not key:
        client_ip = "unknown"
        if request.client:
            client_ip = request.client.host
        logger.warning(
            f"Dashboard endpoint accessed without key: "
            f"{request.method} {request.url.path} "
            f"from {client_ip}"
        )
        raise HTTPException(
            status_code=401,
            detail={
                "error": "DASHBOARD_KEY_REQUIRED",
                "message": "X-Dashboard-Key header required for this endpoint."
            }
        )

    if key != settings.DASHBOARD_READ_KEY:
        logger.warning(
            f"Invalid dashboard key on: {request.url.path}"
        )
        raise HTTPException(
            status_code=401,
            detail={
                "error": "INVALID_DASHBOARD_KEY",
                "message": "Invalid X-Dashboard-Key."
            }
        )
