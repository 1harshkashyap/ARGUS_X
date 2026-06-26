"""
ARGUS-X — Chat Router

HTTP boundary only. Handles:
  - Route definition and decorators
  - Per-request correlation ID (ARCH-010)
  - Delegation to services/pipeline.py
  - Thin safety-net exception handler

All security pipeline logic lives in services/pipeline.py.
"""

import asyncio
from fastapi import APIRouter

from utils.logger import logger
from utils.context import new_request_id, set_request_id, reset_request_id
from services.pipeline import run_pipeline
from schemas.chat import ChatRequest, ChatResponse

router = APIRouter(prefix="/api/v1", tags=["Chat"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Analyze and respond to user message",
    description=(
        "Full 10-step ARGUS-X security pipeline. "
        "Always returns ChatResponse — never raises."
    )
)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    HTTP boundary for the ARGUS-X security pipeline.

    Sets a per-request correlation ID so all pipeline log lines
    share a common [XXXXXXXX] prefix for easy debugging.
    Delegates all pipeline logic to services.pipeline.run_pipeline().
    """
    _req_id = new_request_id()
    _ctx_token = set_request_id(_req_id)
    try:
        logger.info(
            f"Request: session={request.session_id[:12]} "
            f"len={len(request.message)}"
        )
        return await run_pipeline(request)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        # Safety net — run_pipeline() should never raise, but if it does:
        logger.error(
            f"Unhandled chat route error: "
            f"{type(e).__name__}: {str(e)[:200]}"
        )
        return ChatResponse(
            response="System temporarily unavailable. Please try again.",
            error="INTERNAL_ERROR",
            session_id=request.session_id,
        )
    finally:
        reset_request_id(_ctx_token)


@router.post(
    "/redteam",
    response_model=ChatResponse,
    summary="Red team testing endpoint",
    description="Same pipeline as /chat. For testing attack patterns directly.",
    tags=["Red Team"]
)
async def redteam(request: ChatRequest) -> ChatResponse:
    """Red team endpoint — delegates to chat() which sets its own context."""
    return await chat(request)
