import asyncio
import os
import re
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import settings
from utils.logger import logger
from routers.health import router as health_router
from routers.chat import router as chat_router
from routers.analytics import router as analytics_router
from routers.battle import router as battle_router
from routers.agents import router as agents_router
from security.firewall import firewall
from utils.db import close_db_client

# ── Log sanitization ─────────────────────────────────────────────────
_CONTROL_CHAR_RE = re.compile(r'[\x00-\x1f\x7f-\x9f]')


def _sanitize_for_log(value: str) -> str:
    """Remove control characters to prevent log injection."""
    return _CONTROL_CHAR_RE.sub('', value)


# ── Client IP extraction (reverse-proxy aware) ───────────────────────

def _get_client_ip(request: Request) -> str:
    """
    Extract real client IP via uvicorn's proxy-header resolution.

    When --proxy-headers is set, uvicorn rewrites request.client.host
    to the value from X-Forwarded-For (using its trusted-proxy list).
    We do NOT parse X-Forwarded-For manually — that is trivially spoofed
    when forwarded_allow_ips is broad (SEC-005 fix).
    """
    if request.client:
        return request.client.host
    return "unknown"


# ── Rate limiter ──────────────────────────────────────────────────────

class _RateLimiter:
    """
    In-memory per-IP sliding-window rate limiter.
    Thread-safe for asyncio (single event loop).
    """

    def __init__(self, max_requests: int, window_seconds: int = 60):
        self._max = max(max_requests, 1)  # Floor at 1 to prevent lockout
        self._window = window_seconds
        self._requests: dict = defaultdict(list)
        self._cleanup_counter = 0

    def is_allowed(self, client_ip: str) -> bool:
        now = time.time()
        cutoff = now - self._window

        timestamps = self._requests[client_ip]
        self._requests[client_ip] = [t for t in timestamps if t > cutoff]

        if len(self._requests[client_ip]) >= self._max:
            return False

        self._requests[client_ip].append(now)

        self._cleanup_counter += 1
        if self._cleanup_counter % 500 == 0:
            self._full_cleanup(cutoff)

        return True

    def _full_cleanup(self, cutoff: float):
        """Remove IPs with no recent requests."""
        empty_ips = [ip for ip, ts in self._requests.items()
                     if not ts or all(t <= cutoff for t in ts)]
        for ip in empty_ips:
            del self._requests[ip]


_rate_limiter = _RateLimiter(
    max_requests=settings.RATE_LIMIT_PER_MINUTE,
    window_seconds=60
)

# Stricter limiter for expensive endpoints (SEC-006: LLM cost protection)
# /chat triggers Gemini calls; /agents/cycle triggers full battle tick
_expensive_rate_limiter = _RateLimiter(
    max_requests=max(settings.RATE_LIMIT_PER_MINUTE // 3, 5),  # ~10 req/min
    window_seconds=60
)

# Paths that trigger LLM calls (Gemini/OpenAI) — need stricter rate limiting
_EXPENSIVE_PATHS = frozenset({
    "/api/v1/chat",
    "/api/v1/redteam",
    "/api/v1/agents/cycle",
})

# ── Body size limit ──────────────────────────────────────────────────
# Reject requests > 64KB BEFORE reading the body into memory.
# Prevents OOM from malicious multi-GB payloads.
_MAX_BODY_BYTES = 65_536  # 64 KB — enough for 10,000 chars + JSON overhead


# ── Lifespan ──────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")

    # Initialize firewall first (battle engine depends on it)
    await firewall.initialize()
    logger.info("Firewall initialized")

    # ── Startup configuration summary (ARCH-005) ─────────────────────
    def _status(condition: bool, ok: str = "configured",
                 bad: str = "MISSING") -> str:
        return ok if condition else bad

    logger.info("─" * 52)
    logger.info("ARGUS-X — STARTUP CONFIGURATION")
    logger.info(f"  Environment     {settings.ENVIRONMENT}")
    logger.info(
        f"  Gemini key      "
        f"{_status(bool(settings.GEMINI_API_KEY))}"
    )
    logger.info(
        f"  Supabase        "
        f"{_status(bool(settings.SUPABASE_URL and settings.SUPABASE_KEY))}"
    )
    logger.info(
        f"  ML Classifier   "
        f"{'enabled' if settings.ML_ENABLED else 'disabled'}"
    )
    logger.info(
        f"  Battle Engine   "
        f"{'enabled' if settings.BATTLE_ENABLED else 'disabled'}"
    )
    logger.info(
        f"  Dashboard Key   "
        f"{_status(bool(settings.DASHBOARD_READ_KEY), 'configured', 'FAIL-CLOSED — endpoints will 503')}"
    )
    logger.info(
        f"  Rate limit      "
        f"{settings.RATE_LIMIT_PER_MINUTE}/min (chat: 10/min)"
    )
    logger.info("─" * 52)

    # ── Single-instance constraint ────────────────────────────────────
    # ARGUS-X uses in-memory state for: rate limiting, session tracking,
    # campaign correlation, mutation cooldown, and the battle engine.
    # This design is correct for single-instance Railway deployment.
    # Do NOT run multiple instances without externalizing this state.
    # See docs/ARCHITECTURE.md — "Single-Instance Constraint" section.

    # Start battle engine if enabled
    battle_task = None
    if settings.BATTLE_ENABLED:
        from agents.battle_engine import battle_engine
        battle_task = asyncio.create_task(
            battle_engine.run(),
            name="battle_engine"
        )
        logger.info(
            f"Battle engine started "
            f"(interval={settings.BATTLE_INTERVAL_SECONDS}s)"
        )
    else:
        logger.info("Battle engine disabled (BATTLE_ENABLED=false)")

    logger.info(f"{settings.APP_NAME} ready.")
    yield

    # Graceful shutdown
    logger.info(f"{settings.APP_NAME} shutting down...")
    if battle_task and not battle_task.done():
        battle_task.cancel()
        try:
            await asyncio.gather(battle_task, return_exceptions=True)
        except Exception:
            pass
        logger.info("Battle engine stopped")

    await firewall.shutdown()
    await close_db_client()
    logger.info(f"{settings.APP_NAME} shutdown complete.")


app = FastAPI(
    title=settings.APP_NAME,
    description="9-layer AI security system protecting LLM applications.",
    version=settings.APP_VERSION,
    docs_url=None if settings.is_production else "/docs",
    redoc_url=None if settings.is_production else "/redoc",
    lifespan=lifespan
)

# SEC-004 fix: no allow_credentials (auth is header-based, not cookie-based)
# Production strips localhost origins to prevent CSRF from local dev servers
_cors_origins = settings.cors_origins
if settings.is_production:
    _cors_origins = [
        o for o in _cors_origins
        if not o.startswith("http://localhost")
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Dashboard-Key"],
)


# ── Exception handlers ───────────────────────────────────────────────

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Return clean 422 for malformed requests — don't expose internal field paths."""
    safe_path = _sanitize_for_log(request.url.path)
    logger.warning(f"Validation error on {safe_path}: {len(exc.errors())} error(s)")
    return JSONResponse(
        status_code=422,
        content={
            "error": "VALIDATION_ERROR",
            "message": "Invalid request format. Check your request body.",
            "detail_count": len(exc.errors())
        }
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all for unhandled exceptions. Never leaks internals."""
    if isinstance(exc, HTTPException):
        raise exc
    logger.error(f"Unhandled exception: {type(exc).__name__} on {_sanitize_for_log(request.url.path)}")
    return JSONResponse(
        status_code=500,
        content={"error": "INTERNAL_ERROR", "message": "An internal error occurred."}
    )


# ── Middleware ────────────────────────────────────────────────────────
# Registration order matters: LAST registered = OUTERMOST (runs first).
# We want: security_headers (outermost) -> body_size_limit -> rate_limit -> log_requests (innermost)
# So register in REVERSE: log_requests first, security_headers last.

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing (except /health to reduce noise)."""
    start = time.time()
    response = await call_next(request)
    duration = round((time.time() - start) * 1000, 2)
    if request.url.path != "/health":
        safe_path = _sanitize_for_log(request.url.path)
        logger.info(f"{request.method} {safe_path} -> {response.status_code} ({duration}ms)")
    return response


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Enforce per-IP rate limiting. Skips health checks.
    SEC-006: Expensive endpoints (LLM-triggering) get a stricter limit.
    """
    if request.url.path == "/health":
        return await call_next(request)

    client_ip = _get_client_ip(request)

    # Global rate limit
    if not _rate_limiter.is_allowed(client_ip):
        logger.warning(f"Rate limit exceeded for {_sanitize_for_log(client_ip)}")
        return JSONResponse(
            status_code=429,
            content={"error": "RATE_LIMIT_EXCEEDED", "message": "Too many requests. Please slow down."}
        )

    # Stricter limit for expensive endpoints (SEC-006)
    if request.url.path in _EXPENSIVE_PATHS:
        if not _expensive_rate_limiter.is_allowed(client_ip):
            logger.warning(f"Expensive endpoint rate limit exceeded for {_sanitize_for_log(client_ip)}")
            return JSONResponse(
                status_code=429,
                content={"error": "RATE_LIMIT_EXCEEDED", "message": "Too many requests to this endpoint. Please slow down."}
            )

    return await call_next(request)


@app.middleware("http")
async def body_size_limit(request: Request, call_next):
    """Reject oversized request bodies — checks BOTH header AND actual stream.

    Defends against chunked-encoding bypass where Content-Length is absent.
    Wraps the receive channel to count bytes as they arrive.
    """
    # ── Fast path: reject via Content-Length header if present ────
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            length = int(content_length)
        except (ValueError, OverflowError):
            return JSONResponse(
                status_code=400,
                content={"error": "INVALID_CONTENT_LENGTH", "message": "Invalid Content-Length header."}
            )
        if length > _MAX_BODY_BYTES:
            return JSONResponse(
                status_code=413,
                content={"error": "PAYLOAD_TOO_LARGE", "message": f"Request body exceeds {_MAX_BODY_BYTES} bytes."}
            )

    # ── Stream guard: wrap receive to count actual bytes ──────────
    # Catches chunked-encoding payloads that have no Content-Length.
    bytes_received = 0
    body_too_large = False
    original_receive = request._receive  # type: ignore[attr-defined]

    async def _guarded_receive():
        nonlocal bytes_received, body_too_large
        message = await original_receive()
        if message.get("type") == "http.request":
            chunk = message.get("body", b"")
            bytes_received += len(chunk)
            if bytes_received > _MAX_BODY_BYTES:
                body_too_large = True
                # Replace body with empty to stop further processing
                message["body"] = b""
                message["more_body"] = False
        return message

    request._receive = _guarded_receive  # type: ignore[attr-defined]

    if request.method in ("POST", "PUT", "PATCH"):
        # Force body read through the guarded channel
        try:
            body = await request.body()
        except Exception:
            pass
        if body_too_large:
            return JSONResponse(
                status_code=413,
                content={"error": "PAYLOAD_TOO_LARGE", "message": f"Request body exceeds {_MAX_BODY_BYTES} bytes."}
            )

    return await call_next(request)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    """Add security headers to EVERY response (outermost middleware)."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    response.headers["Permissions-Policy"] = "geolocation=(), camera=(), microphone=()"
    response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
    if settings.is_production:
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


# ── Routers ───────────────────────────────────────────────────────────
app.include_router(health_router)
app.include_router(chat_router)
app.include_router(analytics_router)
app.include_router(battle_router)
app.include_router(agents_router)


@app.get("/", include_in_schema=False)
async def root():
    result = {"status": "online"}
    if not settings.is_production:
        result["name"] = settings.APP_NAME
        result["version"] = settings.APP_VERSION
        result["docs"] = "/docs"
    return result


if __name__ == "__main__":
    import uvicorn
    # SEC-009: forwarded_allow_ips from env (default '*' for dev, restrict in prod)
    _fwd_ips = os.environ.get("FORWARDED_ALLOW_IPS", "*")
    uvicorn.run("main:app", host="0.0.0.0", port=8000,
                reload=settings.is_development, log_level="info",
                proxy_headers=True, forwarded_allow_ips=_fwd_ips)
