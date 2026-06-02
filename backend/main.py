import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import settings
from utils.logger import logger
from routers.health import router as health_router


# ── Lifespan: startup and shutdown logic ──────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs on server startup and shutdown.
    Add background task starts here on later days.
    """
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"LLM mode: {settings.LLM_MODE}")
    logger.info(f"Battle engine: {'enabled' if settings.BATTLE_ENABLED else 'disabled'}")
    logger.info(f"ML classifier: {'enabled' if settings.ML_ENABLED else 'disabled'}")

    # Startup complete
    logger.info(f"{settings.APP_NAME} ready.")

    yield  # Server is running

    # Shutdown
    logger.info(f"{settings.APP_NAME} shutting down.")


# ── FastAPI app ───────────────────────────────────────────────────────
app = FastAPI(
    title=settings.APP_NAME,
    description="9-layer AI security system protecting LLM applications.",
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)


# ── CORS middleware ───────────────────────────────────────────────────
# In production this will be locked to the Vercel domain only.
# In development it allows localhost.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


# ── Global exception handler ──────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catch-all exception handler.
    NEVER exposes internal error details to the client.
    Logs the real error server-side.
    """
    logger.error(f"Unhandled exception on {request.method} {request.url.path}: {type(exc).__name__}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "INTERNAL_ERROR",
            "message": "An internal error occurred. Please try again.",
            "path": str(request.url.path)
        }
    )


# ── Request logging middleware ────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests with timing. Never logs request bodies (may contain keys)."""
    start = time.time()
    response = await call_next(request)
    duration = round((time.time() - start) * 1000, 2)
    logger.info(f"{request.method} {request.url.path} -> {response.status_code} ({duration}ms)")
    return response


# ── Register routers ──────────────────────────────────────────────────
app.include_router(health_router)

# These routers will be added on their respective days:
# Day 2: from routers.chat import router as chat_router
# Day 3: from routers.analytics import router as analytics_router
# Day 6: from routers.battle import router as battle_router
# Day 6: from routers.agents import router as agents_router


# ── Root endpoint ─────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "online",
        "docs": "/docs"
    }


# ── Entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.is_development,
        log_level="info"
    )
