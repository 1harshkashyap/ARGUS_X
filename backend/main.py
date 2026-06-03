import asyncio
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import settings
from utils.logger import logger
from routers.health import router as health_router
from routers.chat import router as chat_router
from security.firewall import firewall


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")

    # Initialize firewall (loads dynamic rules from Supabase)
    await firewall.initialize()
    logger.info("Firewall initialized")

    logger.info(f"{settings.APP_NAME} ready.")
    yield

    # Graceful shutdown: cancel firewall background refresh task
    await firewall.shutdown()
    logger.info(f"{settings.APP_NAME} shutting down.")


app = FastAPI(
    title=settings.APP_NAME,
    description="9-layer AI security system protecting LLM applications.",
    version=settings.APP_VERSION,
    docs_url=None if settings.is_production else "/docs",
    redoc_url=None if settings.is_production else "/redoc",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(f"Unhandled exception: {type(exc).__name__} on {request.url.path}")
    return JSONResponse(
        status_code=500,
        content={"error": "INTERNAL_ERROR", "message": "An internal error occurred."}
    )

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = round((time.time() - start) * 1000, 2)
    if request.url.path != "/health":
        logger.info(f"{request.method} {request.url.path} → {response.status_code} ({duration}ms)")
    return response

# Register all Day 2 routers
app.include_router(health_router)
app.include_router(chat_router)

@app.get("/", include_in_schema=False)
async def root():
    result = {"name": settings.APP_NAME, "version": settings.APP_VERSION, "status": "online"}
    if not settings.is_production:
        result["docs"] = "/docs"
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000,
                reload=settings.is_development, log_level="info")
