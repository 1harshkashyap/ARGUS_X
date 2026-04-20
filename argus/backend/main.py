"""
ARGUS-X — Autonomous AI Defense Operating System
Main FastAPI Application Entry Point — v3.0

All 9 layers wired:
  L1: Database + Realtime (Supabase)
  L2: Core Security Pipeline (Firewall + LLM + Auditor)
  L3: Attack Fingerprinter + Threat Correlator
  L4: Mutation Engine
  L5: AI vs AI Battle Engine
  L6: Learning Layer (Clusterer + Evolution)
  L7: XAI Engine
  L8: Visualization (frontend)
  L9: Defense Command Center (frontend)
"""
import asyncio
import hashlib
import hmac
import json
import os
import logging
import secrets
import time as _time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# Rate limiting
# SECURITY: Use API key for rate limiting when available; fallback to IP.
# This prevents per-IP bypass via proxies/VPNs.
# CRITICAL: slowapi is a required dependency — never silently disable rate limiting.
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded


def _rate_limit_key(request: Request) -> str:
    """Rate limit by API key first, fallback to IP."""
    api_key = request.headers.get("x-api-key", "")
    if api_key:
        # Hash the key so it's not stored in plaintext in rate limit backend
        return "key:" + hashlib.sha256(api_key.encode()).hexdigest()[:16]
    return "ip:" + get_remote_address(request)


limiter = Limiter(key_func=_rate_limit_key)

# Internal imports
from utils.logger import setup_logger
from utils.auth import require_api_key, validate_api_key_config
from utils.container import init_container
from routers import chat, redteam, analytics, agents, knowledge, battle, xai, benchmark, compliance

# Optional Sentry
try:
    import sentry_sdk
    sentry_dsn = os.getenv("SENTRY_DSN", "")
    if sentry_dsn:
        sentry_sdk.init(
            dsn=sentry_dsn,
            traces_sample_rate=0.1,
            environment=os.getenv("ENV", "development"),
            release=os.getenv("RAILWAY_GIT_COMMIT_SHA", "local"),
        )
except ImportError:
    pass

log = setup_logger("argus.main")


# ─── Task Supervisor ──────────────────────────────────────────────────────────
RESTART_DELAY_S = 5

async def supervised_task(coro_factory, name: str) -> None:
    """
    Supervisor wrapper for autonomous background loops.

    Runs the coroutine in an infinite restart loop: if the task crashes,
    the exception is logged and the task is restarted after a delay.
    CancelledError (from shutdown) is re-raised to allow clean exit.

    Args:
        coro_factory: A zero-arg callable that returns the coroutine to run.
                      Must be a factory (not an awaitable) so we can restart it.
        name: Human-readable name for logging.
    """
    while True:
        try:
            log.info(f"🟢 [{name}] Starting…")
            await coro_factory()
        except asyncio.CancelledError:
            log.info(f"🔴 [{name}] Cancelled — shutting down")
            raise
        except Exception:
            log.exception(f"💥 [{name}] Crashed — restarting in {RESTART_DELAY_S}s")
            await asyncio.sleep(RESTART_DELAY_S)


# ─── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("🛡️  ARGUS-X booting up — initializing all 9 layers...")

    # SECURITY: Validate API key configuration before anything else.
    validate_api_key_config()

    # Build the service container (constructs all 15 services in dependency order)
    container = await init_container()
    app.state.container = container

    # Backward-compat aliases — routers not yet migrated to container
    # can still use request.app.state.xyz
    app.state.db = container.db
    app.state.models = container.models
    app.state.firewall = container.firewall
    app.state.auditor = container.auditor
    app.state.llm = container.llm
    app.state.fingerprinter = container.fingerprinter
    app.state.correlator = container.correlator
    app.state.mutator = container.mutator
    app.state.red_agent = container.red_agent
    app.state.blue_agent = container.blue_agent
    app.state.battle = container.battle
    app.state.evolution = container.evolution
    app.state.clusterer = container.clusterer
    app.state.xai = container.xai
    app.state.session_store = container.session_store
    app.state.task_queue = container.task_queue

    # Start bounded background task queue
    await container.task_queue.start()

    # WebSocket clients — set + lock for safe concurrent access
    app.state.ws_clients: set[WebSocket] = set()
    app.state.ws_lock = asyncio.Lock()

    # Load persisted dynamic rules from DB into firewall memory
    try:
        db_rules = await container.db.get_dynamic_rules()
        for rule in db_rules:
            container.firewall.add_dynamic_rule(
                rule.get("pattern", ""),
                rule.get("threat_type", "UNKNOWN"),
                rule.get("severity", 0.85) if "severity" in rule else 0.85,
            )
        if db_rules:
            log.info(f"📦 Loaded {len(db_rules)} dynamic rules from DB")
    except Exception as e:
        log.warning(f"Dynamic rules load failed (non-fatal): {e}")

    # Load persisted evolution data from DB
    try:
        evo_snapshot = await container.db.get_latest_evolution()
        if evo_snapshot:
            container.evolution.last_avg = evo_snapshot.get("window_avg", 0.0)
            container.evolution.escalation_count = evo_snapshot.get("escalation_count", 0)
            log.info(f"📦 Loaded evolution state: avg={container.evolution.last_avg:.2f}, escalations={container.evolution.escalation_count}")
    except Exception as e:
        log.warning(f"Evolution load failed (non-fatal): {e}")

    # Load persisted cluster data from DB
    try:
        db_clusters = await container.db.get_latest_clusters()
        if db_clusters:
            container.clusterer.clusters = [
                {
                    "cluster_id": c.get("cluster_id", ""),
                    "size": c.get("cluster_size", 0),
                    "dominant_type": c.get("dominant_type", ""),
                    "avg_sophistication": c.get("avg_sophistication", 0),
                    "sample_hash": c.get("sample_text", ""),
                    "types": json.loads(c["types"]) if isinstance(c.get("types"), str) else c.get("types", []),
                }
                for c in db_clusters
            ]
            log.info(f"📦 Loaded {len(db_clusters)} threat clusters from DB")
    except Exception as e:
        log.warning(f"Cluster load failed (non-fatal): {e}")

    # Broadcast helper
    app.state.broadcast = _broadcast

    # Wire broadcast into correlator for real-time campaign push
    container.correlator._broadcast = lambda data: _broadcast(app, "campaign", data)

    # Start autonomous loops — supervised for auto-restart on crash.
    _tasks = [
        asyncio.create_task(supervised_task(
            lambda: container.red_agent.autonomous_loop(), "Red Agent"
        )),
        asyncio.create_task(supervised_task(
            lambda: container.correlator.correlation_loop(), "Threat Correlator"
        )),
        asyncio.create_task(supervised_task(
            lambda: container.battle.start(), "Battle Engine"
        )),
    ]

    log.info("✅ ARGUS-X fully operational — all 9 layers active")
    yield
    log.info("🔴 ARGUS-X shutting down")
    # Graceful shutdown
    container.red_agent.stop()
    container.correlator.stop()
    container.battle.stop()
    for task in _tasks:
        task.cancel()
    await asyncio.gather(*_tasks, return_exceptions=True)
    await container.shutdown()


# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ARGUS-X",
    description="Autonomous AI Defense Operating System — The AI that defends AI",
    version="3.0.0",
    lifespan=lifespan,
)

# ─── CORS ─────────────────────────────────────────────────────────────────────
# SECURITY: Only the trusted frontend origin is allowed.
# Set FRONTEND_URL in production (e.g. "https://argus-x.example.com").
# Auto-detects Railway public domain if available.
_railway_domain = os.getenv("RAILWAY_PUBLIC_DOMAIN", "")
_default_frontend = f"https://{_railway_domain}" if _railway_domain else "http://localhost:5173"
FRONTEND_URL = os.getenv("FRONTEND_URL", _default_frontend)

# Guard: reject wildcard — it must never reach production.
if FRONTEND_URL.strip() == "*":
    raise RuntimeError("FATAL: FRONTEND_URL must not be '*'. Set a specific origin.")

# SECURITY: Warn if production is relying on auto-detected FRONTEND_URL.
_env_mode = os.getenv("ENV", "development").lower()
if _env_mode == "production" and not os.getenv("FRONTEND_URL"):
    log.critical(
        "🚨 FRONTEND_URL not explicitly set in production — auto-detected '%s' from "
        "RAILWAY_PUBLIC_DOMAIN. Set FRONTEND_URL explicitly for security.", FRONTEND_URL
    )

ALLOWED_ORIGINS = [FRONTEND_URL]
# Also allow the Railway domain if it differs from FRONTEND_URL
if _railway_domain and f"https://{_railway_domain}" not in ALLOWED_ORIGINS:
    ALLOWED_ORIGINS.append(f"https://{_railway_domain}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With", "X-API-Key"],
)


# ─── Request Body Size Limit ──────────────────────────────────────────────────
# SECURITY: Reject oversized request bodies to prevent DoS / memory exhaustion.
# Max 1MB — largest legitimate payload is a 10000-char message (~10KB).
MAX_BODY_SIZE = 1_048_576  # 1 MB


@app.middleware("http")
async def limit_request_body(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_BODY_SIZE:
        return JSONResponse(
            status_code=413,
            content={"detail": f"Request body too large. Max {MAX_BODY_SIZE} bytes."},
        )
    return await call_next(request)


# ─── Request ID Tracing ──────────────────────────────────────────────────────
# OBSERVABILITY: Every request gets a unique ID for log correlation and debugging.
import uuid as _uuid


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("x-request-id", str(_uuid.uuid4()))
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ─── Security Headers ─────────────────────────────────────────────────────────
# SECURITY: Harden all responses against XSS, clickjacking, and MIME-sniffing.
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    # CSP: restrict script sources to same-origin only; block inline scripts.
    # 'unsafe-inline' for styles is needed for CSS-in-JS (e.g., styled-components).
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "img-src 'self' data: blob:; "
        "connect-src 'self' wss: ws:; "
        "frame-ancestors 'none'; "
        "base-uri 'self'"
    )
    return response


# ─── Rate Limiting ────────────────────────────────────────────────────────────
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
log.info("✅ Rate limiting active (slowapi)")

# ─── Include Routers ──────────────────────────────────────────────────────────
# All /api/v1 routes require X-API-Key header (when API_KEY env var is set).
# Health, WebSocket, and static routes remain public.
_auth = [Depends(require_api_key)]
app.include_router(chat.router, prefix="/api/v1", tags=["chat"], dependencies=_auth)
app.include_router(redteam.router, prefix="/api/v1", tags=["redteam"], dependencies=_auth)
app.include_router(analytics.router, prefix="/api/v1", tags=["analytics"], dependencies=_auth)
app.include_router(agents.router, prefix="/api/v1", tags=["agents"], dependencies=_auth)
app.include_router(knowledge.router, prefix="/api/v1", tags=["knowledge"], dependencies=_auth)
app.include_router(battle.router, prefix="/api/v1", tags=["battle"], dependencies=_auth)
app.include_router(xai.router, prefix="/api/v1", tags=["xai"], dependencies=_auth)
app.include_router(benchmark.router, prefix="/api/v1", tags=["benchmark"], dependencies=_auth)
app.include_router(compliance.router, prefix="/api/v1", tags=["compliance"], dependencies=_auth)


# ─── Short-Lived WebSocket Token ──────────────────────────────────────────────
# SECURITY: Clients exchange their API key for a short-lived WS token via
# POST /api/v1/ws-token. The token is an HMAC of a nonce + timestamp,
# valid for WS_TOKEN_TTL seconds. This prevents permanent API keys from
# appearing in WebSocket frames visible in browser devtools.
WS_TOKEN_TTL = 60  # seconds
_WS_TOKEN_SECRET = os.getenv("WS_TOKEN_SECRET", "") or secrets.token_hex(32)

# In-memory token store: token_str → expiry_timestamp
# Cleaned up lazily on each token creation.
_ws_tokens: dict[str, float] = {}
_ws_tokens_lock = asyncio.Lock()


@app.post("/api/v1/ws-token", dependencies=[Depends(require_api_key)])
async def create_ws_token():
    """Exchange API key for a short-lived WebSocket auth token."""
    nonce = secrets.token_hex(16)
    ts = str(int(_time.time()))
    token = hmac.new(
        _WS_TOKEN_SECRET.encode(), (nonce + ts).encode(), hashlib.sha256
    ).hexdigest()
    expiry = _time.time() + WS_TOKEN_TTL

    async with _ws_tokens_lock:
        # Lazy cleanup of expired tokens
        now = _time.time()
        expired = [t for t, exp in _ws_tokens.items() if exp <= now]
        for t in expired:
            del _ws_tokens[t]
        _ws_tokens[token] = expiry

    return {"token": token, "expires_in": WS_TOKEN_TTL}


# ─── WebSocket Live Feed ───────────────────────────────────────────────────────
# SECURITY: Auth-after-connect protocol.
# 1. Accept the connection (required by WebSocket spec to send close frames)
# 2. Wait for {"type": "auth", "token": "..."} within AUTH_TIMEOUT seconds
# 3. Validate token is a valid short-lived WS token OR the API key (fallback)
# 4. Close with 4003 if auth fails or times out
WS_AUTH_TIMEOUT = 10  # seconds
MAX_WS_CLIENTS = 100


async def _validate_ws_token(token: str) -> bool:
    """Check if token is a valid short-lived WS token (one-time use)."""
    async with _ws_tokens_lock:
        expiry = _ws_tokens.get(token)
        if expiry is not None and _time.time() < expiry:
            del _ws_tokens[token]  # One-time use
            return True
    return False


@app.websocket("/ws/live")
async def ws_live_feed(ws: WebSocket):
    await ws.accept()

    # ── SECURITY: Auth-after-connect ──────────────────────────────────
    _expected = os.getenv("API_KEY", "")
    if _expected:
        try:
            # Wait for auth message within timeout
            raw = await asyncio.wait_for(ws.receive_text(), timeout=WS_AUTH_TIMEOUT)
            msg = json.loads(raw)
            token = msg.get("token", "")
            # Accept short-lived WS token (preferred) or API key (fallback)
            is_valid = await _validate_ws_token(token) if token else False
            if msg.get("type") != "auth" or (not is_valid and token != _expected):
                log.warning("🔒 WebSocket rejected — invalid auth")
                await ws.close(code=4003)
                return
        except asyncio.TimeoutError:
            log.warning("🔒 WebSocket rejected — auth timeout")
            await ws.close(code=4003)
            return
        except Exception:
            log.warning("🔒 WebSocket rejected — malformed auth")
            await ws.close(code=4003)
            return

    # ── Enforce connection cap ────────────────────────────────────────
    async with app.state.ws_lock:
        if len(app.state.ws_clients) >= MAX_WS_CLIENTS:
            log.warning("🔒 WebSocket rejected — connection limit reached")
            await ws.close(code=4008)
            return
        app.state.ws_clients.add(ws)

    log.info(f"Dashboard connected (authenticated). Total: {len(app.state.ws_clients)}")

    # Send last 30 events on connect
    try:
        recent = await app.state.db.get_recent_events(30)
        for ev in recent:
            await ws.send_text(json.dumps({"type": "history", "data": ev}, default=str))
    except Exception as e:
        log.warning(f"History send failed: {e}")

    try:
        while True:
            # Listen for client messages (or detect disconnect)
            # The receive_text() call also serves as a keepalive detector
            try:
                data = await asyncio.wait_for(ws.receive_text(), timeout=30)
                # Client can send pong or other messages — ignore
            except asyncio.TimeoutError:
                # No message from client in 30s — send ping
                await ws.send_text(json.dumps({"type": "ping", "ts": datetime.now(timezone.utc).isoformat()}))
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        async with app.state.ws_lock:
            app.state.ws_clients.discard(ws)


# ─── Broadcast helper ─────────────────────────────────────────────────
async def _broadcast(app_instance, event_type: str, data: dict):
    payload = json.dumps({"type": event_type, "data": data}, default=str)
    # Snapshot the set under lock to avoid mutation during iteration
    async with app_instance.state.ws_lock:
        clients = set(app_instance.state.ws_clients)
    dead = []
    for ws in clients:
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)
    if dead:
        async with app_instance.state.ws_lock:
            for ws in dead:
                app_instance.state.ws_clients.discard(ws)


# ─── Health ───────────────────────────────────────────────────────────────────
# SECURITY: Public health returns minimal info only.
# Detailed layer/agent info requires authentication to prevent architecture mapping.
@app.get("/health")
async def health(request: Request):
    # Check if request is authenticated
    _expected = os.getenv("API_KEY", "")
    _provided = request.headers.get("x-api-key", "")
    is_authenticated = (not _expected) or (_provided == _expected)

    # Public response: minimal status for load balancers / uptime monitors
    response = {
        "status": "operational",
        "service": "ARGUS-X",
        "version": "3.0.0",
        "ts": datetime.now(timezone.utc).isoformat() + "Z",
    }

    # Authenticated response: full internal details
    if is_authenticated:
        response["layers"] = {
            "L1_database": app.state.db.available,
            "L2_firewall": app.state.firewall.mode(),
            "L2_llm": app.state.llm.get_mode(),
            "L3_fingerprinter": app.state.fingerprinter.ready,
            "L3_correlator": app.state.correlator.status(),
            "L4_mutation_engine": True,
            "L5_battle_engine": app.state.battle.get_state().get("status", "UNKNOWN"),
            "L6_evolution": True,
            "L6_clusterer": True,
            "L7_xai": True,
        }
        response["agents"] = {
            "red_team": app.state.red_agent.status(),
            "blue_team": app.state.blue_agent.status(),
            "correlator": app.state.correlator.status(),
        }
        response["ws_connections"] = len(app.state.ws_clients)
        response["task_queue"] = app.state.task_queue.status()

    return response


@app.exception_handler(Exception)
async def global_error(req: Request, exc: Exception):
    log.error(f"Unhandled: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# ─── Frontend Serving ────────────────────────────────────────────────
# Priority: built /dist (production) → raw source (dev fallback)
FRONTEND_DIST = Path(__file__).parent.parent / "frontend" / "dist"
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
_STATIC_DIR = FRONTEND_DIST if FRONTEND_DIST.exists() else FRONTEND_DIR

if _STATIC_DIR.exists():
    # Serve static assets first (JS, CSS, images)
    _assets_dir = _STATIC_DIR / "assets"
    if _assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(_assets_dir)), name="assets")
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    # SPA fallback: serve index.html for all non-API routes
    @app.get("/{path:path}")
    async def serve_frontend(path: str):
        # Don't intercept API or health routes
        if path.startswith("api/") or path == "health" or path.startswith("ws/"):
            return JSONResponse(status_code=404, content={"detail": "Not found"})
        index = _STATIC_DIR / "index.html"
        if index.exists():
            return FileResponse(str(index))
        return JSONResponse(status_code=404, content={"detail": "Frontend not built"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
