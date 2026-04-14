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
import json
import os
import logging
from contextlib import asynccontextmanager
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# Rate limiting
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    limiter = Limiter(key_func=get_remote_address)
    RATE_LIMIT_AVAILABLE = True
except ImportError:
    limiter = None
    RATE_LIMIT_AVAILABLE = False

# Internal imports
from utils.logger import setup_logger
from utils.supabase_client import SupabaseClient
from utils.model_loader import ModelLoader
from utils.session_store import SessionStore
from utils.auth import require_api_key, validate_api_key_config
from ml.firewall import InputFirewall
from ml.auditor import OutputAuditor
from ml.llm_core import LLMCore
from ml.fingerprinter import AttackFingerprinter
from ml.mutation_engine import SemanticMutationEngine
from ml.xai_engine import XAIEngine
from ml.evolution_tracker import EvolutionTracker
from ml.threat_clusterer import ThreatClusterer
from agents.red_team_agent import RedTeamAgent
from agents.blue_agent import BlueAgent
from agents.battle_engine import BattleEngine
from agents.threat_correlator import ThreatCorrelator
from routers import chat, redteam, analytics, agents, knowledge, battle, xai, benchmark, compliance

# Optional Sentry
try:
    import sentry_sdk
    sentry_dsn = os.getenv("SENTRY_DSN", "")
    if sentry_dsn:
        sentry_sdk.init(dsn=sentry_dsn, traces_sample_rate=0.1)
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
    # Crashes in production if API_KEY is missing. Warns in dev.
    validate_api_key_config()

    # Layer 1: Database
    app.state.db = SupabaseClient()

    # Layer 2: Core Security Pipeline
    app.state.models = ModelLoader()
    app.state.firewall = InputFirewall(app.state.models)
    app.state.auditor = OutputAuditor(app.state.models)
    app.state.llm = LLMCore()

    # Layer 3: Attack Intelligence
    app.state.fingerprinter = AttackFingerprinter(app.state.models)
    app.state.correlator = ThreatCorrelator(app.state.db)

    # Layer 4: Mutation Engine
    app.state.mutator = SemanticMutationEngine(app.state.models)

    # Layer 5: AI vs AI Battle
    app.state.red_agent = RedTeamAgent(app.state.db, app.state.firewall, app.state.correlator)
    app.state.blue_agent = BlueAgent(app.state.firewall, app.state.mutator, app.state.fingerprinter)
    app.state.battle = BattleEngine(app.state.red_agent, app.state.blue_agent, app.state.db)

    # Layer 6: Learning Layer
    app.state.evolution = EvolutionTracker()
    app.state.clusterer = ThreatClusterer(app.state.models)

    # Layer 7: XAI Engine
    app.state.xai = XAIEngine()

    # WebSocket clients — set + lock for safe concurrent access
    app.state.ws_clients: set[WebSocket] = set()
    app.state.ws_lock = asyncio.Lock()

    # Session threat tracking (Redis-backed, persistent)
    app.state.session_store = SessionStore()
    await app.state.session_store.init()

    # Load persisted dynamic rules from DB into firewall memory
    try:
        db_rules = await app.state.db.get_dynamic_rules()
        for rule in db_rules:
            app.state.firewall.add_dynamic_rule(
                rule.get("pattern", ""),
                rule.get("threat_type", "UNKNOWN"),
                rule.get("severity", 0.85) if "severity" in rule else 0.85,
            )
        if db_rules:
            log.info(f"📦 Loaded {len(db_rules)} dynamic rules from DB")
    except Exception as e:
        log.warning(f"Dynamic rules load failed (non-fatal): {e}")

    # Broadcast helper (must be set inside lifespan after state exists)
    app.state.broadcast = _broadcast

    # Wire broadcast into correlator for real-time campaign push
    app.state.correlator._broadcast = lambda data: _broadcast(app, "campaign", data)

    # Start autonomous loops — supervised for auto-restart on crash.
    # Each task uses a lambda factory so the coroutine can be re-created on restart.
    _tasks = [
        asyncio.create_task(supervised_task(
            lambda: app.state.red_agent.autonomous_loop(), "Red Agent"
        )),
        asyncio.create_task(supervised_task(
            lambda: app.state.correlator.correlation_loop(), "Threat Correlator"
        )),
        asyncio.create_task(supervised_task(
            lambda: app.state.battle.start(), "Battle Engine"
        )),
    ]

    log.info("✅ ARGUS-X fully operational — all 9 layers active")
    yield
    log.info("🔴 ARGUS-X shutting down")
    # Graceful shutdown: signal loops to stop, then cancel supervisor tasks
    app.state.red_agent.stop()
    app.state.correlator.stop()
    app.state.battle.stop()
    for task in _tasks:
        task.cancel()
    await asyncio.gather(*_tasks, return_exceptions=True)
    await app.state.session_store.close()


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
# Defaults to localhost:5173 for local development only.
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

# Guard: reject wildcard — it must never reach production.
if FRONTEND_URL.strip() == "*":
    raise RuntimeError("FATAL: FRONTEND_URL must not be '*'. Set a specific origin.")

ALLOWED_ORIGINS = [FRONTEND_URL]

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

# ─── Rate Limiting ────────────────────────────────────────────────────────────
if RATE_LIMIT_AVAILABLE:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    log.info("✅ Rate limiting active (slowapi)")
else:
    log.warning("⚠️  slowapi not installed — rate limiting is DISABLED")

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


# ─── WebSocket Live Feed ───────────────────────────────────────────────────────
# SECURITY: Auth-after-connect protocol.
# 1. Accept the connection (required by WebSocket spec to send close frames)
# 2. Wait for {"type": "auth", "token": "..."} within AUTH_TIMEOUT seconds
# 3. Block all other messages until authenticated
# 4. Close with 4003 if auth fails or times out
WS_AUTH_TIMEOUT = 10  # seconds
MAX_WS_CLIENTS = 100


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
            if msg.get("type") != "auth" or msg.get("token") != _expected:
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
                await ws.send_text(json.dumps({"type": "ping", "ts": datetime.utcnow().isoformat()}))
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
        "ts": datetime.utcnow().isoformat() + "Z",
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
