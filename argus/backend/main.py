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
from typing import List

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# Internal imports
from utils.logger import setup_logger
from utils.supabase_client import SupabaseClient
from utils.model_loader import ModelLoader
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
from routers import chat, redteam, analytics, agents, knowledge, battle, xai

# Optional Sentry
try:
    import sentry_sdk
    sentry_dsn = os.getenv("SENTRY_DSN", "")
    if sentry_dsn:
        sentry_sdk.init(dsn=sentry_dsn, traces_sample_rate=0.1)
except ImportError:
    pass

log = setup_logger("argus.main")


# ─── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("🛡️  ARGUS-X booting up — initializing all 9 layers...")

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

    # WebSocket clients
    app.state.ws_clients: List[WebSocket] = []

    # Session threat tracking
    app.state.session_threats: dict = {}

    # Broadcast helper (must be set inside lifespan after state exists)
    app.state.broadcast = _broadcast

    # Start autonomous loops (wrapped for safety)
    try:
        asyncio.create_task(app.state.red_agent.autonomous_loop())
        log.info("🔴 Red Agent autonomous loop started")
    except Exception as e:
        log.warning(f"Red Agent loop failed to start: {e}")

    try:
        asyncio.create_task(app.state.correlator.correlation_loop())
        log.info("🔗 Threat Correlator loop started")
    except Exception as e:
        log.warning(f"Correlator loop failed to start: {e}")

    try:
        asyncio.create_task(app.state.battle.start())
        log.info("⚔️ Battle Engine loop started")
    except Exception as e:
        log.warning(f"Battle Engine loop failed to start: {e}")

    log.info("✅ ARGUS-X fully operational — all 9 layers active")
    yield
    log.info("🔴 ARGUS-X shutting down")


# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ARGUS-X",
    description="Autonomous AI Defense Operating System — The AI that defends AI",
    version="3.0.0",
    lifespan=lifespan,
)

# ─── CORS ─────────────────────────────────────────────────────────────────────
_default_origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8000",
]
_env_origins = os.getenv("CORS_ORIGINS", "")
_extra_origins = [o.strip() for o in _env_origins.split(",") if o.strip()] if _env_origins else []
ALLOWED_ORIGINS = _extra_origins + _default_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
)

# ─── Include Routers ──────────────────────────────────────────────────────────
app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
app.include_router(redteam.router, prefix="/api/v1", tags=["redteam"])
app.include_router(analytics.router, prefix="/api/v1", tags=["analytics"])
app.include_router(agents.router, prefix="/api/v1", tags=["agents"])
app.include_router(knowledge.router, prefix="/api/v1", tags=["knowledge"])
app.include_router(battle.router, prefix="/api/v1", tags=["battle"])
app.include_router(xai.router, prefix="/api/v1", tags=["xai"])


# ─── WebSocket Live Feed ───────────────────────────────────────────────────────
@app.websocket("/ws/live")
async def ws_live_feed(ws: WebSocket):
    await ws.accept()
    app.state.ws_clients.append(ws)
    log.info(f"Dashboard connected. Total: {len(app.state.ws_clients)}")

    # Send last 30 events on connect
    try:
        recent = await app.state.db.get_recent_events(30)
        for ev in recent:
            await ws.send_text(json.dumps({"type": "history", "data": ev}, default=str))
    except Exception as e:
        log.warning(f"History send failed: {e}")

    try:
        while True:
            await asyncio.sleep(15)
            await ws.send_text(json.dumps({"type": "ping", "ts": datetime.utcnow().isoformat()}))
    except WebSocketDisconnect:
        if ws in app.state.ws_clients:
            app.state.ws_clients.remove(ws)


# ─── Broadcast helper ─────────────────────────────────────────────────
async def _broadcast(app_instance, event_type: str, data: dict):
    dead = []
    for ws in app_instance.state.ws_clients:
        try:
            await ws.send_text(json.dumps({"type": event_type, "data": data}, default=str))
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in app_instance.state.ws_clients:
            app_instance.state.ws_clients.remove(ws)


# ─── Health ───────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "operational",
        "service": "ARGUS-X",
        "version": "3.0.0",
        "tagline": "The AI that defends AI",
        "layers": {
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
        },
        "agents": {
            "red_team": app.state.red_agent.status(),
            "blue_team": app.state.blue_agent.status(),
            "correlator": app.state.correlator.status(),
        },
        "ws_connections": len(app.state.ws_clients),
        "ts": datetime.utcnow().isoformat() + "Z",
    }


@app.exception_handler(Exception)
async def global_error(req: Request, exc: Exception):
    log.error(f"Unhandled: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": str(exc)})


# ─── Frontend Serving ────────────────────────────────────────────────
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    @app.get("/")
    async def serve_frontend():
        return FileResponse(str(FRONTEND_DIR / "index.html"))
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
