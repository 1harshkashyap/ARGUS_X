"""
ARGUS-X — Lightweight Dependency Injection Container

Central registry of all services. Each service is constructed once and
accessed by name. Routers use `get_container(request)` to resolve deps
instead of reaching into `request.app.state` directly.

Design principles:
  - No framework (no injector, python-inject, etc.)
  - Zero magic — explicit construction in `build()`
  - Typed access via properties for IDE autocomplete
  - Testable: swap any service by calling `container.register()`
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from utils.logger import setup_logger

log = setup_logger("argus.container")


class ServiceContainer:
    """
    Holds all ARGUS-X services as a typed, named registry.

    Construction order matters — services that depend on others must be
    registered after their dependencies. `build()` handles this.
    """

    def __init__(self) -> None:
        self._services: dict[str, Any] = {}
        self._built = False

    # ── Registration ──────────────────────────────────────────────

    def register(self, name: str, instance: Any) -> None:
        """Register (or override) a named service."""
        self._services[name] = instance

    def get(self, name: str) -> Any:
        """Retrieve a service by name. Raises KeyError if not found."""
        if name not in self._services:
            raise KeyError(f"Service '{name}' not registered in container")
        return self._services[name]

    # ── Typed Accessors (IDE autocomplete + type safety) ─────────

    @property
    def task_queue(self):
        """Bounded background task queue."""
        return self._services["task_queue"]

    @property
    def db(self):
        """L1: Supabase database client."""
        return self._services["db"]

    @property
    def models(self):
        """L2: Model loader (ONNX, embeddings)."""
        return self._services["models"]

    @property
    def firewall(self):
        """L2: Input firewall (regex + ML)."""
        return self._services["firewall"]

    @property
    def auditor(self):
        """L2: Output auditor."""
        return self._services["auditor"]

    @property
    def llm(self):
        """L2: LLM core (Gemini/OpenAI)."""
        return self._services["llm"]

    @property
    def fingerprinter(self):
        """L3: Attack fingerprinter."""
        return self._services["fingerprinter"]

    @property
    def correlator(self):
        """L3: Threat correlator."""
        return self._services["correlator"]

    @property
    def mutator(self):
        """L4: Semantic mutation engine."""
        return self._services["mutator"]

    @property
    def red_agent(self):
        """L5: Red team agent."""
        return self._services["red_agent"]

    @property
    def blue_agent(self):
        """L5: Blue defense agent."""
        return self._services["blue_agent"]

    @property
    def battle(self):
        """L5: AI vs AI battle engine."""
        return self._services["battle"]

    @property
    def evolution(self):
        """L6: Evolution tracker."""
        return self._services["evolution"]

    @property
    def clusterer(self):
        """L6: Threat clusterer."""
        return self._services["clusterer"]

    @property
    def xai(self):
        """L7: Explainable AI engine."""
        return self._services["xai"]

    @property
    def session_store(self):
        """Session threat tracking store."""
        return self._services["session_store"]

    # ── Build (wires everything in correct order) ────────────────

    async def build(self) -> "ServiceContainer":
        """
        Construct all services in dependency order.
        Call once during app startup (lifespan).
        """
        from utils.supabase_client import SupabaseClient
        from utils.model_loader import ModelLoader
        from utils.session_store import SessionStore
        from utils.task_queue import BackgroundTaskQueue
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

        # L0: Background Task Queue (bounded, with worker pool)
        task_queue = BackgroundTaskQueue(maxsize=50, workers=3, name="argus-bg")
        self.register("task_queue", task_queue)

        # L1: Database
        db = SupabaseClient()
        self.register("db", db)

        # L2: Core Security Pipeline
        models = ModelLoader()
        self.register("models", models)
        self.register("firewall", InputFirewall(models))
        self.register("auditor", OutputAuditor(models))
        self.register("llm", LLMCore())

        # L3: Attack Intelligence
        self.register("fingerprinter", AttackFingerprinter(models))
        self.register("correlator", ThreatCorrelator(db))

        # L4: Mutation Engine
        self.register("mutator", SemanticMutationEngine(models, db=db))

        # L5: AI vs AI Battle
        red = RedTeamAgent(db, self.firewall, self.correlator)
        blue = BlueAgent(self.firewall, self.mutator, self.fingerprinter, db)
        self.register("red_agent", red)
        self.register("blue_agent", blue)
        self.register("battle", BattleEngine(red, blue, db))

        # L6: Learning Layer
        self.register("evolution", EvolutionTracker(db=db, task_queue=task_queue))
        self.register("clusterer", ThreatClusterer(models, db=db, task_queue=task_queue))

        # L7: XAI Engine
        self.register("xai", XAIEngine())

        # Session store
        session = SessionStore()
        await session.init()
        self.register("session_store", session)

        self._built = True
        log.info(f"✅ Container built — {len(self._services)} services registered")
        return self

    async def shutdown(self) -> None:
        """Graceful shutdown — close resources that need cleanup."""
        if "task_queue" in self._services:
            await self.task_queue.shutdown()
        if "session_store" in self._services:
            await self.session_store.close()
        log.info("🔴 Container shutdown complete")

    def __repr__(self) -> str:
        names = ", ".join(sorted(self._services.keys()))
        return f"<ServiceContainer [{names}]>"


# ── Module-level singleton ────────────────────────────────────────────
_container: ServiceContainer | None = None


def get_container_instance() -> ServiceContainer:
    """Get the global container singleton. Raises if not initialized."""
    if _container is None:
        raise RuntimeError("ServiceContainer not initialized — call init_container() first")
    return _container


async def init_container() -> ServiceContainer:
    """Build and set the global container singleton."""
    global _container
    _container = ServiceContainer()
    await _container.build()
    return _container


def get_container(request) -> ServiceContainer:
    """
    FastAPI dependency — resolves the container from app state.

    Usage in routers:
        from utils.container import get_container, ServiceContainer

        @router.get("/example")
        async def example(c: ServiceContainer = Depends(get_container)):
            result = c.firewall.check(...)
    """
    return request.app.state.container
