import asyncio
import time
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from utils.logger import logger
from utils.db import update_battle_state, get_battle_state
from agents.red_agent import red_agent, AttackResult
from agents.blue_agent import blue_agent
from security.firewall import firewall
from config import settings


# ── Constants ─────────────────────────────────────────────────────────

# Seconds to sleep between ticks when an error occurs (not the normal interval)
_ERROR_SLEEP = 5.0

# Maximum payload length stored in battle state preview
_PREVIEW_LEN = 120


# ── Battle Engine ─────────────────────────────────────────────────────

class BattleEngine:
    """
    Autonomous 60-second battle loop between Red Agent and Blue Agent.

    Each tick:
      1. Increment tick counter, escalate Red tier (tier = min(5, tick//10 + 1))
      2. Red Agent generates attack for current tier
      3. Firewall analyzes attack (same pipeline as real requests)
      4. If blocked: blue_blocks += 1
      5. If bypassed: red_bypasses += 1, Blue Agent generates patch
      6. Update Supabase battle_state (single row, id=1)
      7. Sleep for BATTLE_INTERVAL_SECONDS

    Lifecycle:
      - Started as asyncio.create_task() in FastAPI lifespan
      - Cancelled cleanly on server shutdown (CancelledError re-raised)
      - Pause/resume via is_paused flag (no task cancellation needed)
      - Forced single tick via cycle() (for /agents/cycle endpoint)

    Invariants:
      - asyncio.CancelledError always re-raised — never swallowed
      - All other exceptions logged and loop continues
      - DB failure never crashes the loop
      - All counters only increment (never reset without explicit reset)
      - Never uses user API keys
    """

    def __init__(self) -> None:
        self.tick:              int   = 0
        self.red_attacks:       int   = 0
        self.red_bypasses:      int   = 0
        self.blue_blocks:       int   = 0
        self.blue_patches:      int   = 0
        self.red_tier:          int   = 1
        self.red_strategy:      str   = "NAIVE"
        self.is_paused:         bool  = False
        self.last_attack:       str   = ""
        self.last_result:       str   = "NONE"
        self._task: Optional[asyncio.Task[None]] = None
        self._start_time: float = time.monotonic()

    # ── Public lifecycle methods ──────────────────────────────────────

    def pause(self) -> None:
        """Pause the battle loop (completes current tick if running)."""
        self.is_paused = True
        logger.info("BattleEngine: paused")

    def resume(self) -> None:
        """Resume the battle loop."""
        self.is_paused = False
        logger.info("BattleEngine: resumed")

    def get_state(self) -> Dict[str, Any]:
        """Return current battle state as a plain dict (for API + DB)."""
        return {
            "tick":                    self.tick,
            "red_attacks":             self.red_attacks,
            "red_bypasses":            self.red_bypasses,
            "blue_blocks":             self.blue_blocks,
            "blue_patches":            self.blue_patches,
            "red_tier":                self.red_tier,
            "red_strategy":            self.red_strategy,
            "current_attack_preview":  self.last_attack[:_PREVIEW_LEN],
            "last_attack_result":      self.last_result,
            "updated_at":              datetime.now(timezone.utc).isoformat(),
        }

    def get_status(self) -> Dict[str, Any]:
        """Extended status including runtime info (for /agents/status)."""
        return {
            **self.get_state(),
            "is_paused":       self.is_paused,
            "uptime_seconds":  round(time.monotonic() - self._start_time, 2),
            "task_running":    self._task is not None and not self._task.done(),
        }

    # ── Main loop ─────────────────────────────────────────────────────

    async def run(self) -> None:
        """
        Main battle loop. Runs until cancelled.
        Started as asyncio.create_task() in FastAPI lifespan.
        """
        logger.info(
            f"BattleEngine: started "
            f"(interval={settings.BATTLE_INTERVAL_SECONDS}s)"
        )

        while True:
            try:
                if not self.is_paused:
                    await self._run_tick()
                await asyncio.sleep(settings.BATTLE_INTERVAL_SECONDS)

            except asyncio.CancelledError:
                logger.info("BattleEngine: cancelled — shutting down cleanly")
                raise  # ALWAYS re-raise CancelledError

            except Exception as e:
                logger.error(
                    f"BattleEngine: tick error: {type(e).__name__}: {str(e)[:200]}"
                )
                # Brief pause before retrying to avoid tight error loops
                try:
                    await asyncio.sleep(_ERROR_SLEEP)
                except asyncio.CancelledError:
                    raise

    # ── Single tick ───────────────────────────────────────────────────

    async def _run_tick(self) -> None:
        """
        Execute one battle tick.
        Any exception here is caught by run() and logged — never propagated.
        """
        self.tick += 1

        # Tier escalates: 1→2 at tick 10, 2→3 at tick 20, ... 5 at tick 40+
        self.red_tier     = min(5, self.tick // 10 + 1)
        self.red_strategy = _tier_strategy(self.red_tier)

        logger.info(
            f"BattleEngine tick {self.tick}: "
            f"tier={self.red_tier} strategy={self.red_strategy}"
        )

        # ── Red Agent attacks ─────────────────────────────────────────
        context: Dict[str, Any] = {
            "tick":           self.tick,
            "recent_bypasses": [],  # Could track history here in future
        }

        attack: AttackResult = await red_agent.generate_attack(
            tier=self.red_tier,
            context=context
        )

        if not attack.payload:
            logger.warning("BattleEngine: Red Agent returned empty payload — skipping tick")
            self.last_result = "SKIPPED"
            await self._push_state()
            return

        # Count only after confirming a real payload was generated
        self.red_attacks += 1
        self.last_attack = attack.payload[:_PREVIEW_LEN]

        # ── Firewall checks attack ────────────────────────────────────
        firewall_result = await firewall.analyze(
            attack.payload,
            session_id=f"battle_tick_{self.tick}"
        )

        if firewall_result.blocked:
            # Defense wins this round
            self.blue_blocks += 1
            self.last_result = "BLOCKED"
            logger.info(
                f"BattleEngine tick {self.tick}: BLOCKED "
                f"rule={firewall_result.matched_rule} "
                f"conf={firewall_result.confidence:.2f}"
            )

        else:
            # Red Agent bypassed — Blue Agent patches
            self.red_bypasses += 1
            self.last_result = "BYPASSED"
            logger.warning(
                f"BattleEngine tick {self.tick}: BYPASSED "
                f"tier={self.red_tier} — Blue Agent analyzing"
            )

            patch = await blue_agent.analyze_bypass(
                attack_payload=attack.payload,
                threat_type="PROMPT_INJECTION",
                tier=self.red_tier
            )

            if patch.success and patch.rule_written:
                self.blue_patches += 1
                logger.info(
                    f"BattleEngine tick {self.tick}: patched "
                    f"pattern={patch.pattern[:50]!r}"
                )

        # ── Push state to Supabase ────────────────────────────────────
        await self._push_state()

    async def _push_state(self) -> None:
        """Write current state to Supabase. Non-fatal on failure."""
        try:
            await update_battle_state(self.get_state())
        except Exception as e:
            logger.warning(f"BattleEngine: state push failed (non-fatal): {type(e).__name__}")

    # ── Forced cycle (for /agents/cycle endpoint) ─────────────────────

    async def cycle(self) -> Dict[str, Any]:
        """
        Force one immediate battle tick regardless of is_paused.
        Returns the state after the tick.
        Used by POST /api/v1/agents/cycle.
        """
        try:
            await self._run_tick()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"BattleEngine.cycle error: {type(e).__name__}")

        return {
            **self.get_state(),
            "forced_cycle": True
        }


# ── Helpers ───────────────────────────────────────────────────────────

def _tier_strategy(tier: int) -> str:
    return {1: "NAIVE", 2: "EVASIVE", 3: "OBFUSCATED",
            4: "CONTEXTUAL", 5: "APEX"}.get(tier, "NAIVE")


# ── Global singleton ──────────────────────────────────────────────────
battle_engine = BattleEngine()
