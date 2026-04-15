"""
ARGUS-X — Battle Engine (Layer 5)
Orchestrates AI vs AI battle between Red Agent and Blue Agent.
Broadcasts battle state to Supabase Realtime every tick (1.5s).
"""
import asyncio
import random
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from agents.red_team_agent import RedTeamAgent

log = logging.getLogger("argus.battle_engine")

# Tier-to-strategy mapping
_TIER_STRATEGY = {
    1: "NAIVE", 2: "INDIRECT", 3: "OBFUSCATED",
    4: "MULTI_STEP", 5: "ADVERSARIAL",
}


class BattleEngine:
    """
    AI vs AI Battle Orchestrator:
    - Red Agent attacks with escalating tiers
    - Blue Agent defends, patches, spawns mutations
    - State broadcast to dashboard via Supabase Realtime
    """

    def __init__(self, red_agent, blue_agent, db):
        self.red_agent = red_agent
        self.blue_agent = blue_agent
        self.db = db
        self.running = False
        self.paused = False
        self.tick = 0
        self.tick_interval = 15  # seconds between battle ticks (fast enough for demos)
        
        # Battle scoreboard
        self.state = {
            "tick": 0,
            "red_attacks": 0,
            "red_bypasses": 0,
            "blue_blocks": 0,
            "blue_auto_patches": 0,
            "red_tier": 1,
            "red_strategy": "NAIVE",
            "battle_score": {"red": 0, "blue": 0},
            "mutations_total": 0,
            "status": "READY",
            "last_update": None,
        }
        log.info("✅ Battle Engine initialized")

    async def start(self):
        """Start the autonomous battle loop."""
        self.running = True
        self.state["status"] = "ACTIVE"
        log.info("⚔️ Battle Engine started — AI vs AI combat active")
        
        while self.running:
            if not self.paused:
                await self._run_tick()
            await asyncio.sleep(self.tick_interval)

    async def _run_tick(self):
        """Execute one battle tick."""
        self.tick += 1
        self.state["tick"] = self.tick

        # Update tier based on cycle count
        tier = min(5, 1 + (self.tick // 10))
        self.state["red_tier"] = tier
        self.state["red_strategy"] = _TIER_STRATEGY.get(tier, "UNKNOWN")

        # Red agent generates attacks for the current tier (escalation-aware)
        batch = self.red_agent.generate_batch(tier, count=5)

        for attack in batch:
            # Red agent tries the attack through the live firewall
            red_result = await self.red_agent._try_attack(attack)

            # Blue agent defends
            defense = await self.blue_agent.defend(attack["text"], attack["type"])
            
            self.state["red_attacks"] += 1
            
            if defense["blocked"]:
                self.state["blue_blocks"] += 1
                self.state["battle_score"]["blue"] += 1
                self.state["mutations_total"] += defense.get("mutations", 0)
            else:
                self.state["red_bypasses"] += 1
                self.state["battle_score"]["red"] += 1
                if defense.get("auto_patched"):
                    self.state["blue_auto_patches"] += 1
        
        self.state["last_update"] = datetime.now(timezone.utc).isoformat() + "Z"

        # Broadcast to Supabase (triggers Realtime)
        await self.db.update_battle_state(self.state)
        
        log.info(
            f"⚔️ Tick {self.tick}: "
            f"Red {self.state['battle_score']['red']} vs Blue {self.state['battle_score']['blue']} | "
            f"Tier {tier}"
        )

    async def force_cycle(self) -> dict:
        """Force one battle cycle (triggered by dashboard)."""
        await self._run_tick()
        return self.state

    def get_state(self) -> dict:
        return self.state

    def pause(self):
        self.paused = True
        self.state["status"] = "PAUSED"
        log.info("⏸️ Battle Engine paused")

    def resume(self):
        self.paused = False
        self.state["status"] = "ACTIVE"
        log.info("▶️ Battle Engine resumed")

    def stop(self):
        self.running = False
        self.state["status"] = "STOPPED"
        log.info("🛑 Battle Engine stopped")
