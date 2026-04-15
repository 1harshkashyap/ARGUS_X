"""
ARGUS-X — Supabase Client
All database operations. Uses dual-client architecture:
  - _write_client (service role key) → bypasses RLS, writes only
  - _read_client  (anon key)         → respects RLS, reads only
CRITICAL: supabase-py is synchronous, so all calls use run_in_executor.
"""
import os
import asyncio
import json
import logging
from datetime import datetime
from typing import Optional, Dict, List, Any
from functools import partial

log = logging.getLogger("argus.supabase")

# Try to import supabase, graceful fallback if unavailable
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    log.warning("⚠️ supabase-py not installed — running in memory-only mode")


class SupabaseClient:
    """
    Wrapper around Supabase with async support via run_in_executor.
    Enforces least-privilege access:
      - _write_client: service role key (bypasses RLS) — INSERT/UPDATE/UPSERT/RPC only
      - _read_client:  anon key (respects RLS) — SELECT only
    Gracefully degrades if Supabase is unavailable (logs warning, continues).
    """

    def __init__(self):
        self._write_client: Optional[Any] = None
        self._read_client: Optional[Any] = None
        self.available = False

        url = os.getenv("SUPABASE_URL", "")

        # ── Write client (service role key — full admin, bypasses RLS) ────
        # SECURITY: This key must NEVER be exposed to frontend, logs, or
        # error messages. Used exclusively for write operations.
        service_key = os.getenv("SUPABASE_SERVICE_KEY", "")

        # ── Read client (anon key — respects RLS) ────────────────────────
        # Safe for read operations. If leaked, RLS policies still protect data.
        anon_key = os.getenv("SUPABASE_ANON_KEY", "")

        if SUPABASE_AVAILABLE and url and service_key:
            try:
                self._write_client = create_client(url, service_key)
                log.info("✅ Supabase write client connected (service role)")

                if anon_key:
                    self._read_client = create_client(url, anon_key)
                    log.info("✅ Supabase read client connected (anon — RLS enforced)")
                elif os.getenv("ENV", "development").lower() == "production":
                    # SECURITY: In production, NEVER fall back to service key for reads.
                    # This would bypass RLS and expose all data.
                    log.error(
                        "🚨 SUPABASE_ANON_KEY not set in production — reads DISABLED. "
                        "Set SUPABASE_ANON_KEY to enable read operations with RLS."
                    )
                    self._read_client = None
                else:
                    # Dev/demo only: use write client with warning
                    self._read_client = self._write_client
                    log.warning(
                        "⚠️ SUPABASE_ANON_KEY not set — reads will use service role key "
                        "(RLS bypassed). This is ONLY acceptable in development."
                    )

                self.available = True
            except Exception as e:
                log.warning(f"⚠️ Supabase connection failed: {e} — running in memory-only mode")
        else:
            if SUPABASE_AVAILABLE and url and not service_key:
                log.warning("⚠️ SUPABASE_SERVICE_KEY not set — running in memory-only mode")
            else:
                log.warning("⚠️ Supabase credentials not set — running in memory-only mode")

        # In-memory fallback storage
        self._memory_events: List[dict] = []
        self._memory_stats = {
            "total": 0, "blocked": 0, "sanitized": 0, "clean": 0,
            "bypasses_found": 0, "mutations_preblocked": 0, "campaigns_detected": 0
        }

    async def _run_sync(self, func, *args, **kwargs):
        """Run synchronous supabase-py calls in executor to avoid blocking.
        Accepts either a zero-arg callable (lambda/closure) or func + args."""
        loop = asyncio.get_running_loop()
        if args or kwargs:
            return await loop.run_in_executor(None, partial(func, *args, **kwargs))
        return await loop.run_in_executor(None, func)

    # ── Events ────────────────────────────────────────────────────────────────

    async def log_event(self, event: dict):
        """Insert event into events table. Triggers Realtime broadcast."""
        try:
            if self.available:
                row = {
                    "user_id":              event.get("user_id", "anonymous"),
                    "session_id":           event.get("session_id"),
                    "org_id":               event.get("org_id", "default"),
                    "preview":              event.get("preview", ""),
                    "action":               event.get("action", "CLEAN"),
                    "threat_type":          event.get("threat_type"),
                    "score":                event.get("score", 0),
                    "layer":                event.get("layer", "NONE"),
                    "latency_ms":           event.get("latency_ms", 0),
                    "method":               event.get("method", ""),
                    "sophistication_score": event.get("sophistication", 0),
                    "attack_fingerprint":   event.get("fingerprint"),
                    "mutations_preblocked": event.get("mutations_preblocked", 0),
                    "session_threat_level": event.get("session_threat_level", "LOW"),
                    "explanation":          event.get("explanation"),
                }
                await self._run_sync(
                    lambda: self._write_client.table("events").insert(row).execute()
                )
            else:
                self._memory_events.append(event)
                if len(self._memory_events) > 1000:
                    self._memory_events = self._memory_events[-500:]
        except Exception as e:
            log.warning(f"Event log failed: {e}")
            self._memory_events.append(event)

    async def get_recent_events(self, limit: int = 30) -> list:
        """Fetch recent events for dashboard hydration."""
        try:
            if self.available:
                result = await self._run_sync(
                    lambda: self._read_client.table("events")
                        .select("*")
                        .order("created_at", desc=True)
                        .limit(limit)
                        .execute()
                )
                return result.data if result.data else []
            else:
                return self._memory_events[-limit:]
        except Exception as e:
            log.warning(f"Get events failed: {e}")
            return self._memory_events[-limit:]

    # ── Stats ─────────────────────────────────────────────────────────────────

    # Valid stat columns — defense-in-depth against SQL injection.
    # Only these column names are allowed in increment_stat().
    _VALID_STAT_COLUMNS = frozenset({
        "blocked", "sanitized", "clean",
        "bypasses_found", "mutations_preblocked", "campaigns_detected",
    })

    async def increment_stat(self, column: str):
        """Atomically increment a stat column using the DB function.
        Column name is validated against a whitelist before execution."""
        # Defense-in-depth: reject any column not in the whitelist
        if column not in self._VALID_STAT_COLUMNS:
            log.warning(f"🔒 increment_stat rejected invalid column: {column!r}")
            return

        try:
            if self.available:
                await self._run_sync(
                    lambda: self._write_client.rpc("increment_stat", {"col_name": column}).execute()
                )
            else:
                if column in self._memory_stats:
                    self._memory_stats[column] += 1
                if column in ("blocked", "sanitized", "clean"):
                    self._memory_stats["total"] += 1
        except Exception as e:
            log.warning(f"Stat increment failed: {e}")
            if column in self._memory_stats:
                self._memory_stats[column] += 1

    async def get_stats(self) -> dict:
        """Get current stats."""
        try:
            if self.available:
                result = await self._run_sync(
                    lambda: self._read_client.table("stats").select("*").eq("id", 1).execute()
                )
                if result.data:
                    return result.data[0]
            return self._memory_stats
        except Exception as e:
            log.warning(f"Get stats failed: {e}")
            return self._memory_stats

    # ── Red Team Sessions ─────────────────────────────────────────────────────

    async def log_red_team_result(self, result: dict):
        """Log a red team attack result.
        SECURITY: attack_text is hashed before storage — raw payloads are
        never persisted to prevent attack intelligence leakage."""
        try:
            if self.available:
                # Hash attack text — align with events table redaction policy
                raw_text = result.get("attack", "")[:500]
                import hashlib
                text_hash = hashlib.sha256(raw_text.encode()).hexdigest()[:32] if raw_text else ""
                row = {
                    "attack_text":  text_hash + " [HASHED]",
                    "attack_type":  result.get("attack_type"),
                    "tier":         result.get("tier", 1),
                    "bypassed":     result.get("bypassed", False),
                    "auto_patched": result.get("auto_patched", False),
                    "cycle":        result.get("cycle", 0),
                    "score":        result.get("score", 0),
                    "latency_ms":   result.get("latency_ms", 0),
                }
                await self._run_sync(
                    lambda: self._write_client.table("red_team_sessions").insert(row).execute()
                )
        except Exception as e:
            log.warning(f"Red team log failed: {e}")

    async def get_bypass_history(self, limit: int = 10) -> list:
        """Get recent auto-patched bypasses for dashboard display."""
        try:
            if self.available:
                result = await self._run_sync(
                    lambda: self._read_client.table("red_team_sessions")
                        .select("attack_text,attack_type,tier,score,cycle,created_at")
                        .eq("bypassed", True)
                        .eq("auto_patched", True)
                        .order("created_at", desc=True)
                        .limit(limit)
                        .execute()
                )
                return result.data if result.data else []
        except Exception as e:
            log.warning(f"Bypass history fetch failed: {e}")
        return []

    # ── Campaigns ─────────────────────────────────────────────────────────────

    async def log_campaign(self, campaign: dict):
        """Log a detected campaign."""
        try:
            if self.available:
                row = {
                    "campaign_id":     campaign.get("id"),
                    "type":            campaign.get("type"),
                    "threat_type":     campaign.get("threat_type"),
                    "severity":        campaign.get("severity"),
                    "events_count":    campaign.get("events_count", 0),
                    "unique_users":    campaign.get("unique_users", 0),
                    "unique_sessions": campaign.get("unique_sessions", 0),
                    "window_minutes":  campaign.get("window_minutes"),
                    "status":          campaign.get("status", "ACTIVE"),
                }
                await self._run_sync(
                    lambda: self._write_client.table("campaigns").insert(row).execute()
                )
                await self.increment_stat("campaigns_detected")
        except Exception as e:
            log.warning(f"Campaign log failed: {e}")

    # ── XAI Decisions ─────────────────────────────────────────────────────────

    async def log_xai_decision(self, decision: dict):
        """Log an XAI decision for the explainability stream."""
        try:
            if self.available:
                # Serialize complex fields for Supabase
                row = {
                    "attack_preview": decision.get("attack_preview", ""),
                    "verdict": decision.get("verdict", "UNKNOWN"),
                    "sophistication_score": decision.get("sophistication_score", 0),
                    "primary_reason": decision.get("primary_reason", ""),
                    "pattern_family": decision.get("pattern_family", ""),
                    "evolution_note": decision.get("evolution_note", ""),
                    "recommended_action": decision.get("recommended_action", ""),
                    "layer_decisions": json.dumps(decision.get("layer_decisions", [])),
                    "confidence_breakdown": json.dumps(decision.get("confidence_breakdown", {})),
                }
                await self._run_sync(
                    lambda: self._write_client.table("xai_decisions").insert(row).execute()
                )
        except Exception as e:
            log.warning(f"XAI decision log failed: {e}")

    # ── Battle State ──────────────────────────────────────────────────────────

    async def update_battle_state(self, state: dict):
        """Upsert battle state (single row, id=1). Triggers Realtime."""
        try:
            if self.available:
                # Only include columns that exist in the battle_state table
                row = {
                    "id": 1,
                    "tick": state.get("tick", 0),
                    "red_attacks": state.get("red_attacks", 0),
                    "red_bypasses": state.get("red_bypasses", 0),
                    "blue_blocks": state.get("blue_blocks", 0),
                    "blue_auto_patches": state.get("blue_auto_patches", 0),
                    "red_tier": state.get("red_tier", 1),
                    "red_strategy": state.get("red_strategy", "NAIVE"),
                    "mutations_total": state.get("mutations_total", 0),
                    "status": state.get("status", "READY"),
                    "last_update": state.get("last_update"),
                }
                await self._run_sync(
                    lambda: self._write_client.table("battle_state").upsert(row).execute()
                )
        except Exception as e:
            log.warning(f"Battle state update failed: {e}")

    async def get_battle_state(self) -> dict:
        """Get current battle state."""
        try:
            if self.available:
                result = await self._run_sync(
                    lambda: self._read_client.table("battle_state").select("*").eq("id", 1).execute()
                )
                if result.data:
                    return result.data[0]
        except Exception as e:
            log.warning(f"Get battle state failed: {e}")
        return {}

    # ── Dynamic Rules ─────────────────────────────────────────────────────────

    async def add_dynamic_rule(self, pattern: str, threat_type: str, source: str = "MUTATION_ENGINE"):
        """Add a dynamic firewall rule. Deduplicates by pattern via UPSERT."""
        try:
            if self.available:
                await self._run_sync(
                    lambda: self._write_client.table("dynamic_rules").upsert({
                        "pattern": pattern[:500],
                        "threat_type": threat_type,
                        "source": source,
                        "active": True,
                    }, on_conflict="pattern").execute()
                )
        except Exception as e:
            log.warning(f"Dynamic rule add failed: {e}")

    async def get_dynamic_rules(self) -> list:
        """Get all active dynamic rules."""
        try:
            if self.available:
                result = await self._run_sync(
                    lambda: self._read_client.table("dynamic_rules")
                        .select("*")
                        .eq("active", True)
                        .execute()
                )
                return result.data if result.data else []
        except Exception as e:
            log.warning(f"Get dynamic rules failed: {e}")
        return []

    # ── Fingerprints ──────────────────────────────────────────────────────────

    async def upsert_fingerprint(self, fp_id: str, threat_type: str, sophistication: int, explanation: str):
        """Upsert an attack fingerprint."""
        try:
            if self.available:
                await self._run_sync(
                    lambda: self._write_client.rpc("upsert_fingerprint", {
                        "p_fingerprint_id": fp_id,
                        "p_threat_type": threat_type,
                        "p_sophistication": sophistication,
                        "p_explanation": explanation,
                    }).execute()
                )
        except Exception as e:
            log.warning(f"Fingerprint upsert failed: {e}")

    # ── Analytics Queries ─────────────────────────────────────────────────────

    async def get_threat_summary(self) -> list:
        """Get hourly threat summary from the view."""
        try:
            if self.available:
                result = await self._run_sync(
                    lambda: self._read_client.table("threat_summary").select("*").execute()
                )
                return result.data if result.data else []
        except Exception as e:
            log.warning(f"Threat summary failed: {e}")
        return []

    async def get_attack_heatmap(self) -> list:
        """Get attack heatmap data from the view."""
        try:
            if self.available:
                result = await self._run_sync(
                    lambda: self._read_client.table("attack_heatmap").select("*").execute()
                )
                return result.data if result.data else []
        except Exception as e:
            log.warning(f"Heatmap failed: {e}")
        return []

    async def get_recent_xai_decisions(self, limit: int = 20) -> list:
        """Get recent XAI decisions for the stream."""
        try:
            if self.available:
                result = await self._run_sync(
                    lambda: self._read_client.table("xai_decisions")
                        .select("*")
                        .order("created_at", desc=True)
                        .limit(limit)
                        .execute()
                )
                return result.data if result.data else []
        except Exception as e:
            log.warning(f"XAI decisions fetch failed: {e}")
        return []

    async def get_battle_history(self, limit: int = 20) -> list:
        """Get recent battle ticks from database (read-only, RLS-respecting)."""
        try:
            if self.available:
                result = await self._run_sync(
                    lambda: self._read_client.table("battle_state")
                        .select("*")
                        .order("last_update", desc=True)
                        .limit(limit)
                        .execute()
                )
                return result.data if result.data else []
        except Exception as e:
            log.warning(f"Battle history fetch failed: {e}")
        return []

    # ── Evolution Persistence ─────────────────────────────────────────────────

    async def log_evolution_snapshot(self, window_avg: float, trend: str,
                                     data_points: int, escalation_count: int):
        """Persist an evolution window snapshot to evolution_log table.
        Called every window_size events to preserve trend history across restarts."""
        try:
            if self.available:
                row = {
                    "window_avg": round(window_avg, 4),
                    "trend": trend,
                    "data_points": data_points,
                    "escalation_count": escalation_count,
                }
                await self._run_sync(
                    lambda: self._write_client.table("evolution_log").insert(row).execute()
                )
        except Exception as e:
            log.warning(f"Evolution snapshot log failed: {e}")

    # ── Cluster Persistence ───────────────────────────────────────────────────

    async def batch_upsert_clusters(self, clusters: list):
        """Persist current cluster state to threat_clusters table.
        Called after each recluster to preserve cluster intelligence across restarts."""
        try:
            if self.available and clusters:
                rows = []
                for c in clusters[:50]:  # Cap batch size
                    rows.append({
                        "cluster_id": c.get("cluster_id", ""),
                        "cluster_size": c.get("size", 0),
                        "dominant_type": c.get("dominant_type", ""),
                        "avg_sophistication": c.get("avg_sophistication", 0),
                        "sample_text": c.get("sample_hash", ""),  # Hash only, never raw
                        "types": json.dumps(c.get("types", [])),
                    })
                await self._run_sync(
                    lambda: self._write_client.table("threat_clusters").insert(rows).execute()
                )
        except Exception as e:
            log.warning(f"Cluster batch upsert failed: {e}")

    async def get_latest_evolution(self) -> dict:
        """Load the most recent evolution snapshot from DB for startup hydration."""
        try:
            if self.available:
                result = await self._run_sync(
                    lambda: self._read_client.table("evolution_log")
                        .select("*")
                        .order("created_at", desc=True)
                        .limit(1)
                        .execute()
                )
                if result.data:
                    return result.data[0]
        except Exception as e:
            log.warning(f"Evolution load failed: {e}")
        return {}

    async def get_latest_clusters(self) -> list:
        """Load the most recent cluster snapshots from DB for startup hydration."""
        try:
            if self.available:
                result = await self._run_sync(
                    lambda: self._read_client.table("threat_clusters")
                        .select("*")
                        .order("created_at", desc=True)
                        .limit(50)
                        .execute()
                )
                return result.data if result.data else []
        except Exception as e:
            log.warning(f"Clusters load failed: {e}")
        return []

