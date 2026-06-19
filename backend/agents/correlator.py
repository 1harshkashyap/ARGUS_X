import asyncio
import time
import hashlib
from datetime import datetime, timezone
from typing import Dict, Set, List, Optional
from utils.logger import logger
from utils.db import add_campaign


# ── Constants ─────────────────────────────────────────────────────────

# How many distinct sessions hitting same pattern = campaign
_CAMPAIGN_THRESHOLD = 3

# Maximum sessions tracked per pattern family (prevents memory growth)
_MAX_SESSIONS_PER_PATTERN = 100

# Maximum distinct patterns tracked (prevents unbounded growth)
_MAX_PATTERNS = 500

# TTL for pattern tracking — clear patterns older than this (seconds)
_PATTERN_TTL = 3600  # 1 hour


# ── Correlator ────────────────────────────────────────────────────────

class ThreatCorrelator:
    """
    In-memory campaign detector.

    Tracks which sessions have sent attacks of each pattern family.
    When 3+ distinct sessions hit the same pattern: creates a campaign.

    Memory safety:
      - _MAX_SESSIONS_PER_PATTERN caps sessions per pattern
      - _MAX_PATTERNS caps total pattern tracking
      - _cleanup() called every 100 events to remove stale patterns
      - TTL-based expiry prevents unbounded growth

    Thread safety:
      - Single asyncio event loop — no locks needed
      - All methods are synchronous (no await)
      - Campaigns are written to DB via async helper in chat.py context
    """

    def __init__(self) -> None:
        # pattern_family → set of session_ids
        self._pattern_sessions: Dict[str, Set[str]] = {}
        # pattern_family → last_seen timestamp
        self._pattern_timestamps: Dict[str, float] = {}
        # campaign_ids already alerted (prevents duplicate campaign writes)
        self._alerted_campaigns: Set[str] = set()
        self._event_count: int = 0

    def record(
        self,
        session_id: str,
        pattern_family: str,
        attack_fingerprint: str = ""
    ) -> Optional[Dict]:
        """
        Record a threat event. Returns campaign dict if one is detected,
        otherwise None.

        Args:
            session_id:        Session that sent the attack
            pattern_family:    Threat type (PROMPT_INJECTION, JAILBREAK, etc.)
            attack_fingerprint: SHA256 fingerprint from fingerprinter

        Returns:
            Campaign dict if 3+ sessions hit same pattern, else None.
        """
        try:
            if not session_id or not pattern_family:
                return None

            self._event_count += 1

            # ── Memory guard: max patterns ────────────────────────
            if (pattern_family not in self._pattern_sessions
                    and len(self._pattern_sessions) >= _MAX_PATTERNS):
                logger.warning(
                    f"Correlator: max patterns ({_MAX_PATTERNS}) reached — "
                    f"dropping oldest to make room"
                )
                oldest = min(
                    self._pattern_timestamps,
                    key=lambda k: self._pattern_timestamps[k]
                )
                self._pattern_sessions.pop(oldest, None)
                self._pattern_timestamps.pop(oldest, None)

            # ── Track session for this pattern ────────────────────
            if pattern_family not in self._pattern_sessions:
                self._pattern_sessions[pattern_family] = set()

            sessions = self._pattern_sessions[pattern_family]

            # Memory guard: max sessions per pattern
            if len(sessions) < _MAX_SESSIONS_PER_PATTERN:
                sessions.add(session_id)

            self._pattern_timestamps[pattern_family] = time.monotonic()

            # ── Check campaign threshold ──────────────────────────
            if len(sessions) >= _CAMPAIGN_THRESHOLD:
                campaign_id = self._campaign_id(pattern_family)
                now_iso = datetime.now(timezone.utc).isoformat()

                is_new = campaign_id not in self._alerted_campaigns
                if is_new:
                    self._alerted_campaigns.add(campaign_id)

                # Always build campaign dict with current counts + timestamp
                campaign: Dict = {
                    "campaign_id":     campaign_id,
                    "attack_pattern":  pattern_family,
                    "source_sessions": list(sessions)[:20],
                    "hit_count":       len(sessions),
                    "severity":        self._severity(len(sessions)),
                    "last_seen":       now_iso,
                }

                if is_new:
                    campaign["first_seen"] = now_iso
                    logger.warning(
                        f"Correlator: CAMPAIGN DETECTED "
                        f"pattern={pattern_family} "
                        f"sessions={len(sessions)} "
                        f"id={campaign_id}"
                    )

                # Return campaign for DB upsert (updates last_seen + hit_count)
                return campaign

            # ── Periodic cleanup ──────────────────────────────────
            if self._event_count % 100 == 0:
                self._cleanup()

            return None

        except Exception as e:
            logger.warning(f"Correlator.record error (non-fatal): {type(e).__name__}")
            return None

    def get_stats(self) -> Dict:
        """Return correlator statistics."""
        return {
            "tracked_patterns":  len(self._pattern_sessions),
            "alerted_campaigns": len(self._alerted_campaigns),
            "total_events":      self._event_count,
        }

    def _campaign_id(self, pattern_family: str) -> str:
        """Generate a stable campaign ID from the pattern family."""
        return hashlib.sha256(
            pattern_family.encode("utf-8")
        ).hexdigest()[:12].upper()

    def _severity(self, session_count: int) -> str:
        """Derive severity from how many sessions are involved."""
        if session_count >= 10:
            return "CRITICAL"
        if session_count >= 6:
            return "HIGH"
        if session_count >= 3:
            return "MEDIUM"
        return "LOW"

    def _cleanup(self) -> None:
        """Remove patterns older than _PATTERN_TTL."""
        now = time.monotonic()
        cutoff = now - _PATTERN_TTL
        expired = [
            pattern for pattern, ts in self._pattern_timestamps.items()
            if ts < cutoff
        ]
        for pattern in expired:
            self._pattern_sessions.pop(pattern, None)
            self._pattern_timestamps.pop(pattern, None)
        if expired:
            logger.info(f"Correlator cleanup: removed {len(expired)} expired patterns")


# ── Global singleton ──────────────────────────────────────────────────
correlator = ThreatCorrelator()


# ── Async helper called from chat pipeline ────────────────────────────

async def check_and_record_campaign(
    session_id: str,
    pattern_family: str,
    attack_fingerprint: str = ""
) -> None:
    """
    Record a threat event and write campaign to Supabase if detected.
    Fire-and-forget — called as asyncio.create_task().
    Never raises.
    """
    try:
        campaign = correlator.record(
            session_id=session_id,
            pattern_family=pattern_family,
            attack_fingerprint=attack_fingerprint
        )
        if campaign:
            await add_campaign(campaign)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.warning(f"check_and_record_campaign error (non-fatal): {type(e).__name__}")
