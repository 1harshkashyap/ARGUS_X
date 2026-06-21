import time
from typing import Dict, Optional
from dataclasses import dataclass, field
from utils.logger import logger


@dataclass
class SessionData:
    """Data tracked per user session."""
    session_id: str = ""
    total_requests: int = 0
    threat_count: int = 0
    last_seen: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)
    escalation_count: int = 0  # Consecutive threats (resets on clean request)
    last_threat_type: str = "CLEAN"


class SessionTracker:
    """
    In-memory session threat level tracker.

    Tracks per-session threat ratios to detect:
    - Sessions sending repeated attacks
    - Progressive escalation attacks
    - Campaign-coordinated attacks

    Memory-safe: auto-cleans sessions older than 1 hour.
    Thread-safe for asyncio (single event loop).
    """

    # Threat level thresholds (ratio of threats to total requests)
    _THRESHOLDS = {
        "LOW":      0.0,   # 0% threats
        "MEDIUM":   0.25,  # 25%+ threats
        "HIGH":     0.50,  # 50%+ threats
        "CRITICAL": 0.75,  # 75%+ threats
    }

    # Clean up sessions older than this (seconds)
    _TTL_SECONDS = 3600  # 1 hour

    # Clean up every N requests to avoid overhead
    _CLEANUP_EVERY = 100

    def __init__(self):
        self._sessions: Dict[str, SessionData] = {}
        self._request_count = 0

    def get_level(self, session_id: str) -> str:
        """
        Get current threat level for a session.
        Returns: LOW | MEDIUM | HIGH | CRITICAL
        Always returns a string — never None.
        """
        if not session_id or session_id not in self._sessions:
            return "LOW"

        session = self._sessions[session_id]
        if session.total_requests == 0:
            return "LOW"

        ratio = session.threat_count / session.total_requests

        # Consecutive-threat escalation boosts the effective ratio
        # 5+ consecutive threats in a row bumps up by one tier
        if session.escalation_count >= 5:
            ratio = min(1.0, ratio + 0.25)

        if ratio >= self._THRESHOLDS["CRITICAL"]:
            return "CRITICAL"
        elif ratio >= self._THRESHOLDS["HIGH"]:
            return "HIGH"
        elif ratio >= self._THRESHOLDS["MEDIUM"]:
            return "MEDIUM"
        return "LOW"

    def update(
        self,
        session_id: str,
        was_threat: bool,
        threat_type: str = "CLEAN"
    ) -> str:
        """
        Update session data after a request.
        Returns the new threat level.
        """
        if not session_id:
            return "LOW"

        self._request_count += 1

        # Create or update session
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionData(session_id=session_id)

        session = self._sessions[session_id]
        session.total_requests += 1
        session.last_seen = time.time()

        if was_threat:
            session.threat_count += 1
            session.last_threat_type = threat_type
            session.escalation_count += 1
        else:
            session.escalation_count = 0  # Reset on clean request

        new_level = self.get_level(session_id)

        # Log escalations
        if new_level in ("HIGH", "CRITICAL"):
            logger.warning(
                f"Session {session_id[:8]}... threat level: {new_level} "
                f"({session.threat_count}/{session.total_requests} requests)"
            )

        # Periodic cleanup to prevent memory growth
        if self._request_count % self._CLEANUP_EVERY == 0:
            self._cleanup()

        return new_level

    def get_session(self, session_id: str) -> Optional[SessionData]:
        """Get raw session data for a session ID."""
        return self._sessions.get(session_id)

    def get_stats(self) -> Dict:
        """Get overall session statistics."""
        return {
            "active_sessions": len(self._sessions),
            "total_tracked_requests": self._request_count,
            "high_risk_sessions": sum(
                1 for sid in self._sessions
                if self.get_level(sid) in ("HIGH", "CRITICAL")
            )
        }

    def _cleanup(self):
        """Remove sessions older than TTL to prevent memory growth."""
        now = time.time()
        cutoff = now - self._TTL_SECONDS
        expired = [
            sid for sid, data in self._sessions.items()
            if data.last_seen < cutoff
        ]
        for sid in expired:
            del self._sessions[sid]
        if expired:
            logger.info(f"Session cleanup: removed {len(expired)} expired sessions")


# ── Global singleton ──────────────────────────────────────────────────
session_tracker = SessionTracker()
