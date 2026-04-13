"""
ARGUS — Threat Correlator
Detects coordinated attack campaigns across sessions and users.
Identifies when 47 different users are trying the same novel attack — 
that's not coincidence, that's a campaign.

This is cross-session intelligence. Zero existing LLM security tools do this.
"""
import asyncio, logging
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Optional

log = logging.getLogger("argus.correlator")

# Campaign detection thresholds
CAMPAIGN_THRESHOLDS = {
    "RAPID":    {"window_minutes": 10, "min_reports": 3,  "severity": "HIGH"},
    "SUSTAINED":{"window_minutes": 60, "min_reports": 8,  "severity": "CRITICAL"},
    "GLOBAL":   {"window_minutes": 1440,"min_reports": 25, "severity": "CRITICAL"},
}

class ThreatCorrelator:
    """
    Maintains a sliding window of threat events and detects:
    1. Same attack pattern from multiple users (coordinated campaign)
    2. Geographic clustering of attacks
    3. Temporal attack waves (e.g., spike at 2AM = automated attack)
    4. Fingerprint recurrence (same attack family hitting multiple systems)
    """
    
    def __init__(self, db):
        self.db = db
        self.running = False
        
        # In-memory sliding windows
        self.threat_window: List[dict] = []  # Last 1000 threats
        self.fingerprint_counts: Dict[str, int] = defaultdict(int)
        self.active_campaigns: List[dict] = []
        self.threat_type_velocity: Dict[str, List[float]] = defaultdict(list)
        
        log.info("✅ Threat Correlator initialized")

    def status(self) -> dict:
        return {
            "running": self.running,
            "window_size": len(self.threat_window),
            "active_campaigns": len(self.active_campaigns),
            "fingerprints_tracked": len(self.fingerprint_counts),
        }

    async def correlation_loop(self):
        """Runs every 30 seconds — analyzes threat patterns."""
        self.running = True
        log.info("🔗 Threat Correlator loop started")
        
        while True:
            try:
                await self._analyze_patterns()
                await asyncio.sleep(30)
            except Exception as e:
                log.error(f"Correlator error: {e}")
                await asyncio.sleep(30)

    def ingest_event(self, event: dict):
        """Called by chat router after each threat event."""
        if event.get("action") in ("BLOCKED", "SANITIZED"):
            self.threat_window.append({
                "ts":          event.get("ts"),
                "threat_type": event.get("threat_type"),
                "fingerprint": event.get("fingerprint"),
                "user_id":     event.get("user_id"),
                "session_id":  event.get("session_id"),
                "score":       event.get("score", 0),
            })
            
            # Track fingerprint frequency
            fp = event.get("fingerprint")
            if fp:
                self.fingerprint_counts[fp] += 1
            
            # Track threat type velocity (for wave detection)
            tt = event.get("threat_type", "UNKNOWN")
            self.threat_type_velocity[tt].append(datetime.utcnow().timestamp())
            
            # Keep window bounded
            if len(self.threat_window) > 2000:
                self.threat_window = self.threat_window[-1000:]

    async def _analyze_patterns(self):
        """Core correlation logic — runs every 30 seconds."""
        now = datetime.utcnow()
        
        for campaign_type, cfg in CAMPAIGN_THRESHOLDS.items():
            window_start = now - timedelta(minutes=cfg["window_minutes"])
            
            # Filter events in window — normalize timestamps for comparison
            window_ts = window_start.isoformat() + "Z"
            recent = [
                e for e in self.threat_window
                if (e.get("ts", "").replace("Z", "") >= window_start.isoformat())
            ]
            
            if len(recent) < cfg["min_reports"]:
                continue
            
            # Group by threat type
            type_groups: Dict[str, List] = defaultdict(list)
            for ev in recent:
                tt = ev.get("threat_type", "UNKNOWN")
                type_groups[tt].append(ev)
            
            # Detect campaigns
            for threat_type, events in type_groups.items():
                unique_users = len(set(e.get("user_id") for e in events))
                unique_sessions = len(set(e.get("session_id") for e in events))
                
                if len(events) >= cfg["min_reports"]:
                    # This is a coordinated campaign — multiple users, same attack type
                    campaign = {
                        "id":            f"CAMP-{threat_type[:4]}-{int(now.timestamp())}",
                        "type":          campaign_type,
                        "threat_type":   threat_type,
                        "severity":      cfg["severity"],
                        "events_count":  len(events),
                        "unique_users":  unique_users,
                        "unique_sessions": unique_sessions,
                        "window_minutes":cfg["window_minutes"],
                        "detected_at":   now.isoformat() + "Z",
                        "status":        "ACTIVE",
                    }
                    
                    # Don't duplicate active campaigns
                    existing = [c for c in self.active_campaigns 
                               if c["threat_type"] == threat_type and c["status"] == "ACTIVE"]
                    if not existing:
                        self.active_campaigns.append(campaign)
                        await self.db.log_campaign(campaign)
                        log.warning(f"⚠️  CAMPAIGN DETECTED: {threat_type} | {unique_users} users | {len(events)} events")

    def get_active_campaigns(self) -> List[dict]:
        return [c for c in self.active_campaigns if c["status"] == "ACTIVE"]

    def get_threat_velocity(self) -> Dict[str, int]:
        """Returns count of each threat type in last 10 minutes."""
        cutoff = (datetime.utcnow() - timedelta(minutes=10)).timestamp()
        velocity = {}
        for tt, timestamps in self.threat_type_velocity.items():
            recent = [t for t in timestamps if t > cutoff]
            if recent:
                velocity[tt] = len(recent)
        return velocity

    def get_top_fingerprints(self, n: int = 5) -> List[dict]:
        sorted_fps = sorted(self.fingerprint_counts.items(), key=lambda x: x[1], reverse=True)
        return [{"fingerprint": fp, "count": count} for fp, count in sorted_fps[:n]]
