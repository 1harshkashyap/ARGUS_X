"""
ARGUS-X — Evolution Tracker (Layer 6)
Tracks sophistication trends over time. Detects escalation patterns.
Auto-tightens firewall thresholds when attacks are getting smarter.
"""
import logging
import time
from datetime import datetime
from collections import deque
from typing import Dict, List, Optional

log = logging.getLogger("argus.evolution")

class EvolutionTracker:
    """
    Sliding window tracker for attack sophistication trends.
    Detects when attackers are escalating and auto-adjusts defenses.
    """

    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.scores: deque = deque(maxlen=window_size)
        self.history: List[dict] = []  # Full log for analytics
        self.trend_windows: List[float] = []  # Average per window
        self.escalation_count = 0
        self.last_avg = 0.0
        # Cached result for should_tighten_firewall (avoid recomputation per request)
        self._tighten_cache: bool = False
        self._tighten_cache_ts: float = 0.0
        self._TIGHTEN_CACHE_TTL: float = 30.0  # seconds
        log.info("✅ Evolution Tracker initialized")

    def record(self, sophistication_score: int, threat_type: str, tier: int = 0):
        """Record a new sophistication data point."""
        entry = {
            "score": sophistication_score,
            "threat_type": threat_type,
            "tier": tier,
            "ts": datetime.utcnow().isoformat() + "Z",
        }
        self.scores.append(sophistication_score)
        self.history.append(entry)
        
        # Keep history bounded
        if len(self.history) > 5000:
            self.history = self.history[-2500:]

        # Recalculate trend every window_size events
        if len(self.scores) == self.window_size:
            current_avg = sum(self.scores) / len(self.scores)
            self.trend_windows.append(current_avg)
            
            # Check for escalation
            if current_avg > self.last_avg + 0.5:
                self.escalation_count += 1
                log.warning(f"📈 Sophistication escalation detected: {self.last_avg:.1f} → {current_avg:.1f}")
            
            self.last_avg = current_avg

    def get_evolution_report(self) -> Dict:
        """Get current evolution analysis."""
        if not self.scores:
            return {
                "current_avg": 0,
                "trend": "STABLE",
                "threat_level": 1,
                "is_escalating": False,
                "escalation_count": 0,
                "window_size": self.window_size,
                "data_points": 0,
                "recommendation": "Insufficient data",
            }

        current_avg = sum(self.scores) / len(self.scores)
        
        # Determine trend
        is_escalating = False
        trend = "STABLE"
        
        if len(self.trend_windows) >= 3:
            recent = self.trend_windows[-3:]
            if all(recent[i] < recent[i+1] for i in range(len(recent)-1)):
                trend = "ESCALATING"
                is_escalating = True
            elif all(recent[i] > recent[i+1] for i in range(len(recent)-1)):
                trend = "DECLINING"
            elif recent[-1] - recent[0] > 1.0:
                trend = "RISING"
                is_escalating = True
        elif len(self.trend_windows) >= 2:
            if self.trend_windows[-1] > self.trend_windows[-2] + 0.5:
                trend = "RISING"
                is_escalating = True

        recommendation = "Continue monitoring."
        if is_escalating and self.escalation_count >= 3:
            recommendation = "⚠️ TIGHTEN ML THRESHOLD — sustained sophistication escalation detected."
        elif is_escalating:
            recommendation = "Escalation detected. Monitor closely."

        return {
            "current_avg": round(current_avg, 2),
            "trend": trend,
            "threat_level": min(5, max(1, round(current_avg / 2))) if current_avg > 0 else 1,
            "is_escalating": is_escalating,
            "escalation_count": self.escalation_count,
            "window_size": self.window_size,
            "data_points": len(self.history),
            "recent_scores": list(self.scores)[-20:],
            "trend_windows": self.trend_windows[-10:],
            "recommendation": recommendation,
        }

    def should_tighten_firewall(self) -> bool:
        """Returns True if firewall threshold should be tightened.
        Result is cached for 30s to avoid recomputing on every request."""
        now = time.time()
        if now - self._tighten_cache_ts < self._TIGHTEN_CACHE_TTL:
            return self._tighten_cache
        report = self.get_evolution_report()
        self._tighten_cache = report["is_escalating"] and self.escalation_count >= 3
        self._tighten_cache_ts = now
        return self._tighten_cache

    def get_sparkline_data(self, points: int = 30) -> List[float]:
        """Get last N average scores for sparkline visualization."""
        if len(self.history) < 2:
            return [0.0] * points
        
        scores = [h["score"] for h in self.history[-points*3:]]
        # Downsample to `points` values
        if len(scores) <= points:
            return scores
        
        chunk_size = len(scores) // points
        result = []
        for i in range(0, len(scores), chunk_size):
            chunk = scores[i:i+chunk_size]
            result.append(round(sum(chunk) / len(chunk), 1))
        return result[:points]
