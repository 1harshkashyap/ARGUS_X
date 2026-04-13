"""
ARGUS-X — Threat Clusterer (Layer 6)
Uses DBSCAN on sentence embeddings to group attacks into semantic families.
Re-clusters every 10 attacks for real-time intelligence.
CRITICAL: SBERT encoding and DBSCAN fit are CPU-bound — offloaded via asyncio.to_thread().
"""
import asyncio
import logging
import hashlib
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict

log = logging.getLogger("argus.clusterer")

# Try sklearn import
try:
    from sklearn.cluster import DBSCAN
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    log.warning("⚠️ scikit-learn not installed — using keyword clustering fallback")

# Try sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    log.warning("⚠️ sentence-transformers not installed — using keyword clustering")


class ThreatClusterer:
    """
    Groups attacks into semantic families using DBSCAN + sentence embeddings.
    Falls back to keyword-based clustering if ML libraries unavailable.
    """

    def __init__(self, models=None):
        self.models = models
        self.encoder = None
        self.attacks: List[dict] = []
        self.embeddings: List = []
        self.clusters: List[dict] = []
        self.recluster_interval = 10
        self._attack_count = 0
        
        # Try to load sentence transformer
        if SBERT_AVAILABLE:
            try:
                self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
                log.info("✅ Threat Clusterer initialized (SBERT + DBSCAN mode)")
            except Exception as e:
                log.warning(f"⚠️ SBERT load failed: {e} — using keyword fallback")
        else:
            log.info("✅ Threat Clusterer initialized (keyword clustering mode)")

    async def ingest(self, text: str, threat_type: str, sophistication: int = 0):
        """Ingest a new attack for clustering."""
        self.attacks.append({
            "text": text[:300],
            "threat_type": threat_type,
            "sophistication": sophistication,
            "ts": datetime.utcnow().isoformat() + "Z",
        })
        self._attack_count += 1

        # Compute embedding if encoder available
        # CRITICAL: SBERT encode() is CPU-bound (30-100ms+) — offload to thread
        if self.encoder:
            try:
                embedding = await asyncio.to_thread(self.encoder.encode, text[:300])
                self.embeddings.append(embedding)
            except Exception as e:
                log.warning(f"Embedding failed: {e}")

        # Re-cluster every N attacks
        # CRITICAL: DBSCAN fit() is CPU-heavy — offload to thread
        if self._attack_count % self.recluster_interval == 0:
            await asyncio.to_thread(self._recluster)

        # Keep bounded
        if len(self.attacks) > 2000:
            self.attacks = self.attacks[-1000:]
            self.embeddings = self.embeddings[-1000:] if self.embeddings else []

    def _recluster(self):
        """Run DBSCAN clustering on accumulated attacks."""
        if SKLEARN_AVAILABLE and self.embeddings and len(self.embeddings) >= 5:
            try:
                X = np.array(self.embeddings)
                clustering = DBSCAN(eps=0.5, min_samples=3, metric="cosine").fit(X)
                labels = clustering.labels_
                
                # Build cluster summary
                cluster_groups = defaultdict(list)
                for i, label in enumerate(labels):
                    if label != -1:  # Skip noise
                        cluster_groups[label].append(self.attacks[i])

                self.clusters = []
                for label, members in cluster_groups.items():
                    types = [m["threat_type"] for m in members]
                    most_common = max(set(types), key=types.count)
                    avg_soph = sum(m["sophistication"] for m in members) / len(members)
                    
                    self.clusters.append({
                        "cluster_id": f"C{label}",
                        "size": len(members),
                        "dominant_type": most_common,
                        "avg_sophistication": round(avg_soph, 1),
                        "sample": members[0]["text"][:100],
                        "types": list(set(types)),
                    })
                
                log.info(f"📊 Re-clustered: {len(self.clusters)} clusters from {len(self.attacks)} attacks")
            except Exception as e:
                log.warning(f"DBSCAN clustering failed: {e}")
                self._keyword_cluster()
        else:
            self._keyword_cluster()

    def _keyword_cluster(self):
        """Fallback: cluster by threat_type (keyword-based)."""
        groups = defaultdict(list)
        for attack in self.attacks:
            groups[attack["threat_type"]].append(attack)
        
        self.clusters = []
        for threat_type, members in groups.items():
            avg_soph = sum(m["sophistication"] for m in members) / max(len(members), 1)
            self.clusters.append({
                "cluster_id": f"K-{threat_type[:8]}",
                "size": len(members),
                "dominant_type": threat_type,
                "avg_sophistication": round(avg_soph, 1),
                "sample": members[-1]["text"][:100] if members else "",
                "types": [threat_type],
            })

    def get_clusters(self) -> List[dict]:
        """Return current cluster summary."""
        return self.clusters

    def get_cluster_summary(self) -> dict:
        """High-level summary for analytics."""
        return {
            "total_attacks_ingested": len(self.attacks),
            "total_clusters": len(self.clusters),
            "clusters": self.clusters[:20],
            "mode": "DBSCAN" if (SKLEARN_AVAILABLE and self.encoder) else "KEYWORD",
        }
