// ═══════════════════════════════════════════════════════════════════════
// ARGUS-X — useStats Hook
// Polls /api/v1/analytics/stats every 3s for dashboard statistics
// ═══════════════════════════════════════════════════════════════════════

import { useState, useEffect, useRef } from 'react';
import type { Stats, AlertEntry } from '../types';
import { API_URL } from '../utils/config';
import type { PatchEvent } from '../components/PatchBanner';

interface StatsState {
  stats: Stats;
  tier: number;
  threatLevel: number;
  campaignCount: number;
  threatVelocity: Record<string, number>;
  lastPatch: PatchEvent | null;
  showPatch: boolean;
  showAlert: boolean;
  alertMsg: string;
  alertHistory: AlertEntry[];
  showAlertHistory: boolean;
  setShowAlertHistory: (v: boolean | ((prev: boolean) => boolean)) => void;
}

export function useStats(): StatsState {
  const [stats, setStats] = useState<Stats>({
    total: 0,
    blocked: 0,
    muts: 0,
    bypasses: 0,
    patched: 0,
  });
  const [tier, setTier] = useState(1);
  const [threatLevel, setThreatLevel] = useState(1);
  const [campaignCount, setCampaignCount] = useState(0);
  const [showAlert, setShowAlert] = useState(false);
  const [alertMsg, setAlertMsg] = useState('');
  const [alertHistory, setAlertHistory] = useState<AlertEntry[]>([]);
  const [showAlertHistory, setShowAlertHistory] = useState(false);
  const [threatVelocity, setThreatVelocity] = useState<Record<string, number>>({});
  const [lastPatch, setLastPatch] = useState<PatchEvent | null>(null);
  const [showPatch, setShowPatch] = useState(false);
  const lastPatchTsRef = useRef<string>('');

  useEffect(() => {
    let active = true;
    const controller = new AbortController();

    const BASE_INTERVAL = 3000;
    const MAX_INTERVAL = 30000;
    let currentInterval = BASE_INTERVAL;

    async function poll() {
      try {
        const apiKey = localStorage.getItem('ARGUS_API_KEY') || '';
        const headers: Record<string, string> = {};
        if (apiKey) headers['X-API-Key'] = apiKey;
        const res = await fetch(`${API_URL}/api/v1/analytics/stats`, {
          signal: controller.signal,
          headers,
        });
        if (!active) return;
        if (!res.ok) {
          // Non-2xx: backoff but keep polling
          currentInterval = Math.min(currentInterval * 2, MAX_INTERVAL);
          return;
        }
        const data = await res.json();
        if (!active) return;

        // Reset backoff on success
        currentInterval = BASE_INTERVAL;

        setStats({
          total: data.total || 0,
          blocked: data.blocked || 0,
          muts: data.mutations_preblocked || 0,
          bypasses: data.sanitized || 0,
          patched: data.sanitized || 0,
        });

        if (data.battle) {
          setTier(data.battle.red_tier || 1);
        }

        if (data.threat_velocity && typeof data.threat_velocity === 'object') {
          setThreatVelocity(data.threat_velocity);
        }

        if (data.evolution && typeof data.evolution.threat_level === 'number') {
          setThreatLevel(data.evolution.threat_level);
        } else if (data.total > 0) {
          const bypassRate = (data.sanitized || 0) / data.total;
          setThreatLevel(Math.min(5, Math.round(bypassRate * 10)));
        }

        if (data.campaigns && data.campaigns.length > 0) {
          setCampaignCount(data.campaigns.length);
          const latest = data.campaigns[data.campaigns.length - 1];
          if (latest && latest.detected_at) {
            const age = Date.now() - new Date(latest.detected_at).getTime();
            if (age < 10000) {
              const msg = `CAMPAIGN DETECTED: ${(latest.pattern || 'UNKNOWN').replace(/_/g, ' ')} · ${latest.event_count || '?'} events from ${latest.source_count || '?'} unique sources`;
              setShowAlert(true);
              setAlertMsg(msg);
              setAlertHistory((prev) => [
                {
                  ts: new Date().toLocaleTimeString('en-US', { hour12: false }),
                  msg,
                },
                ...prev.slice(0, 19),
              ]);
              setTimeout(() => setShowAlert(false), 5000);
            }
          }
        }

        // Detect new auto-patch events from the red agent
        if (data.agent?.last_patch?.ts && data.agent.last_patch.ts !== lastPatchTsRef.current) {
          lastPatchTsRef.current = data.agent.last_patch.ts;
          setLastPatch(data.agent.last_patch as PatchEvent);
          setShowPatch(true);
          setTimeout(() => setShowPatch(false), 8000);
        }
      } catch (err: unknown) {
        // Aborted fetches are expected during cleanup — don't backoff
        if (err instanceof DOMException && err.name === 'AbortError') return;

        // Exponential backoff: 3s → 6s → 12s → 24s → capped at 30s
        currentInterval = Math.min(currentInterval * 2, MAX_INTERVAL);
      } finally {
        // Schedule next poll ALWAYS (unless component unmounted)
        if (active) {
          setTimeout(poll, currentInterval);
        }
      }
    }

    poll();
    return () => {
      active = false;
      controller.abort();
    };
  }, []);

  return {
    stats,
    tier,
    threatLevel,
    campaignCount,
    threatVelocity,
    lastPatch,
    showPatch,
    showAlert,
    alertMsg,
    alertHistory,
    showAlertHistory,
    setShowAlertHistory,
  };
}
