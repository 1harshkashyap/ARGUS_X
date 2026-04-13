// ═══════════════════════════════════════════════════════════════════════
// ARGUS-X — useStats Hook
// Polls /api/v1/analytics/stats every 3s for dashboard statistics
// ═══════════════════════════════════════════════════════════════════════

import { useState, useEffect } from 'react';
import type { Stats, AlertEntry } from '../types';
import { API_URL } from '../utils/config';

interface StatsState {
  stats: Stats;
  tier: number;
  threatLevel: number;
  campaignCount: number;
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

  useEffect(() => {
    let active = true;

    async function poll() {
      try {
        const res = await fetch(`${API_URL}/api/v1/analytics/stats`);
        if (!res.ok || !active) return;
        const data = await res.json();
        if (!active) return;

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
      } catch {
        /* silently retry next cycle */
      }
    }

    poll();
    const interval = setInterval(poll, 3000);
    return () => {
      active = false;
      clearInterval(interval);
    };
  }, []);

  return {
    stats,
    tier,
    threatLevel,
    campaignCount,
    showAlert,
    alertMsg,
    alertHistory,
    showAlertHistory,
    setShowAlertHistory,
  };
}
