// ═══════════════════════════════════════════════════════════════════════
// ARGUS-X — useRealtimeFeed Hook
// WebSocket connection to /ws/live with reconnection + event normalization
// ═══════════════════════════════════════════════════════════════════════

import { useState, useEffect } from 'react';
import type { AttackEvent, LogEntry, WSMessage } from '../types';
import { WS_URL } from '../utils/config';
import { normalizeEvent } from '../utils/sanitize';
import { uid } from '../utils/helpers';
import { getApiKey } from '../utils/apiKey';

interface CampaignWsAlert {
  pattern: string;
  eventCount: number;
  sourceCount: number;
  severity: string;
}

interface RealtimeFeedState {
  attacks: AttackEvent[];
  defenseLog: LogEntry[];
  lastUpdated: Date | null;
  sophHistory: number[];
  latHistory: number[];
  campaignWsAlert: CampaignWsAlert | null;
  connected: boolean;
}

export function useRealtimeFeed(): RealtimeFeedState {
  const [attacks, setAttacks] = useState<AttackEvent[]>([]);
  const [defenseLog, setDefenseLog] = useState<LogEntry[]>([]);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [sophHistory, setSophHistory] = useState<number[]>([]);
  const [latHistory, setLatHistory] = useState<number[]>([]);
  const [campaignWsAlert, setCampaignWsAlert] = useState<CampaignWsAlert | null>(null);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    let ws: WebSocket | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout>;
    let retries = 0;
    const historyBuf: AttackEvent[] = [];
    let historyFlush: ReturnType<typeof setTimeout> | null = null;

    function connect() {
      try {
        // SECURITY: No token in URL — auth is sent after connection
        ws = new WebSocket(`${WS_URL}/ws/live`);
      } catch {
        return;
      }

      ws.onopen = () => {
        // Send auth message as first frame (auth-after-connect protocol)
        const apiKey = typeof window !== 'undefined'
          ? getApiKey()
          : '';
        if (apiKey) {
          ws!.send(JSON.stringify({ type: 'auth', token: apiKey }));
        }
        retries = 0;
        setConnected(true);
      };

      ws.onmessage = (e: MessageEvent) => {
        try {
          const msg: WSMessage = JSON.parse(e.data);
          if (msg.type === 'ping') return;
          const ev = msg.data;
          if (!ev) return;

          if (msg.type === 'history') {
            historyBuf.push(normalizeEvent(ev));
            if (historyFlush) clearTimeout(historyFlush);
            historyFlush = setTimeout(() => {
              const batch = historyBuf.splice(0);
              setAttacks((prev) => [...batch, ...prev].slice(0, 60));
            }, 150);
            return;
          }

          // Campaign alert pushed from correlator
          if (msg.type === 'campaign' && msg.data) {
            const d = msg.data as Record<string, unknown>;
            setCampaignWsAlert({
              pattern: String(d.threat_type || 'UNKNOWN').replace(/_/g, ' '),
              eventCount: Number(d.events_count || 0),
              sourceCount: Number(d.unique_users || 0),
              severity: String(d.severity || 'HIGH'),
            });
            setTimeout(() => setCampaignWsAlert(null), 6000);
            return;
          }

          // Live event (attack, sanitized, clean)
          const atk = normalizeEvent(ev);
          setAttacks((prev) => [atk, ...prev.slice(0, 59)]);
          setLastUpdated(new Date());
          setSophHistory((prev) => [...prev.slice(-19), atk.soph]);
          setLatHistory((prev) => [...prev.slice(-19), atk.latency]);

          // Defense log entry
          const logEntry: LogEntry = {
            id: uid(),
            ts: new Date().toLocaleTimeString('en-US', { hour12: false }),
            type: atk.blocked ? 'BLOCK' : ev.action === 'SANITIZED' ? 'SANITIZE' : 'CLEAN',
            msg: atk.blocked
              ? `${atk.type.replace(/_/g, ' ')} blocked · T${atk.tier} · ${Math.round(atk.score * 100)}% · ${atk.latency}ms`
              : ev.action === 'SANITIZED'
                ? `⚠ SANITIZED: ${atk.type.replace(/_/g, ' ')} · output auditor`
                : `✓ CLEAN: input passed all layers`,
            color: atk.blocked ? '#00e676' : ev.action === 'SANITIZED' ? '#ffab00' : '#00e5ff',
          };

          if (atk.muts > 0) {
            setDefenseLog((prev) => [
              {
                id: uid(),
                ts: logEntry.ts,
                type: 'MUTATE' as const,
                msg: `Mutation engine: +${atk.muts} variants pre-blocked`,
                color: '#5a0090',
              },
              logEntry,
              ...prev.slice(0, 38),
            ]);
          } else {
            setDefenseLog((prev) => [logEntry, ...prev.slice(0, 39)]);
          }
        } catch {
          /* ignore malformed messages */
        }
      };

      ws.onclose = (event) => {
        setConnected(false);
        // 4003 = auth failure — don't retry with same credentials
        if (event.code === 4003) {
          console.error('ARGUS WS: Authentication failed (4003)');
          return;
        }
        retries++;
        const delay = Math.min(1000 * Math.pow(2, retries), 15000);
        reconnectTimer = setTimeout(connect, delay);
      };

      ws.onerror = () => {
        ws?.close();
      };
    }

    connect();
    return () => {
      clearTimeout(reconnectTimer);
      if (historyFlush) clearTimeout(historyFlush);
      if (ws) {
        ws.onclose = null;
        ws.close();
      }
    };
  }, []);

  return { attacks, defenseLog, lastUpdated, sophHistory, latHistory, campaignWsAlert, connected };
}
