// ═══════════════════════════════════════════════════════════════════════
// ARGUS-X — useRealtimeFeed Hook
// WebSocket connection to /ws/live with reconnection + event normalization
// ═══════════════════════════════════════════════════════════════════════

import { useState, useEffect } from 'react';
import type { AttackEvent, LogEntry, WSMessage } from '../types';
import { WS_URL } from '../utils/config';
import { normalizeEvent } from '../utils/sanitize';
import { uid } from '../utils/helpers';

interface RealtimeFeedState {
  attacks: AttackEvent[];
  defenseLog: LogEntry[];
  lastUpdated: Date | null;
  sophHistory: number[];
  latHistory: number[];
}

export function useRealtimeFeed(): RealtimeFeedState {
  const [attacks, setAttacks] = useState<AttackEvent[]>([]);
  const [defenseLog, setDefenseLog] = useState<LogEntry[]>([]);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [sophHistory, setSophHistory] = useState<number[]>([1, 2, 1, 3, 2, 4, 3, 4, 5, 4]);
  const [latHistory, setLatHistory] = useState<number[]>([28, 35, 22, 41, 30, 26, 38, 44, 29, 33]);

  useEffect(() => {
    let ws: WebSocket | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout>;
    let retries = 0;
    const historyBuf: AttackEvent[] = [];
    let historyFlush: ReturnType<typeof setTimeout> | null = null;

    function connect() {
      try {
        ws = new WebSocket(`${WS_URL}/ws/live`);
      } catch {
        return;
      }

      ws.onopen = () => {
        retries = 0;
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

      ws.onclose = () => {
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

  return { attacks, defenseLog, lastUpdated, sophHistory, latHistory };
}
