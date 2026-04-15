// ═══════════════════════════════════════════════════════════════════════
// ARGUS-X — useStatsPoller Hook
// TanStack Query-powered stats polling with auto-retry and caching
// Pushes all data into Zustand store
// ═══════════════════════════════════════════════════════════════════════

import { useRef } from 'react';
import { useQuery } from '@tanstack/react-query';
import { API_URL } from '../utils/config';
import { getApiKey } from '../utils/apiKey';
import { useArgusStore } from '../store/useArgusStore';
import type { PatchEvent } from '../components/PatchBanner';

// ── Fetch function (pure, no side effects) ──────────────────────────

async function fetchStats(signal: AbortSignal) {
  const apiKey = getApiKey();
  const headers: Record<string, string> = {};
  if (apiKey) headers['X-API-Key'] = apiKey;

  const res = await fetch(`${API_URL}/api/v1/analytics/stats`, {
    signal,
    headers,
  });

  if (!res.ok) throw new Error(`Stats fetch failed: ${res.status}`);
  return res.json();
}

// ── Hook ─────────────────────────────────────────────────────────────

/**
 * Polls the stats endpoint via TanStack Query and pushes data into Zustand.
 * Handles: auto-retry, backoff, deduplication, background refetch.
 * Call this hook once in the root component (CommandCenter).
 */
export function useStatsPoller(): void {
  const lastPatchTsRef = useRef<string>('');

  useQuery({
    queryKey: ['argus-stats'],
    queryFn: ({ signal }) => fetchStats(signal),
    refetchInterval: 15_000, // 15s — real-time events come via WS; stats are aggregate
    // TanStack Query handles retry + backoff via queryClient defaults

    select: (data) => {
      // Push into Zustand store (side effect in select is fine for this pattern)
      const { updateStats, triggerAlert, dismissAlert, triggerPatch, dismissPatch } =
        useArgusStore.getState();

      updateStats({
        total: data.total || data.stats?.total || 0,
        blocked: data.blocked || data.stats?.blocked || 0,
        muts: data.mutations_preblocked || data.stats?.mutations_preblocked || 0,
        bypasses: data.stats?.bypasses_found || data.bypasses_found || 0,
        patched: data.blue_agent?.auto_patches || 0,
        loading: false,
      });

      // Red agent status
      if (data.agent && typeof data.agent.running === 'boolean') {
        updateStats({ redAgentRunning: data.agent.running });
      } else {
        updateStats({ redAgentRunning: !!data.agent });
      }

      if (data.battle) {
        updateStats({ tier: data.battle.red_tier || 1 });
      }

      if (data.threat_velocity && typeof data.threat_velocity === 'object') {
        updateStats({ threatVelocity: data.threat_velocity });
      }

      if (data.evolution && typeof data.evolution.threat_level === 'number') {
        updateStats({ threatLevel: data.evolution.threat_level });
      } else if (data.total > 0) {
        const bypassRate = (data.sanitized || 0) / data.total;
        updateStats({ threatLevel: Math.min(5, Math.round(bypassRate * 10)) });
      }

      if (data.campaigns && data.campaigns.length > 0) {
        updateStats({ campaignCount: data.campaigns.length });
        const latest = data.campaigns[data.campaigns.length - 1];
        if (latest?.detected_at) {
          const age = Date.now() - new Date(latest.detected_at).getTime();
          if (age < 10000) {
            const msg = `CAMPAIGN DETECTED: ${(latest.pattern || 'UNKNOWN').replace(/_/g, ' ')} · ${latest.event_count || '?'} events from ${latest.source_count || '?'} unique sources`;
            triggerAlert(msg);
            setTimeout(() => dismissAlert(), 5000);
          }
        }
      }

      // Detect new auto-patch events from the red agent
      if (data.agent?.last_patch?.ts && data.agent.last_patch.ts !== lastPatchTsRef.current) {
        lastPatchTsRef.current = data.agent.last_patch.ts;
        triggerPatch(data.agent.last_patch as PatchEvent);
        setTimeout(() => dismissPatch(), 8000);
      }

      return data; // Return for TanStack Query cache
    },
  });
}
