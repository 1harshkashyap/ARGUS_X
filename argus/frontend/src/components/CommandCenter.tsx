// ═══════════════════════════════════════════════════════════════════════
// ARGUS-X — DEFENSE COMMAND CENTER
// Layer 9: The living, breathing unified system view
// Every pixel is intentional. Every animation has meaning.
//
// State is sourced from Zustand store (useArgusStore).
// Hooks (useRealtimeFeed, useStatsPoller) are side-effect-only.
// ═══════════════════════════════════════════════════════════════════════

import { useMemo } from 'react';
import { THREAT_COLORS } from '../constants';
import { useRealtimeFeed } from '../hooks/useRealtimeFeed';
import { useStatsPoller } from '../hooks/useStats';
import { useArgusStore } from '../store/useArgusStore';
import { NeuralCanvas } from './NeuralCanvas';
import { XAICard } from './XAICard';
import { FeedItem } from './FeedItem';
import { Sparkline } from './Sparkline';
import { ThreatLevelBar } from './ThreatLevelBar';
import { BattleStatus } from './BattleStatus';
import { MiniClusterMap } from './MiniClusterMap';
import { CampaignAlert } from './CampaignAlert';
import { DefenseLog } from './DefenseLog';
import { ErrorBoundary } from './ErrorBoundary';
import { AttackTimeline } from './AttackTimeline';
import { PatchBanner } from './PatchBanner';
import { BypassHistory } from './BypassHistory';
import { FingerprintCard } from './FingerprintCard';
import { ComplianceExport } from './ComplianceExport';

export default function CommandCenter() {
  // ── Side-effect hooks (push data into Zustand) ──────────────────
  useRealtimeFeed();
  useStatsPoller();

  // ── Read from Zustand store ─────────────────────────────────────
  const attacks = useArgusStore((s) => s.attacks);
  const defenseLog = useArgusStore((s) => s.defenseLog);
  const lastUpdated = useArgusStore((s) => s.lastUpdated);
  const sophHistory = useArgusStore((s) => s.sophHistory);
  const latHistory = useArgusStore((s) => s.latHistory);
  const connected = useArgusStore((s) => s.connected);

  const total = useArgusStore((s) => s.total);
  const blocked = useArgusStore((s) => s.blocked);
  const muts = useArgusStore((s) => s.muts);
  const bypasses = useArgusStore((s) => s.bypasses);
  const tier = useArgusStore((s) => s.tier);
  const threatLevel = useArgusStore((s) => s.threatLevel);
  const campaignCount = useArgusStore((s) => s.campaignCount);
  const threatVelocity = useArgusStore((s) => s.threatVelocity);
  const redAgentRunning = useArgusStore((s) => s.redAgentRunning);
  const loading = useArgusStore((s) => s.loading);

  const campaignWsAlert = useArgusStore((s) => s.campaignWsAlert);
  const showAlert = useArgusStore((s) => s.showAlert);
  const alertMsg = useArgusStore((s) => s.alertMsg);
  const alertHistory = useArgusStore((s) => s.alertHistory);
  const showAlertHistory = useArgusStore((s) => s.showAlertHistory);
  const lastPatch = useArgusStore((s) => s.lastPatch);
  const showPatch = useArgusStore((s) => s.showPatch);
  const toggleAlertHistory = useArgusStore((s) => s.toggleAlertHistory);

  // ── Derived values ─────────────────────────────────────────────
  const blockRate = total > 0 ? Math.round((blocked / total) * 100) : 100;
  const sophAvg = sophHistory.length
    ? +(sophHistory.reduce((a, b) => a + b, 0) / sophHistory.length).toFixed(1)
    : 0;
  const sophTrend = sophHistory.length >= 6
    ? +(
        sophHistory.slice(-3).reduce((a, b) => a + b, 0) / 3 -
        sophHistory.slice(-6, -3).reduce((a, b) => a + b, 0) / 3
      ).toFixed(1)
    : 0;

  const canvasAttacks = useMemo(
    () => attacks.slice(0, 30),
    // Stable key: length changes on every new event (prepend)
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [attacks.length],
  );

  const effectiveShowAlert = campaignWsAlert ? true : showAlert;
  const effectiveAlertMsg = campaignWsAlert
    ? `CAMPAIGN DETECTED: ${campaignWsAlert.pattern} · ${campaignWsAlert.eventCount} events from ${campaignWsAlert.sourceCount} unique sources`
    : alertMsg;

  return (
    <div
      className="bg-argus-bg text-argus-text font-body h-full flex flex-col overflow-hidden relative"
      role="application"
      aria-label="ARGUS-X Defense Command Center"
    >
      {/* ── CAMPAIGN ALERTS ── */}
      <CampaignAlert
        showAlert={effectiveShowAlert}
        alertMsg={effectiveAlertMsg}
        alertHistory={alertHistory}
        showAlertHistory={showAlertHistory}
        onToggleHistory={toggleAlertHistory}
      />

      {/* ── PATCH BANNER ── */}
      <PatchBanner patch={lastPatch} visible={showPatch} />

      {/* ── HEADER ── */}
      <div className="h-[50px] shrink-0 bg-argus-glass border-b border-argus-border flex items-center justify-between px-5 gap-3 flex-wrap">
        {/* Logo */}
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-full bg-gradient-to-br from-argus-cyan to-argus-purple flex items-center justify-center text-sm shadow-[0_0_14px_rgba(0,200,255,0.25)]">
            ⚔
          </div>
          <div>
            <div className="font-display text-[13px] font-bold text-argus-cyan tracking-[0.12em]">
              ARGUS<span className="text-argus-purple">-X</span>
            </div>
            <div className="font-mono text-[7px] text-argus-dim tracking-[0.2em]">
              DEFENSE COMMAND CENTER
            </div>
          </div>
        </div>

        {/* Center stats */}
        <div className="flex gap-5 flex-wrap">
          {[
            { label: 'BLOCKED', val: blocked, color: '#ff1744' },
            { label: 'PRE-BLOCKED', val: muts, color: '#d500f9' },
            { label: 'BYPASSES', val: bypasses, color: '#ffab00' },
            { label: 'DEFENSE RATE', val: `${blockRate}%`, color: '#00e676' },
            { label: 'AVG SOPH', val: `${sophAvg}/10`, color: sophAvg > 6 ? '#ff1744' : sophAvg > 4 ? '#ffab00' : '#00e676' },
            { label: 'CAMPAIGNS', val: campaignCount, color: '#ff6d00' },
          ].map((s) => (
            <div key={s.label} className="text-center">
              <div className="font-display text-base font-bold leading-none" style={{ color: s.color }}>{s.val}</div>
              <div className="font-mono text-[7px] text-argus-dim tracking-[0.15em]">{s.label}</div>
            </div>
          ))}
        </div>

        {/* Status pills */}
        <div className="flex gap-2" role="status" aria-live="polite">
          {[
            connected
              ? { color: '#00e676', label: 'ARGUS ONLINE', fast: false }
              : { color: '#ffab00', label: 'RECONNECTING…', fast: true },
            redAgentRunning
              ? { color: '#ff1744', label: 'RED AGENT LIVE', fast: true }
              : { color: '#3a5070', label: 'RED AGENT IDLE', fast: false },
          ].map((p) => (
            <div
              key={p.label}
              aria-label={p.label}
              className="flex items-center gap-[5px] py-[3px] px-2.5 rounded-xl font-mono text-[8px]"
              style={{
                background: p.color + '12',
                border: `1px solid ${p.color}30`,
                color: p.color,
              }}
            >
              <div
                className="w-1.5 h-1.5 rounded-full"
                style={{
                  background: p.color,
                  animation: p.fast ? 'pulse 0.7s ease-in-out infinite' : 'none',
                  opacity: p.fast ? undefined : 0.6,
                }}
              />
              {p.label}
            </div>
          ))}
        </div>
      </div>

      {/* ── MAIN GRID ── */}
      <div className="argus-grid flex-1 grid gap-px bg-[#0d1628] overflow-hidden min-h-0">
        {/* ── LEFT: Live Feed ── */}
        <div className="bg-argus-bg flex flex-col overflow-hidden row-span-2">
          <div className="px-3 py-2 border-b border-argus-border shrink-0">
            <div className="font-mono text-[8px] tracking-[0.2em] text-argus-muted flex items-center gap-1.5">
              <div className="w-1 h-1 rounded-full bg-argus-cyan animate-pulse-dot" />
              LIVE THREAT FEED
              {lastUpdated && (
                <span className="ml-auto text-[7px] text-argus-dim">
                  last: {lastUpdated.toLocaleTimeString('en-US', { hour12: false })}
                </span>
              )}
            </div>
          </div>
          <div className="flex-1 overflow-y-auto px-2 py-1.5 flex flex-col gap-[5px]">
            {attacks.length === 0 ? (
              <div className="font-mono text-[9px] text-argus-dim text-center py-5">
                {loading ? (
                  <>
                    <div className="w-4 h-4 border-2 border-argus-border border-t-argus-cyan rounded-full animate-spin mx-auto mb-2" />
                    Connecting to live feed…
                  </>
                ) : (
                  'Waiting for events…'
                )}
              </div>
            ) : (
              attacks.slice(0, 25).map((a) => <FeedItem key={a.id} attack={a} />)
            )}
          </div>
        </div>

        {/* ── CENTER TOP: Neural Map + XAI ── */}
        <div className="bg-argus-bg grid grid-cols-2 gap-px overflow-hidden">
          {/* Neural Threat Map */}
          <div className="flex flex-col overflow-hidden">
            <div className="px-3 py-2 border-b border-argus-border shrink-0">
              <div className="font-mono text-[8px] tracking-[0.2em] text-argus-muted flex items-center gap-1.5">
                <div className="w-1 h-1 rounded-full bg-argus-cyan" />
                NEURAL THREAT MAP · LIVE
              </div>
            </div>
            <div className="flex-1 flex items-center justify-center p-2 overflow-hidden">
              <ErrorBoundary label="NEURAL THREAT MAP">
                <NeuralCanvas attacks={canvasAttacks} />
              </ErrorBoundary>
            </div>
          </div>

          {/* XAI Decision Stream */}
          <div className="flex flex-col overflow-hidden">
            <div className="px-3 py-2 border-b border-argus-border shrink-0">
              <div className="font-mono text-[8px] tracking-[0.2em] text-argus-muted flex items-center gap-1.5">
                <div className="w-1 h-1 rounded-full bg-argus-purple" />
                EXPLAINABLE AI · DECISION STREAM
              </div>
            </div>
            <div className="flex-1 overflow-y-auto px-2 py-1.5 flex flex-col gap-[5px]">
              {attacks.slice(0, 8).filter((a) => a.blocked).map((a) => (
                <div key={a.id} className="flex flex-col gap-1">
                  <XAICard attack={a} />
                  <FingerprintCard attack={a} />
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* ── RIGHT: Analytics ── */}
        <div className="bg-argus-bg flex flex-col overflow-hidden row-span-2">
          <div className="px-3 py-2 border-b border-argus-border shrink-0">
            <div className="font-mono text-[8px] tracking-[0.2em] text-argus-muted flex items-center gap-1.5">
              <div className="w-1 h-1 rounded-full bg-argus-purple" />
              ANALYTICS
            </div>
          </div>
          <div className="flex-1 overflow-y-auto p-2 flex flex-col gap-2">
            {/* Threat level */}
            <div className="bg-argus-panel border border-argus-border rounded-lg px-3 py-2.5">
              <ThreatLevelBar level={Math.round(threatLevel)} />
            </div>

            {/* Soph trend */}
            <div className="bg-argus-panel border border-argus-border rounded-lg px-3 py-2.5">
              <div className="font-mono text-[8px] text-argus-muted tracking-[0.15em] mb-[7px]">SOPHISTICATION TREND</div>
              <div className="flex items-baseline gap-2 mb-[5px]">
                <span className="font-display text-xl font-bold" style={{ color: sophAvg > 6 ? '#ff1744' : sophAvg > 4 ? '#ffab00' : '#00e676' }}>{sophAvg}</span>
                <span className="font-mono text-[9px]" style={{ color: sophTrend > 0 ? '#ff1744' : '#00e676' }}>
                  {sophTrend > 0 ? `↑ +${sophTrend}` : `↓ ${sophTrend}`}
                </span>
              </div>
              <Sparkline data={sophHistory} color={sophAvg > 6 ? '#ff1744' : sophAvg > 4 ? '#ffab00' : '#00e676'} height={40} />
              <div className="font-mono text-[8px] text-argus-muted mt-1">
                {sophTrend > 0.5 ? '⚠ ESCALATING — tighten thresholds' : sophTrend < -0.5 ? '✓ DECLINING — defense working' : '— STABLE'}
              </div>
            </div>

            {/* Latency */}
            <div className="bg-argus-panel border border-argus-border rounded-lg px-3 py-2.5">
              <div className="font-mono text-[8px] text-argus-muted tracking-[0.15em] mb-[7px]">RESPONSE LATENCY (MS)</div>
              <Sparkline data={latHistory} color="#00e5ff" height={40} />
              <div className="font-mono text-[8px] text-argus-muted mt-1">
                avg {latHistory.length ? Math.round(latHistory.reduce((a, b) => a + b, 0) / latHistory.length) : 0}ms · p99 {latHistory.length ? Math.round(Math.max(...latHistory)) : 0}ms
              </div>
            </div>

            {/* AI Battle */}
            <div className="bg-argus-panel border border-argus-border rounded-lg px-3 py-2.5">
              <div className="font-mono text-[8px] text-argus-muted tracking-[0.15em] mb-[7px]">AI vs AI BATTLE</div>
              <ErrorBoundary label="AI BATTLE">
                <BattleStatus redAtks={total} blueBlocks={blocked} redBypasses={bypasses} tier={tier} />
              </ErrorBoundary>
            </div>

            {/* Cluster map */}
            <div className="bg-argus-panel border border-argus-border rounded-lg px-3 py-2.5">
              <div className="font-mono text-[8px] text-argus-muted tracking-[0.15em] mb-[7px]">THREAT CLUSTERS</div>
              <ErrorBoundary label="THREAT CLUSTERS">
                <MiniClusterMap attacks={canvasAttacks} />
              </ErrorBoundary>
              <div className="font-mono text-[8px] text-argus-muted mt-1">
                {Object.keys(THREAT_COLORS).length} cluster families active
              </div>
            </div>

            {/* Mutation counter */}
            <div className="bg-argus-panel border border-argus-border rounded-lg px-3 py-2.5">
              <div className="font-mono text-[8px] text-argus-muted tracking-[0.15em] mb-[5px]">MUTATION ENGINE</div>
              <div className="font-display text-2xl font-bold text-argus-purple leading-none">{muts}</div>
              <div className="font-mono text-[8px] text-argus-muted mt-[3px]">variants pre-blocked · zero human input</div>
            </div>

            {/* Self-healing history */}
            <ErrorBoundary label="BYPASS HISTORY">
              <BypassHistory />
            </ErrorBoundary>

            {/* Compliance export */}
            <ErrorBoundary label="COMPLIANCE EXPORT">
              <ComplianceExport />
            </ErrorBoundary>
          </div>
        </div>

        {/* ── BOTTOM: Attack Velocity Timeline + Defense Log ── */}
        <div className="bg-argus-bg grid grid-cols-2 gap-px overflow-hidden">
          {/* Attack Velocity Timeline */}
          <div className="flex flex-col overflow-hidden">
            <div className="px-3 py-2 border-b border-argus-border shrink-0">
              <div className="font-mono text-[8px] tracking-[0.2em] text-argus-muted flex items-center gap-1.5">
                <div className="w-1 h-1 rounded-full bg-argus-red animate-pulse-fast" />
                ATTACK VELOCITY · 5MIN
              </div>
            </div>
            <div className="flex-1 p-2 overflow-hidden flex items-center">
              <ErrorBoundary label="ATTACK TIMELINE">
                <AttackTimeline velocity={threatVelocity} />
              </ErrorBoundary>
            </div>
          </div>
          {/* Defense Log */}
          <DefenseLog entries={defenseLog} />
        </div>
      </div>
    </div>
  );
}
