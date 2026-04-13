// ═══════════════════════════════════════════════════════════════════════
// ARGUS-X — DEFENSE COMMAND CENTER
// Layer 9: The living, breathing unified system view
// Every pixel is intentional. Every animation has meaning.
//
// This is the layout orchestrator only — all logic lives in hooks,
// all sub-panels are modular components.
// ═══════════════════════════════════════════════════════════════════════

import { fonts } from '../theme';
import { THREAT_COLORS } from '../constants';
import { useRealtimeFeed } from '../hooks/useRealtimeFeed';
import { useStats } from '../hooks/useStats';
import { NeuralCanvas } from './NeuralCanvas';
import { XAICard } from './XAICard';
import { FeedItem } from './FeedItem';
import { Sparkline } from './Sparkline';
import { ThreatLevelBar } from './ThreatLevelBar';
import { BattleStatus } from './BattleStatus';
import { MiniClusterMap } from './MiniClusterMap';
import { CampaignAlert } from './CampaignAlert';
import { DefenseLog } from './DefenseLog';

export default function CommandCenter() {
  // ── Data hooks ─────────────────────────────────────────────────────
  const { attacks, defenseLog, lastUpdated, sophHistory, latHistory } = useRealtimeFeed();
  const {
    stats, tier, threatLevel, campaignCount,
    showAlert, alertMsg, alertHistory,
    showAlertHistory, setShowAlertHistory,
  } = useStats();

  // ── Derived values ─────────────────────────────────────────────────
  const blockRate = stats.total > 0 ? Math.round((stats.blocked / stats.total) * 100) : 100;
  const sophAvg = sophHistory.length
    ? +(sophHistory.reduce((a, b) => a + b, 0) / sophHistory.length).toFixed(1)
    : 0;
  const sophTrend = sophHistory.length >= 6
    ? +(
        sophHistory.slice(-3).reduce((a, b) => a + b, 0) / 3 -
        sophHistory.slice(-6, -3).reduce((a, b) => a + b, 0) / 3
      ).toFixed(1)
    : 0;

  return (
    <div
      style={{
        background: '#030508',
        color: '#ddeeff',
        fontFamily: fonts.body,
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
        position: 'relative',
      }}
    >
      {/* ── CAMPAIGN ALERTS ── */}
      <CampaignAlert
        showAlert={showAlert}
        alertMsg={alertMsg}
        alertHistory={alertHistory}
        showAlertHistory={showAlertHistory}
        onToggleHistory={() => setShowAlertHistory((v: boolean) => !v)}
      />

      {/* ── HEADER ── */}
      <div
        style={{
          height: 50, flexShrink: 0,
          background: 'rgba(5,9,20,0.98)',
          borderBottom: '1px solid #1a2845',
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          padding: '0 20px',
        }}
      >
        {/* Logo */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{
            width: 32, height: 32, borderRadius: '50%',
            background: 'linear-gradient(135deg, #00e5ff 0%, #d500f9 100%)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 14, boxShadow: '0 0 14px rgba(0,200,255,0.25)',
          }}>⚔</div>
          <div>
            <div style={{ fontFamily: fonts.display, fontSize: 13, fontWeight: 700, color: '#00e5ff', letterSpacing: '0.12em' }}>
              ARGUS<span style={{ color: '#d500f9' }}>-X</span>
            </div>
            <div style={{ fontFamily: fonts.mono, fontSize: 7, color: '#2a4060', letterSpacing: '0.2em' }}>
              DEFENSE COMMAND CENTER
            </div>
          </div>
        </div>

        {/* Center stats */}
        <div style={{ display: 'flex', gap: 20 }}>
          {[
            { label: 'BLOCKED', val: stats.blocked, color: '#ff1744' },
            { label: 'PRE-BLOCKED', val: stats.muts, color: '#d500f9' },
            { label: 'BYPASSES', val: stats.bypasses, color: '#ffab00' },
            { label: 'DEFENSE RATE', val: `${blockRate}%`, color: '#00e676' },
            { label: 'AVG SOPH', val: `${sophAvg}/10`, color: sophAvg > 6 ? '#ff1744' : sophAvg > 4 ? '#ffab00' : '#00e676' },
            { label: 'CAMPAIGNS', val: campaignCount, color: '#ff6d00' },
          ].map((s) => (
            <div key={s.label} style={{ textAlign: 'center' }}>
              <div style={{ fontFamily: fonts.display, fontSize: 16, fontWeight: 700, color: s.color, lineHeight: 1 }}>{s.val}</div>
              <div style={{ fontFamily: fonts.mono, fontSize: 7, color: '#2a4060', letterSpacing: '0.15em' }}>{s.label}</div>
            </div>
          ))}
        </div>

        {/* Status pills */}
        <div style={{ display: 'flex', gap: 8 }}>
          {[
            { color: '#00e676', label: 'ARGUS ONLINE' },
            { color: '#ff1744', label: 'RED AGENT LIVE', fast: true },
          ].map((p) => (
            <div
              key={p.label}
              style={{
                display: 'flex', alignItems: 'center', gap: 5, padding: '3px 10px',
                borderRadius: 12, background: p.color + '12', border: `1px solid ${p.color}30`,
                fontFamily: fonts.mono, fontSize: 8, color: p.color,
              }}
            >
              <div
                style={{
                  width: 6, height: 6, borderRadius: '50%', background: p.color,
                  animation: `pulse ${p.fast ? '0.7s' : '1.4s'} ease-in-out infinite`,
                }}
              />
              {p.label}
            </div>
          ))}
        </div>
      </div>

      {/* ── MAIN GRID ── */}
      <div
        style={{
          flex: 1, display: 'grid',
          gridTemplateColumns: '240px 1fr 220px',
          gridTemplateRows: '1fr 180px',
          gap: '1px', background: '#0d1628',
          overflow: 'hidden', minHeight: 0,
        }}
      >
        {/* ── LEFT: Live Feed ── */}
        <div
          style={{
            background: '#030508', display: 'flex', flexDirection: 'column',
            overflow: 'hidden', gridRow: '1 / 3',
          }}
        >
          <div style={{ padding: '8px 12px 7px', borderBottom: '1px solid #1a2845', flexShrink: 0 }}>
            <div
              style={{
                fontFamily: fonts.mono, fontSize: 8, letterSpacing: '0.2em', color: '#3a5070',
                display: 'flex', alignItems: 'center', gap: 6,
              }}
            >
              <div style={{ width: 4, height: 4, borderRadius: '50%', background: '#00e5ff', animation: 'pulse 1.4s ease-in-out infinite' }} />
              LIVE THREAT FEED
              {lastUpdated && (
                <span style={{ marginLeft: 'auto', fontSize: 7, color: '#2a4060' }}>
                  last: {lastUpdated.toLocaleTimeString('en-US', { hour12: false })}
                </span>
              )}
            </div>
          </div>
          <div
            style={{
              flex: 1, overflowY: 'auto', padding: '6px 8px',
              display: 'flex', flexDirection: 'column', gap: 5,
            }}
          >
            {attacks.length === 0 ? (
              <div style={{ fontFamily: fonts.mono, fontSize: 9, color: '#2a4060', textAlign: 'center', padding: '20px 0' }}>
                Waiting for live feed…
              </div>
            ) : (
              attacks.slice(0, 25).map((a) => <FeedItem key={a.id} attack={a} />)
            )}
          </div>
        </div>

        {/* ── CENTER TOP: Neural Map + XAI ── */}
        <div style={{ background: '#030508', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1px', overflow: 'hidden' }}>
          {/* Neural Threat Map */}
          <div style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
            <div style={{ padding: '8px 12px 7px', borderBottom: '1px solid #1a2845', flexShrink: 0 }}>
              <div style={{ fontFamily: fonts.mono, fontSize: 8, letterSpacing: '0.2em', color: '#3a5070', display: 'flex', alignItems: 'center', gap: 6 }}>
                <div style={{ width: 4, height: 4, borderRadius: '50%', background: '#00e5ff' }} />
                NEURAL THREAT MAP · LIVE
              </div>
            </div>
            <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 8, overflow: 'hidden' }}>
              <NeuralCanvas attacks={attacks} />
            </div>
          </div>

          {/* XAI Decision Stream */}
          <div style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
            <div style={{ padding: '8px 12px 7px', borderBottom: '1px solid #1a2845', flexShrink: 0 }}>
              <div style={{ fontFamily: fonts.mono, fontSize: 8, letterSpacing: '0.2em', color: '#3a5070', display: 'flex', alignItems: 'center', gap: 6 }}>
                <div style={{ width: 4, height: 4, borderRadius: '50%', background: '#d500f9' }} />
                EXPLAINABLE AI · DECISION STREAM
              </div>
            </div>
            <div style={{ flex: 1, overflowY: 'auto', padding: '6px 8px', display: 'flex', flexDirection: 'column', gap: 5 }}>
              {attacks.slice(0, 8).filter((a) => a.blocked).map((a) => (
                <XAICard key={a.id} attack={a} />
              ))}
            </div>
          </div>
        </div>

        {/* ── RIGHT: Analytics ── */}
        <div style={{ background: '#030508', display: 'flex', flexDirection: 'column', overflow: 'hidden', gridRow: '1 / 3' }}>
          <div style={{ padding: '8px 12px 7px', borderBottom: '1px solid #1a2845', flexShrink: 0 }}>
            <div style={{ fontFamily: fonts.mono, fontSize: 8, letterSpacing: '0.2em', color: '#3a5070', display: 'flex', alignItems: 'center', gap: 6 }}>
              <div style={{ width: 4, height: 4, borderRadius: '50%', background: '#d500f9' }} />
              ANALYTICS
            </div>
          </div>
          <div style={{ flex: 1, overflowY: 'auto', padding: '8px', display: 'flex', flexDirection: 'column', gap: 8 }}>
            {/* Threat level */}
            <div style={{ background: '#080d1c', border: '1px solid #1a2845', borderRadius: 8, padding: '10px 12px' }}>
              <ThreatLevelBar level={Math.round(threatLevel)} />
            </div>

            {/* Soph trend */}
            <div style={{ background: '#080d1c', border: '1px solid #1a2845', borderRadius: 8, padding: '10px 12px' }}>
              <div style={{ fontFamily: fonts.mono, fontSize: 8, color: '#3a5070', letterSpacing: '0.15em', marginBottom: 7 }}>SOPHISTICATION TREND</div>
              <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, marginBottom: 5 }}>
                <span style={{ fontFamily: fonts.display, fontSize: 20, fontWeight: 700, color: sophAvg > 6 ? '#ff1744' : sophAvg > 4 ? '#ffab00' : '#00e676' }}>{sophAvg}</span>
                <span style={{ fontFamily: fonts.mono, fontSize: 9, color: sophTrend > 0 ? '#ff1744' : '#00e676' }}>
                  {sophTrend > 0 ? `↑ +${sophTrend}` : `↓ ${sophTrend}`}
                </span>
              </div>
              <Sparkline data={sophHistory} color={sophAvg > 6 ? '#ff1744' : sophAvg > 4 ? '#ffab00' : '#00e676'} height={40} />
              <div style={{ fontFamily: fonts.mono, fontSize: 8, color: '#3a5070', marginTop: 4 }}>
                {sophTrend > 0.5 ? '⚠ ESCALATING — tighten thresholds' : sophTrend < -0.5 ? '✓ DECLINING — defense working' : '— STABLE'}
              </div>
            </div>

            {/* Latency */}
            <div style={{ background: '#080d1c', border: '1px solid #1a2845', borderRadius: 8, padding: '10px 12px' }}>
              <div style={{ fontFamily: fonts.mono, fontSize: 8, color: '#3a5070', letterSpacing: '0.15em', marginBottom: 7 }}>RESPONSE LATENCY (MS)</div>
              <Sparkline data={latHistory} color="#00e5ff" height={40} />
              <div style={{ fontFamily: fonts.mono, fontSize: 8, color: '#3a5070', marginTop: 4 }}>
                avg {latHistory.length ? Math.round(latHistory.reduce((a, b) => a + b, 0) / latHistory.length) : 0}ms · p99 {latHistory.length ? Math.round(Math.max(...latHistory)) : 0}ms
              </div>
            </div>

            {/* AI Battle */}
            <div style={{ background: '#080d1c', border: '1px solid #1a2845', borderRadius: 8, padding: '10px 12px' }}>
              <div style={{ fontFamily: fonts.mono, fontSize: 8, color: '#3a5070', letterSpacing: '0.15em', marginBottom: 7 }}>AI vs AI BATTLE</div>
              <BattleStatus
                redAtks={stats.total}
                blueBlocks={stats.blocked}
                redBypasses={stats.bypasses}
                tier={tier}
              />
            </div>

            {/* Cluster map */}
            <div style={{ background: '#080d1c', border: '1px solid #1a2845', borderRadius: 8, padding: '10px 12px' }}>
              <div style={{ fontFamily: fonts.mono, fontSize: 8, color: '#3a5070', letterSpacing: '0.15em', marginBottom: 7 }}>THREAT CLUSTERS</div>
              <MiniClusterMap attacks={attacks} />
              <div style={{ fontFamily: fonts.mono, fontSize: 8, color: '#3a5070', marginTop: 4 }}>
                {Object.keys(THREAT_COLORS).length} cluster families active
              </div>
            </div>

            {/* Mutation counter */}
            <div style={{ background: '#080d1c', border: '1px solid #1a2845', borderRadius: 8, padding: '10px 12px' }}>
              <div style={{ fontFamily: fonts.mono, fontSize: 8, color: '#3a5070', letterSpacing: '0.15em', marginBottom: 5 }}>MUTATION ENGINE</div>
              <div style={{ fontFamily: fonts.display, fontSize: 24, fontWeight: 700, color: '#d500f9', lineHeight: 1 }}>{stats.muts}</div>
              <div style={{ fontFamily: fonts.mono, fontSize: 8, color: '#3a5070', marginTop: 3 }}>variants pre-blocked · zero human input</div>
            </div>
          </div>
        </div>

        {/* ── BOTTOM: Defense Log ── */}
        <DefenseLog entries={defenseLog} />
      </div>
    </div>
  );
}
