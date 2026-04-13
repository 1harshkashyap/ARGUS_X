import { memo, useMemo } from 'react';
import { fonts } from '../theme';
import { THREAT_COLORS } from '../constants';
import { SophMeter } from './SophMeter';
import { randFloat } from '../utils/helpers';
import type { AttackEvent } from '../types';

interface XAICardProps {
  attack: AttackEvent;
}

function XAICardInner({ attack }: XAICardProps) {
  const color = THREAT_COLORS[attack.type] || '#ff1744';

  // Stabilized random values — computed once per attack.id
  const layerScores = useMemo(
    () => [
      { name: 'Regex Engine', v: attack.blocked ? randFloat(0.6, 0.99) : randFloat(0.1, 0.4) },
      { name: 'ML Classifier', v: attack.blocked ? randFloat(0.75, 0.98) : randFloat(0.4, 0.72) },
      { name: 'Output Auditor', v: attack.blocked ? randFloat(0.5, 0.95) : randFloat(0.2, 0.5) },
    ],
    [attack.id],
  );

  return (
    <div
      style={{
        background: '#080d1c',
        border: `1px solid ${color}30`,
        borderLeft: `3px solid ${color}`,
        borderRadius: 8,
        padding: '11px 13px',
        animation: 'slideUp 0.3s cubic-bezier(.22,1,.36,1)',
        flexShrink: 0,
      }}
    >
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-start',
          marginBottom: 8,
        }}
      >
        <div style={{ flex: 1, marginRight: 10 }}>
          <div style={{ fontFamily: fonts.mono, fontSize: 10, color: '#5a7090', marginBottom: 3 }}>
            {attack.type.replace(/_/g, ' ')} · T{attack.tier} · {attack.latency}ms
          </div>
          <div style={{ fontFamily: fonts.mono, fontSize: 10, color: '#7a9ac0', lineHeight: 1.5 }}>
            {attack.text.slice(0, 70)}
            {attack.text.length > 70 ? '…' : ''}
          </div>
        </div>
        <div style={{ textAlign: 'right', flexShrink: 0 }}>
          <div
            style={{
              fontFamily: fonts.mono,
              fontSize: 11,
              fontWeight: 700,
              color: attack.blocked ? '#ff1744' : '#ffab00',
              padding: '2px 8px',
              background: attack.blocked ? 'rgba(255,23,68,0.1)' : 'rgba(255,171,0,0.1)',
              borderRadius: 4,
              border: `1px solid ${attack.blocked ? '#ff174440' : '#ffab0040'}`,
            }}
          >
            {attack.blocked ? '⛔ BLOCKED' : '⚠ PARTIAL'}
          </div>
          <div style={{ marginTop: 5 }}>
            <SophMeter score={attack.soph} />
          </div>
        </div>
      </div>

      {/* Layer breakdown */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 3, marginBottom: 7 }}>
        {layerScores.map((l, i) => (
          <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <div
              style={{
                fontFamily: fonts.mono,
                fontSize: 8,
                color: '#3a5070',
                width: 90,
                flexShrink: 0,
              }}
            >
              {l.name}
            </div>
            <div
              style={{
                flex: 1,
                height: 3,
                background: '#0d1a30',
                borderRadius: 2,
                overflow: 'hidden',
              }}
            >
              <div
                style={{
                  height: '100%',
                  borderRadius: 2,
                  width: `${Math.round(l.v * 100)}%`,
                  background: l.v > 0.75 ? '#ff1744' : l.v > 0.5 ? '#ffab00' : '#00e676',
                  transition: 'width 0.6s ease',
                }}
              />
            </div>
            <div
              style={{
                fontFamily: fonts.mono,
                fontSize: 8,
                width: 30,
                textAlign: 'right',
                color: '#4a6080',
              }}
            >
              {Math.round(l.v * 100)}%
            </div>
          </div>
        ))}
      </div>

      {/* XAI Reasoning */}
      <div
        style={{
          background: 'rgba(100,0,200,0.06)',
          border: '1px solid rgba(180,0,255,0.12)',
          borderRadius: 5,
          padding: '6px 9px',
          fontFamily: fonts.mono,
          fontSize: 9,
          color: 'rgba(180,100,255,0.85)',
          lineHeight: 1.6,
          fontStyle: 'italic',
        }}
      >
        🧠 {attack.reason}
      </div>

      {attack.muts > 0 && (
        <div
          style={{
            marginTop: 5,
            fontFamily: fonts.mono,
            fontSize: 9,
            color: '#5a3090',
          }}
        >
          ⌬ {attack.muts} semantic variants auto-pre-blocked
        </div>
      )}
    </div>
  );
}

export const XAICard = memo(XAICardInner);
