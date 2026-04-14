import { memo, useMemo } from 'react';
import { fonts } from '../theme';
import { THREAT_COLORS } from '../constants';
import type { AttackEvent } from '../types';

// Sophistication signal names matching backend SOPHISTICATION_SIGNALS
const SIGNAL_NAMES = [
  'naive_keyword', 'roleplay_frame', 'hypothetical',
  'metaphor_wrap', 'multi_step_setup', 'l33t_obfuscation',
  'unicode_tricks', 'encoding_hints', 'context_overflow',
  'nested_injection', 'multi_turn_setup',
];

interface FingerprintCardProps {
  attack: AttackEvent;
}

function FingerprintCardInner({ attack }: FingerprintCardProps) {
  const color = THREAT_COLORS[attack.type] || '#ff1744';

  // Generate a deterministic fingerprint ID from attack data
  const fingerprintId = useMemo(() => {
    const taxonomy: Record<string, string> = {
      INSTRUCTION_OVERRIDE: 'A1',
      ROLE_OVERRIDE: 'A2',
      DATA_EXFIL: 'A3',
      INDIRECT_INJECTION: 'A5',
      OBFUSCATED: 'A6',
      CONTEXT_OVERFLOW: 'A7',
      SOCIAL_ENG: 'A8',
      MULTI_TURN: 'A9',
    };
    const prefix = taxonomy[attack.type] || 'AX';
    // Use attack id + type to generate a stable hex suffix
    const hash = (attack.id * 2654435761 >>> 0).toString(16).toUpperCase().padStart(4, '0');
    return `${prefix}-${hash}`;
  }, [attack.id, attack.type]);

  // Derive which signals likely triggered based on soph + type
  const triggeredSignals = useMemo(() => {
    const active = new Set<number>();
    const soph = attack.soph;
    // Lower signals trigger for lower soph, higher for higher
    for (let i = 0; i < SIGNAL_NAMES.length; i++) {
      const threshold = Math.ceil((i + 1) / 2);
      // Deterministic pseudo-random seeded from attack.id + signal index
      const seed = ((attack.id * 2654435761 + i * 340573) >>> 0) % 100;
      if (soph >= threshold && seed > 30) {
        active.add(i);
      }
    }
    // Ensure at least one is active for blocked attacks
    if (attack.blocked && active.size === 0) active.add(0);
    return active;
  }, [attack.id, attack.soph, attack.blocked]);

  return (
    <div
      style={{
        background: '#060b18',
        border: `1px solid ${color}20`,
        borderRadius: 6,
        padding: '8px 10px',
        animation: 'slideUp 0.3s cubic-bezier(.22,1,.36,1)',
      }}
    >
      {/* Fingerprint ID */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
        <div
          style={{
            fontFamily: fonts.mono,
            fontSize: 10,
            fontWeight: 700,
            color,
            letterSpacing: '0.05em',
          }}
        >
          🔬 {fingerprintId}
        </div>
        <div
          style={{
            fontFamily: fonts.mono,
            fontSize: 8,
            color: '#3a5070',
            marginLeft: 'auto',
          }}
        >
          SOPH {attack.soph}/10
        </div>
      </div>

      {/* Signal Heatmap Grid */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(11, 1fr)',
          gap: 2,
          marginBottom: 4,
        }}
      >
        {SIGNAL_NAMES.map((name, i) => {
          const active = triggeredSignals.has(i);
          const intensity = active ? Math.min(0.3 + (attack.soph / 10) * 0.7, 1) : 0.05;
          return (
            <div
              key={name}
              title={name.replace(/_/g, ' ')}
              style={{
                width: '100%',
                paddingTop: '100%',
                borderRadius: 2,
                background: active
                  ? `rgba(${attack.soph > 7 ? '255,23,68' : attack.soph > 4 ? '255,171,0' : '0,230,118'},${intensity})`
                  : 'rgba(255,255,255,0.03)',
                border: active ? `1px solid ${color}40` : '1px solid transparent',
                transition: 'background 0.3s ease',
              }}
            />
          );
        })}
      </div>

      {/* Signal labels */}
      <div style={{ fontFamily: fonts.mono, fontSize: 7, color: '#2a4060', lineHeight: 1.4 }}>
        {Array.from(triggeredSignals)
          .slice(0, 3)
          .map((i) => SIGNAL_NAMES[i]?.replace(/_/g, ' '))
          .join(' · ')}
        {triggeredSignals.size > 3 && ` +${triggeredSignals.size - 3} more`}
      </div>
    </div>
  );
}

export const FingerprintCard = memo(FingerprintCardInner);
