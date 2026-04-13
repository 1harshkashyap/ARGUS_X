import { memo } from 'react';
import { fonts } from '../theme';

interface ThreatLevelBarProps {
  level: number;
}

const LEVELS = ['MINIMAL', 'LOW', 'MODERATE', 'ELEVATED', 'HIGH', 'CRITICAL'];
const LEVEL_COLORS = ['#00e676', '#69f0ae', '#ffab00', '#ff6d00', '#ff1744', '#ff1744'];

function ThreatLevelBarInner({ level }: ThreatLevelBarProps) {
  const idx = Math.min(level, 5);
  const pct = ((level + 1) / 6) * 100;
  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 5 }}>
        <span
          style={{ fontFamily: fonts.mono, fontSize: 8, color: '#3a5070', letterSpacing: '0.15em' }}
        >
          THREAT LEVEL
        </span>
        <span style={{ fontFamily: fonts.mono, fontSize: 9, fontWeight: 700, color: LEVEL_COLORS[idx] }}>
          {LEVELS[idx]}
        </span>
      </div>
      <div style={{ height: 4, background: '#0d1830', borderRadius: 2, overflow: 'hidden' }}>
        <div
          style={{
            height: '100%',
            width: `${pct}%`,
            background: LEVEL_COLORS[idx],
            borderRadius: 2,
            transition: 'width 1s ease, background 1s ease',
            boxShadow: `0 0 8px ${LEVEL_COLORS[idx]}66`,
          }}
        />
      </div>
    </div>
  );
}

export const ThreatLevelBar = memo(ThreatLevelBarInner);
