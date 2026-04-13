import { memo } from 'react';
import { fonts } from '../theme';

interface SophMeterProps {
  score: number;
}

function SophMeterInner({ score }: SophMeterProps) {
  const color = score <= 3 ? '#00e676' : score <= 6 ? '#ffab00' : '#ff1744';
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
      {Array.from({ length: 10 }, (_, i) => (
        <div
          key={i}
          style={{
            width: 7,
            height: 10,
            borderRadius: 2,
            background: i < score ? color : '#1a2845',
            transition: 'background 0.3s',
            boxShadow: i < score ? `0 0 6px ${color}66` : 'none',
          }}
        />
      ))}
      <span style={{ fontFamily: fonts.mono, fontSize: 10, color, marginLeft: 4 }}>
        {score}/10
      </span>
    </div>
  );
}

export const SophMeter = memo(SophMeterInner);
