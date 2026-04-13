import { memo } from 'react';
import { fonts } from '../theme';
import { Sparkline } from './Sparkline';

interface StatCardProps {
  label: string;
  value: string | number;
  color: string;
  sub?: string;
  sparkData?: number[];
}

function StatCardInner({ label, value, color, sub, sparkData }: StatCardProps) {
  return (
    <div
      style={{
        background: '#080d1c',
        border: '1px solid #1a2845',
        borderTop: `2px solid ${color}`,
        borderRadius: 8,
        padding: '11px 14px',
        display: 'flex',
        flexDirection: 'column',
        gap: 4,
      }}
    >
      <div
        style={{
          fontFamily: fonts.mono,
          fontSize: 8,
          letterSpacing: '0.18em',
          color: '#3a5070',
          textTransform: 'uppercase',
        }}
      >
        {label}
      </div>
      <div
        style={{
          fontFamily: fonts.display,
          fontSize: 24,
          fontWeight: 700,
          color,
          lineHeight: 1,
        }}
      >
        {value}
      </div>
      {sub && (
        <div style={{ fontFamily: fonts.mono, fontSize: 9, color: '#3a5070' }}>{sub}</div>
      )}
      {sparkData && (
        <div style={{ marginTop: 4 }}>
          <Sparkline data={sparkData} color={color} />
        </div>
      )}
    </div>
  );
}

export const StatCard = memo(StatCardInner);
