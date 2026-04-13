import { memo } from 'react';
import { fonts } from '../theme';
import { THREAT_COLORS } from '../constants';
import { SophMeter } from './SophMeter';
import type { AttackEvent } from '../types';

interface FeedItemProps {
  attack: AttackEvent;
}

function FeedItemInner({ attack }: FeedItemProps) {
  const color = THREAT_COLORS[attack.type] || '#ff1744';
  const action = attack.blocked ? 'BLOCKED' : 'PARTIAL';
  const actionColor = attack.blocked ? '#ff1744' : '#ffab00';
  return (
    <div
      style={{
        padding: '7px 10px',
        borderRadius: 5,
        background: '#080c1a',
        border: '1px solid #1a2845',
        borderLeft: `2px solid ${color}`,
        animation: 'slideLeft 0.25s cubic-bezier(.22,1,.36,1)',
        flexShrink: 0,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 3 }}>
        <div
          style={{
            padding: '1px 6px',
            borderRadius: 3,
            fontFamily: fonts.mono,
            fontSize: 8,
            fontWeight: 700,
            background: `${actionColor}18`,
            color: actionColor,
            border: `1px solid ${actionColor}30`,
          }}
        >
          {action}
        </div>
        <div style={{ fontFamily: fonts.mono, fontSize: 8, color }}>
          {attack.type.replace(/_/g, ' ')}
        </div>
        <div
          style={{
            marginLeft: 'auto',
            fontFamily: fonts.mono,
            fontSize: 8,
            color: '#2a4060',
          }}
        >
          {attack.latency}ms
        </div>
      </div>
      <div
        style={{
          fontFamily: fonts.mono,
          fontSize: 9,
          color: '#4a6080',
          whiteSpace: 'nowrap',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
        }}
      >
        {attack.text.slice(0, 58)}
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 4 }}>
        <SophMeter score={attack.soph} />
        {attack.muts > 0 && (
          <span style={{ fontFamily: fonts.mono, fontSize: 8, color: '#4a1a80' }}>
            +{attack.muts} variants
          </span>
        )}
      </div>
    </div>
  );
}

export const FeedItem = memo(FeedItemInner);
