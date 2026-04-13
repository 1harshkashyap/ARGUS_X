import { memo } from 'react';
import { fonts } from '../theme';
import type { LogEntry } from '../types';

interface DefenseLogProps {
  entries: LogEntry[];
}

function DefenseLogInner({ entries }: DefenseLogProps) {
  return (
    <div
      style={{
        background: '#030508',
        gridColumn: '2 / 3',
        borderTop: '1px solid #0d1628',
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
      }}
    >
      <div
        style={{
          padding: '5px 12px 4px',
          borderBottom: '1px solid #1a2845',
          flexShrink: 0,
        }}
      >
        <div
          style={{
            fontFamily: fonts.mono,
            fontSize: 8,
            letterSpacing: '0.2em',
            color: '#3a5070',
          }}
        >
          DEFENSE LOG · REAL-TIME
        </div>
      </div>
      <div style={{ flex: 1, overflowY: 'auto', padding: '4px 10px' }}>
        {entries.map((l) => (
          <div
            key={l.id}
            style={{
              fontFamily: fonts.mono,
              fontSize: 11,
              lineHeight: 2.0,
              display: 'flex',
              gap: 10,
              animation: 'slideUp 0.2s ease',
            }}
          >
            <span style={{ color: '#1a3050' }}>[{l.ts}]</span>
            <span style={{ color: l.color }}>[{l.type}]</span>
            <span style={{ color: '#4a6080' }}>{l.msg}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export const DefenseLog = memo(DefenseLogInner);
