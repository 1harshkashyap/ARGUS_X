import { memo } from 'react';
import { fonts } from '../theme';

interface BattleStatusProps {
  redAtks: number;
  blueBlocks: number;
  redBypasses: number;
  tier: number;
}

const TIERS = ['NAIVE', 'BASIC', 'OBFUSCATED', 'ADVERSARIAL', 'APEX'];

function BattleStatusInner({ redAtks, blueBlocks, redBypasses, tier }: BattleStatusProps) {
  const blockRate = redAtks > 0 ? Math.round((blueBlocks / redAtks) * 100) : 100;
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
          <div
            style={{
              width: 8, height: 8, borderRadius: '50%',
              background: '#ff1744', animation: 'pulse 0.8s ease-in-out infinite',
            }}
          />
          <span style={{ fontFamily: fonts.mono, fontSize: 9, color: '#ff1744' }}>
            RED AGENT · {TIERS[Math.min(tier - 1, 4)]}
          </span>
        </div>
        <span style={{ fontFamily: fonts.mono, fontSize: 9, color: '#3a5070' }}>
          {redAtks} attacks
        </span>
      </div>
      <div style={{ height: 3, background: '#0d1830', borderRadius: 2 }}>
        <div
          style={{
            height: '100%',
            width: `${Math.min((redBypasses / Math.max(redAtks, 1)) * 100 * 5, 100)}%`,
            background: '#ff1744', borderRadius: 2, transition: 'width 0.5s ease',
          }}
        />
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
          <div
            style={{
              width: 8, height: 8, borderRadius: '50%',
              background: '#00e5ff', animation: 'pulse 1.4s ease-in-out infinite',
            }}
          />
          <span style={{ fontFamily: fonts.mono, fontSize: 9, color: '#00e5ff' }}>
            ARGUS-X DEFENSE
          </span>
        </div>
        <span style={{ fontFamily: fonts.mono, fontSize: 9, color: '#00e676' }}>
          {blockRate}% block rate
        </span>
      </div>
      <div style={{ height: 3, background: '#0d1830', borderRadius: 2 }}>
        <div
          style={{
            height: '100%', width: `${blockRate}%`,
            background: '#00e5ff', borderRadius: 2, transition: 'width 0.5s ease',
            boxShadow: '0 0 6px #00e5ff44',
          }}
        />
      </div>
      <div style={{ fontFamily: fonts.mono, fontSize: 8, color: '#2a4060' }}>
        {redBypasses} bypasses found · {redBypasses} auto-patched
      </div>
    </div>
  );
}

export const BattleStatus = memo(BattleStatusInner);
