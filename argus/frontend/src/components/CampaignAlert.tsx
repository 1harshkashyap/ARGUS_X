import { memo } from 'react';
import { fonts } from '../theme';
import type { AlertEntry } from '../types';

interface CampaignAlertProps {
  showAlert: boolean;
  alertMsg: string;
  alertHistory: AlertEntry[];
  showAlertHistory: boolean;
  onToggleHistory: () => void;
}

function CampaignAlertInner({
  showAlert,
  alertMsg,
  alertHistory,
  showAlertHistory,
  onToggleHistory,
}: CampaignAlertProps) {
  if (showAlert) {
    return (
      <div
        style={{
          position: 'absolute', top: 60, right: 16, zIndex: 100,
          background: 'rgba(6,10,20,0.97)', border: '1px solid #ff1744',
          borderRadius: 8, padding: '12px 16px', maxWidth: 300,
          boxShadow: '0 0 24px rgba(255,23,68,0.3)',
          animation: 'slideInAlert 0.4s cubic-bezier(.22,1,.36,1)',
        }}
      >
        <div
          style={{
            fontFamily: fonts.mono, fontSize: 9, color: '#ff1744',
            letterSpacing: '0.2em', marginBottom: 5,
          }}
        >
          ⚠ CAMPAIGN DETECTED
        </div>
        <div
          style={{
            fontFamily: fonts.mono, fontSize: 10, color: '#c0d0e0', lineHeight: 1.6,
          }}
        >
          {alertMsg}
        </div>
        <div style={{ marginTop: 6, fontFamily: fonts.mono, fontSize: 8, color: '#5a7090' }}>
          Correlator flagged · Auto-escalated
        </div>
      </div>
    );
  }

  if (alertHistory.length === 0) return null;

  return (
    <div style={{ position: 'absolute', top: 60, right: 16, zIndex: 90 }}>
      <div
        onClick={onToggleHistory}
        style={{
          fontFamily: fonts.mono, fontSize: 8, color: '#ff6d00',
          cursor: 'pointer', letterSpacing: '0.1em', padding: '4px 10px',
          background: 'rgba(6,10,20,0.9)', border: '1px solid #331500', borderRadius: 6,
        }}
      >
        ⚠ {alertHistory.length} CAMPAIGN{alertHistory.length > 1 ? 'S' : ''} ·{' '}
        {showAlertHistory ? 'HIDE' : 'SHOW'}
      </div>
      {showAlertHistory && (
        <div
          style={{
            marginTop: 4, background: 'rgba(6,10,20,0.97)', border: '1px solid #331500',
            borderRadius: 6, padding: '8px 10px', maxWidth: 300, maxHeight: 200, overflowY: 'auto',
          }}
        >
          {alertHistory.map((a, i) => (
            <div
              key={i}
              style={{
                fontFamily: fonts.mono, fontSize: 9, color: '#7a8a9a', lineHeight: 1.7,
                borderBottom: i < alertHistory.length - 1 ? '1px solid #1a2030' : 'none',
                padding: '3px 0',
              }}
            >
              <span style={{ color: '#5a4030' }}>[{a.ts}]</span> {a.msg}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export const CampaignAlert = memo(CampaignAlertInner);
