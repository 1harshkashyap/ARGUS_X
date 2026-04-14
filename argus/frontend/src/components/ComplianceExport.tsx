import { useState, useCallback } from 'react';
import { fonts } from '../theme';
import { API_URL } from '../utils/config';
import { getApiKey } from '../utils/apiKey';

export function ComplianceExport() {
  const [loading, setLoading] = useState(false);

  const handleExport = useCallback(async () => {
    setLoading(true);
    try {
      const apiKey = getApiKey();
      const headers: Record<string, string> = { 'Content-Type': 'application/json' };
      if (apiKey) headers['X-API-Key'] = apiKey;

      const res = await fetch(`${API_URL}/api/v1/compliance/export`, { headers });
      if (!res.ok) throw new Error(`Export failed: ${res.status}`);
      const report = await res.json();

      // Open a new window with printable report
      const win = window.open('', '_blank');
      if (!win) return;

      const summary = report.summary || {};
      const topThreats = report.top_threats || [];
      const timeline = report.timeline || [];
      const xaiSamples = report.xai_samples || [];
      const config = report.system_config || {};

      win.document.write(`<!DOCTYPE html>
<html><head><title>ARGUS-X Compliance Report</title>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow:wght@400;600;700&display=swap" rel="stylesheet">
<style>
  body { font-family: 'Barlow', sans-serif; background: #fff; color: #1a1a2e; padding: 40px; max-width: 900px; margin: 0 auto; }
  h1 { font-family: 'Share Tech Mono', monospace; color: #0a0a1a; border-bottom: 2px solid #00e5ff; padding-bottom: 8px; }
  h2 { font-family: 'Share Tech Mono', monospace; color: #333; margin-top: 32px; font-size: 16px; }
  table { width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 13px; }
  th, td { padding: 8px 12px; border: 1px solid #ddd; text-align: left; }
  th { background: #f0f4f8; font-weight: 600; }
  .stat-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin: 16px 0; }
  .stat-card { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; text-align: center; }
  .stat-val { font-size: 28px; font-weight: 700; font-family: 'Share Tech Mono', monospace; }
  .stat-label { font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: 0.1em; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }
  .badge-blocked { background: #fee2e2; color: #dc2626; }
  .badge-clean { background: #dcfce7; color: #16a34a; }
  .footer { margin-top: 40px; font-size: 11px; color: #999; text-align: center; border-top: 1px solid #eee; padding-top: 16px; }
  @media print { body { padding: 20px; } }
</style></head><body>
<h1>⚔ ARGUS-X — Compliance Report</h1>
<p style="color:#666;font-size:13px;">Generated: ${report.generated_at || new Date().toISOString()} · Version: ${report.version || '3.0.0'}</p>

<h2>Summary</h2>
<div class="stat-grid">
  <div class="stat-card"><div class="stat-val" style="color:#dc2626">${summary.blocked || 0}</div><div class="stat-label">Attacks Blocked</div></div>
  <div class="stat-card"><div class="stat-val" style="color:#f59e0b">${summary.sanitized || 0}</div><div class="stat-label">Sanitized</div></div>
  <div class="stat-card"><div class="stat-val" style="color:#16a34a">${summary.clean || 0}</div><div class="stat-label">Clean</div></div>
  <div class="stat-card"><div class="stat-val">${summary.total_events || 0}</div><div class="stat-label">Total Events</div></div>
  <div class="stat-card"><div class="stat-val" style="color:#dc2626">${summary.defense_rate || 0}%</div><div class="stat-label">Defense Rate</div></div>
  <div class="stat-card"><div class="stat-val" style="color:#16a34a">${summary.protection_rate || 0}%</div><div class="stat-label">Protection Rate</div></div>
</div>

<h2>Top Threat Types</h2>
<table><tr><th>Threat Type</th><th>Count</th></tr>
${topThreats.map((t: { type: string; count: number }) => `<tr><td>${t.type}</td><td>${t.count}</td></tr>`).join('')}
</table>

<h2>Defense Timeline (Last ${timeline.length} Events)</h2>
<table><tr><th>Time</th><th>Action</th><th>Threat</th><th>Score</th><th>Layer</th></tr>
${timeline.slice(0, 30).map((e: Record<string, unknown>) => `<tr>
  <td style="font-size:11px">${e.timestamp || ''}</td>
  <td><span class="badge ${e.action === 'BLOCKED' ? 'badge-blocked' : 'badge-clean'}">${e.action || ''}</span></td>
  <td>${e.threat_type || '—'}</td>
  <td>${e.score || 0}</td>
  <td style="font-size:11px">${e.layer || ''}</td>
</tr>`).join('')}
</table>

<h2>XAI Decision Samples</h2>
<table><tr><th>Verdict</th><th>Reason</th><th>Soph</th><th>Action</th></tr>
${xaiSamples.map((x: Record<string, unknown>) => `<tr>
  <td><span class="badge ${x.verdict === 'BLOCKED' ? 'badge-blocked' : 'badge-clean'}">${x.verdict || ''}</span></td>
  <td style="font-size:12px">${x.primary_reason || ''}</td>
  <td>${x.sophistication_score || 0}/10</td>
  <td style="font-size:11px">${x.recommended_action || ''}</td>
</tr>`).join('')}
</table>

<h2>System Configuration</h2>
<table><tr><th>Setting</th><th>Value</th></tr>
  <tr><td>Firewall Mode</td><td>${config.firewall_mode || 'N/A'}</td></tr>
  <tr><td>ML Threshold</td><td>${config.ml_threshold || 'N/A'}</td></tr>
  <tr><td>Dynamic Rules</td><td>${config.dynamic_rules_count || 0}</td></tr>
  <tr><td>Database Connected</td><td>${config.database_connected ? '✅ Yes' : '❌ No'}</td></tr>
</table>

<div class="footer">ARGUS-X Autonomous AI Defense System — The AI that defends AI<br/>This report is auto-generated. All data is based on real system activity.</div>
</body></html>`);

      win.document.close();
      // Auto-trigger print dialog
      setTimeout(() => win.print(), 500);
    } catch (err) {
      console.error('Compliance export failed:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  return (
    <div style={{ background: '#080d1c', border: '1px solid #1a2845', borderRadius: 8, padding: '10px 12px' }}>
      <div style={{ fontFamily: fonts.mono, fontSize: 8, color: '#3a5070', letterSpacing: '0.15em', marginBottom: 7 }}>
        COMPLIANCE EXPORT
      </div>
      <button
        onClick={handleExport}
        disabled={loading}
        style={{
          width: '100%',
          padding: '8px 12px',
          background: loading ? '#1a2845' : 'linear-gradient(135deg, #00e5ff22, #d500f922)',
          border: '1px solid #00e5ff30',
          borderRadius: 6,
          color: loading ? '#3a5070' : '#00e5ff',
          fontFamily: fonts.mono,
          fontSize: 10,
          fontWeight: 700,
          cursor: loading ? 'default' : 'pointer',
          letterSpacing: '0.1em',
          transition: 'all 0.2s ease',
        }}
      >
        {loading ? '⏳ GENERATING...' : '📄 EXPORT REPORT'}
      </button>
      <div style={{ fontFamily: fonts.mono, fontSize: 7, color: '#2a4060', marginTop: 4 }}>
        one-click PDF · real data · audit-ready
      </div>
    </div>
  );
}
