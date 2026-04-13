// ═══════════════════════════════════════════════════════════════════════
// ARGUS-X — Backend Configuration
// Priority: window.__ARGUS_CONFIG__ → auto-detect from location → localhost
// ═══════════════════════════════════════════════════════════════════════

interface ArgusConfig {
  apiUrl?: string;
  wsUrl?: string;
}

declare global {
  interface Window {
    __ARGUS_CONFIG__?: ArgusConfig;
  }
}

const _cfg: ArgusConfig =
  (typeof window !== 'undefined' && window.__ARGUS_CONFIG__) || {};

const _loc = typeof window !== 'undefined' ? window.location : ({} as Location);

const _backendHost =
  _loc.hostname === 'localhost'
    ? 'localhost:8000'
    : _loc.host || 'localhost:8000';

const _proto = _loc.protocol === 'https:' ? 'https' : 'http';
const _wsproto = _loc.protocol === 'https:' ? 'wss' : 'ws';

export const API_URL = _cfg.apiUrl || `${_proto}://${_backendHost}`;
export const WS_URL = _cfg.wsUrl || `${_wsproto}://${_backendHost}`;
