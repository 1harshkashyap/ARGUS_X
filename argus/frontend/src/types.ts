// ═══════════════════════════════════════════════════════════════════════
// ARGUS-X — Core Type Definitions
// ═══════════════════════════════════════════════════════════════════════

/** Normalized attack event from WebSocket feed */
export interface AttackEvent {
  id: number;
  type: string;
  tier: number;
  soph: number;
  blocked: boolean;
  text: string;
  score: number;
  latency: number;
  reason: string;
  muts: number;
  ts: Date;
  layer: string;
}

/** Aggregated statistics from analytics polling */
export interface Stats {
  total: number;
  blocked: number;
  muts: number;
  bypasses: number;
  patched: number;
}

/** Defense log entry */
export interface LogEntry {
  id: number;
  ts: string;
  type: 'BLOCK' | 'SANITIZE' | 'CLEAN' | 'MUTATE';
  msg: string;
  color: string;
}

/** Campaign alert entry */
export interface AlertEntry {
  ts: string;
  msg: string;
}

/** Raw backend event shape (before normalization) */
export interface RawEvent {
  id?: string;
  action?: string;
  threat_type?: string;
  sophistication?: number;
  preview?: string;
  message?: string;
  score?: number;
  latency_ms?: number;
  explanation?: string;
  mutations_preblocked?: number;
  ts?: string;
  layer?: string;
  method?: string;
}

/** Raw WebSocket message wrapper */
export interface WSMessage {
  type: 'history' | 'attack' | 'sanitized' | 'clean' | 'ping';
  data?: RawEvent;
  ts?: string;
}
