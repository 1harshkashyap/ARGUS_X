// ═══════════════════════════════════════════════════════════════════════
// ARGUS-X — Input Sanitization
// SECURITY: All attack payloads are untrusted input.
// WARNING: Do NOT use dangerouslySetInnerHTML anywhere in the frontend.
// ═══════════════════════════════════════════════════════════════════════

import type { AttackEvent, RawEvent } from '../types';
import { uid } from './helpers';

/**
 * Strip HTML angle brackets and null bytes from untrusted strings.
 * Defense-in-depth: React escapes JSX text, but this protects against
 * future dangerouslySetInnerHTML or innerHTML usage.
 */
export function sanitizeText(str: unknown): string {
  if (typeof str !== 'string') return '';
  return str.replace(/[<>]/g, '').replace(/\0/g, '');
}

/**
 * Normalize a raw backend event into the typed AttackEvent shape.
 * All string fields are sanitized at ingestion time.
 */
export function normalizeEvent(ev: RawEvent): AttackEvent {
  return {
    id: ev.id ? Number(ev.id) || uid() : uid(),
    type: sanitizeText(ev.threat_type || 'UNKNOWN'),
    tier: Math.ceil((ev.sophistication || 1) / 2),
    soph: ev.sophistication || 1,
    blocked: ev.action === 'BLOCKED',
    text: sanitizeText(ev.preview || ev.message || ''),
    score: ev.score || 0,
    latency: ev.latency_ms || 0,
    reason: sanitizeText(ev.explanation || ''),
    muts: ev.mutations_preblocked || 0,
    ts: ev.ts ? new Date(ev.ts) : new Date(),
    layer: sanitizeText(ev.layer || ev.method || 'INPUT_FIREWALL'),
  };
}
