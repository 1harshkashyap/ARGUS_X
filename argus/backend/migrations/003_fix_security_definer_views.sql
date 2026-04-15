-- ═══════════════════════════════════════════════════════════════════════
-- ARGUS-X — Fix Security Definer Views
-- Run in: Supabase Dashboard → SQL Editor
-- Fixes: Supabase linter ERROR "Security Definer View" on
--        events_safe and bypass_history_safe
-- ═══════════════════════════════════════════════════════════════════════

-- The linter flags views created WITHOUT security_invoker = true,
-- because PostgreSQL defaults views to SECURITY DEFINER, which
-- bypasses the querying user's RLS policies.
--
-- Fix: DROP + recreate with security_invoker = true.
-- We cannot use CREATE OR REPLACE because the column list may differ
-- from the existing view definition.

-- ── FIX 1: events_safe ──────────────────────────────────────────────────
DROP VIEW IF EXISTS events_safe;
CREATE VIEW events_safe WITH (security_invoker = true) AS
SELECT
    id, created_at, user_id, session_id, org_id,
    action, threat_type, score, layer, latency_ms,
    method, sophistication_score, mutations_preblocked,
    session_threat_level,
    -- Redact preview: only show for CLEAN events, hash for threats
    CASE
        WHEN action = 'CLEAN' THEN LEFT(preview, 50)
        ELSE LEFT(md5(COALESCE(preview, '')), 16) || ' [REDACTED]'
    END AS preview
FROM events;

-- Grant anon access to the safe view
GRANT SELECT ON events_safe TO anon;

-- ── FIX 2: bypass_history_safe ──────────────────────────────────────────
DROP VIEW IF EXISTS bypass_history_safe;
CREATE VIEW bypass_history_safe WITH (security_invoker = true) AS
SELECT
  id, created_at, attack_type, tier, bypassed, auto_patched,
  cycle, score, latency_ms
FROM red_team_sessions
WHERE bypassed = true AND auto_patched = true;

-- ── FIX 3: Analytics views (also flagged if they exist) ─────────────────
CREATE OR REPLACE VIEW threat_summary
WITH (security_invoker = true) AS
SELECT
  DATE_TRUNC('hour', created_at) AS hour,
  action,
  threat_type,
  COUNT(*)                        AS count,
  AVG(score)                      AS avg_score,
  AVG(sophistication_score)       AS avg_sophistication,
  AVG(latency_ms)                 AS avg_latency
FROM events
WHERE created_at > NOW() - INTERVAL '24 hours'
GROUP BY 1, 2, 3
ORDER BY 1 DESC;

CREATE OR REPLACE VIEW attack_heatmap
WITH (security_invoker = true) AS
SELECT
  EXTRACT(HOUR FROM created_at)   AS hour_of_day,
  EXTRACT(DOW  FROM created_at)   AS day_of_week,
  COUNT(*)                         AS threat_count
FROM events
WHERE action = 'BLOCKED'
  AND created_at > NOW() - INTERVAL '7 days'
GROUP BY 1, 2;

-- ── VERIFICATION ────────────────────────────────────────────────────────
DO $$
BEGIN
  RAISE NOTICE '✅ Security Definer View fixes applied:';
  RAISE NOTICE '  • events_safe → security_invoker = true';
  RAISE NOTICE '  • bypass_history_safe → security_invoker = true';
  RAISE NOTICE '  • threat_summary → security_invoker = true';
  RAISE NOTICE '  • attack_heatmap → security_invoker = true';
END $$;
