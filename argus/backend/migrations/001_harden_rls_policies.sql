-- ═══════════════════════════════════════════════════════════════════════
-- ARGUS-X — RLS Hardening Migration
-- Run in: Supabase Dashboard → SQL Editor
-- Safe to re-run (all DROP statements use IF EXISTS).
-- ═══════════════════════════════════════════════════════════════════════

-- ── STEP 1: DROP OLD OVERLY-PERMISSIVE POLICIES ─────────────────────────
-- These policies allowed anon to read ALL tables including attack payloads.

DROP POLICY IF EXISTS "Public read events"       ON events;
DROP POLICY IF EXISTS "Public read stats"        ON stats;
DROP POLICY IF EXISTS "Public read xai"          ON xai_decisions;
DROP POLICY IF EXISTS "Public read battle"       ON battle_state;
DROP POLICY IF EXISTS "Public read evolution"    ON evolution_log;
DROP POLICY IF EXISTS "Public read clusters"     ON threat_clusters;
DROP POLICY IF EXISTS "Public read red_team"     ON red_team_sessions;
DROP POLICY IF EXISTS "Public read fingerprints" ON attack_fingerprints;
DROP POLICY IF EXISTS "Public read campaigns"    ON campaigns;
DROP POLICY IF EXISTS "Public read rules"        ON dynamic_rules;
DROP POLICY IF EXISTS "Public read knowledge"    ON knowledge_base;

DROP POLICY IF EXISTS "Service write events"       ON events;
DROP POLICY IF EXISTS "Service write stats"        ON stats;
DROP POLICY IF EXISTS "Service write xai"          ON xai_decisions;
DROP POLICY IF EXISTS "Service write battle"       ON battle_state;
DROP POLICY IF EXISTS "Service write evolution"    ON evolution_log;
DROP POLICY IF EXISTS "Service write clusters"     ON threat_clusters;
DROP POLICY IF EXISTS "Service write red_team"     ON red_team_sessions;
DROP POLICY IF EXISTS "Service write fingerprints" ON attack_fingerprints;
DROP POLICY IF EXISTS "Service write campaigns"    ON campaigns;
DROP POLICY IF EXISTS "Service write rules"        ON dynamic_rules;
DROP POLICY IF EXISTS "Service write knowledge"    ON knowledge_base;


-- ── STEP 2: ENSURE RLS IS ENABLED ───────────────────────────────────────

ALTER TABLE events              ENABLE ROW LEVEL SECURITY;
ALTER TABLE stats               ENABLE ROW LEVEL SECURITY;
ALTER TABLE xai_decisions       ENABLE ROW LEVEL SECURITY;
ALTER TABLE battle_state        ENABLE ROW LEVEL SECURITY;
ALTER TABLE evolution_log       ENABLE ROW LEVEL SECURITY;
ALTER TABLE threat_clusters     ENABLE ROW LEVEL SECURITY;
ALTER TABLE red_team_sessions   ENABLE ROW LEVEL SECURITY;
ALTER TABLE attack_fingerprints ENABLE ROW LEVEL SECURITY;
ALTER TABLE campaigns           ENABLE ROW LEVEL SECURITY;
ALTER TABLE dynamic_rules       ENABLE ROW LEVEL SECURITY;
ALTER TABLE knowledge_base      ENABLE ROW LEVEL SECURITY;


-- ── STEP 3: ANON READ — DASHBOARD-SAFE TABLES ONLY ─────────────────────

CREATE POLICY "Anon read stats"          ON stats               FOR SELECT TO anon USING (true);
CREATE POLICY "Anon read battle"         ON battle_state        FOR SELECT TO anon USING (true);
CREATE POLICY "Anon read evolution"      ON evolution_log       FOR SELECT TO anon USING (true);
CREATE POLICY "Anon read campaigns"      ON campaigns           FOR SELECT TO anon USING (true);
CREATE POLICY "Anon read fingerprints"   ON attack_fingerprints FOR SELECT TO anon USING (true);
CREATE POLICY "Anon read events"         ON events              FOR SELECT TO anon USING (true);
CREATE POLICY "Anon read xai"            ON xai_decisions       FOR SELECT TO anon USING (true);
CREATE POLICY "Anon read clusters"       ON threat_clusters     FOR SELECT TO anon USING (true);


-- ── STEP 4: SENSITIVE TABLES — SERVICE ROLE READ ONLY ───────────────────

CREATE POLICY "Service only red_team"    ON red_team_sessions FOR SELECT TO authenticated USING (auth.role() = 'service_role');
CREATE POLICY "Service only rules"       ON dynamic_rules     FOR SELECT TO authenticated USING (auth.role() = 'service_role');
CREATE POLICY "Service only knowledge"   ON knowledge_base    FOR SELECT TO authenticated USING (auth.role() = 'service_role');


-- ── STEP 5: SERVICE ROLE WRITES ─────────────────────────────────────────

CREATE POLICY "Service write events"       ON events              FOR INSERT TO authenticated WITH CHECK (auth.role() = 'service_role');
CREATE POLICY "Service write stats"        ON stats               FOR UPDATE TO authenticated USING (auth.role() = 'service_role');
CREATE POLICY "Service write xai"          ON xai_decisions       FOR INSERT TO authenticated WITH CHECK (auth.role() = 'service_role');
CREATE POLICY "Service write battle"       ON battle_state        FOR ALL    TO authenticated USING (auth.role() = 'service_role');
CREATE POLICY "Service write evolution"    ON evolution_log       FOR INSERT TO authenticated WITH CHECK (auth.role() = 'service_role');
CREATE POLICY "Service write clusters"     ON threat_clusters     FOR ALL    TO authenticated USING (auth.role() = 'service_role');
CREATE POLICY "Service write red_team"     ON red_team_sessions   FOR INSERT TO authenticated WITH CHECK (auth.role() = 'service_role');
CREATE POLICY "Service write fingerprints" ON attack_fingerprints FOR ALL    TO authenticated USING (auth.role() = 'service_role');
CREATE POLICY "Service write campaigns"    ON campaigns           FOR ALL    TO authenticated USING (auth.role() = 'service_role');
CREATE POLICY "Service write rules"        ON dynamic_rules       FOR ALL    TO authenticated USING (auth.role() = 'service_role');
CREATE POLICY "Service write knowledge"    ON knowledge_base      FOR ALL    TO authenticated USING (auth.role() = 'service_role');


-- ── STEP 6: SECURITY-BARRIER VIEWS ─────────────────────────────────────

CREATE OR REPLACE VIEW events_safe WITH (security_barrier = true) AS
SELECT
  id, created_at, user_id, session_id, action, threat_type,
  score, layer, latency_ms, method, sophistication_score,
  attack_fingerprint, mutations_preblocked, session_threat_level
FROM events;

CREATE OR REPLACE VIEW bypass_history_safe WITH (security_barrier = true) AS
SELECT
  id, created_at, attack_type, tier, bypassed, auto_patched,
  cycle, score, latency_ms
FROM red_team_sessions
WHERE bypassed = true AND auto_patched = true;


-- ── VERIFICATION ────────────────────────────────────────────────────────

DO $$
BEGIN
  RAISE NOTICE '✅ RLS hardening migration complete';
  RAISE NOTICE 'Sensitive tables (red_team_sessions, dynamic_rules, knowledge_base) locked to service_role';
  RAISE NOTICE 'All write operations restricted to service_role';
  RAISE NOTICE 'Security-barrier views created: events_safe, bypass_history_safe';
END $$;
