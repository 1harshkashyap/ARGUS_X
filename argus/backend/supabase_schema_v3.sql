-- ═══════════════════════════════════════════════════════════════════════
-- ARGUS-X — Complete Supabase Schema v3.0
-- Paste in: Supabase Dashboard → SQL Editor → Run All
-- ═══════════════════════════════════════════════════════════════════════

-- ── EVENTS TABLE ──────────────────────────────────────────────────────────
-- Every single request through ARGUS creates one row here.
-- Supabase Realtime broadcasts each INSERT to all dashboard clients.
CREATE TABLE IF NOT EXISTS events (
  id                   UUID             DEFAULT gen_random_uuid() PRIMARY KEY,
  created_at           TIMESTAMPTZ      DEFAULT NOW() NOT NULL,
  user_id              TEXT             DEFAULT 'anonymous',
  session_id           TEXT,
  org_id               TEXT             DEFAULT 'default',
  preview              TEXT,
  action               TEXT             NOT NULL CHECK (action IN ('BLOCKED','SANITIZED','CLEAN','DEMO_BYPASS')),
  threat_type          TEXT,
  score                DOUBLE PRECISION DEFAULT 0,
  layer                TEXT             CHECK (layer IN ('INPUT','OUTPUT','NONE')),
  latency_ms           DOUBLE PRECISION DEFAULT 0,
  method               TEXT             DEFAULT '',
  sophistication_score INTEGER          DEFAULT 0,
  attack_fingerprint   TEXT,
  mutations_preblocked INTEGER          DEFAULT 0,
  session_threat_level TEXT             DEFAULT 'LOW' CHECK (session_threat_level IN ('LOW','MEDIUM','HIGH','CRITICAL')),
  explanation          TEXT
);

CREATE INDEX IF NOT EXISTS idx_events_created_at    ON events (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_events_action        ON events (action);
CREATE INDEX IF NOT EXISTS idx_events_threat_type   ON events (threat_type) WHERE threat_type IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_events_fingerprint   ON events (attack_fingerprint) WHERE attack_fingerprint IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_events_session       ON events (session_id);
CREATE INDEX IF NOT EXISTS idx_events_sophistication ON events (sophistication_score DESC);
CREATE INDEX IF NOT EXISTS idx_events_org_id        ON events (org_id);

-- ── STATS TABLE ───────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS stats (
  id                   INTEGER          PRIMARY KEY DEFAULT 1,
  total                INTEGER          DEFAULT 0,
  blocked              INTEGER          DEFAULT 0,
  sanitized            INTEGER          DEFAULT 0,
  clean                INTEGER          DEFAULT 0,
  bypasses_found       INTEGER          DEFAULT 0,
  mutations_preblocked INTEGER          DEFAULT 0,
  campaigns_detected   INTEGER          DEFAULT 0,
  updated_at           TIMESTAMPTZ      DEFAULT NOW()
);
INSERT INTO stats (id) VALUES (1) ON CONFLICT (id) DO NOTHING;

CREATE OR REPLACE FUNCTION increment_stat(col_name TEXT)
RETURNS void LANGUAGE plpgsql SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
  EXECUTE format(
    'UPDATE stats SET %I = %I + 1, total = CASE WHEN %L = ANY(ARRAY[''blocked'',''sanitized'',''clean'']) THEN total + 1 ELSE total END, updated_at = NOW() WHERE id = 1',
    col_name, col_name, col_name
  );
END;
$$;

-- ── XAI DECISIONS TABLE (NEW in v3) ──────────────────────────────────────
-- Stores explainable AI reasoning for every blocked/sanitized event.
CREATE TABLE IF NOT EXISTS xai_decisions (
  id                   UUID             DEFAULT gen_random_uuid() PRIMARY KEY,
  created_at           TIMESTAMPTZ      DEFAULT NOW() NOT NULL,
  event_id             UUID,
  attack_preview       TEXT,
  verdict              TEXT             CHECK (verdict IN ('BLOCKED','SANITIZED','FLAGGED')),
  sophistication_score INTEGER          DEFAULT 0,
  primary_reason       TEXT,
  pattern_family       TEXT,
  evolution_note       TEXT,
  recommended_action   TEXT,
  layer_decisions      JSONB            DEFAULT '[]'::jsonb,
  confidence_breakdown JSONB            DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_xai_created ON xai_decisions (created_at DESC);

-- ── BATTLE STATE TABLE (NEW in v3) ───────────────────────────────────────
-- Single-row table updated every battle tick. Supabase Realtime broadcasts.
CREATE TABLE IF NOT EXISTS battle_state (
  id                   INTEGER          PRIMARY KEY DEFAULT 1,
  tick                 INTEGER          DEFAULT 0,
  red_attacks          INTEGER          DEFAULT 0,
  red_bypasses         INTEGER          DEFAULT 0,
  blue_blocks          INTEGER          DEFAULT 0,
  blue_auto_patches    INTEGER          DEFAULT 0,
  red_tier             INTEGER          DEFAULT 1,
  red_strategy         TEXT             DEFAULT 'NAIVE',
  battle_score_red     INTEGER          DEFAULT 0,
  battle_score_blue    INTEGER          DEFAULT 0,
  mutations_total      INTEGER          DEFAULT 0,
  status               TEXT             DEFAULT 'READY',
  last_update          TIMESTAMPTZ      DEFAULT NOW()
);
INSERT INTO battle_state (id) VALUES (1) ON CONFLICT (id) DO NOTHING;

-- ── EVOLUTION LOG TABLE (NEW in v3) ──────────────────────────────────────
-- Tracks sophistication trend history for the evolution chart.
CREATE TABLE IF NOT EXISTS evolution_log (
  id                   UUID             DEFAULT gen_random_uuid() PRIMARY KEY,
  created_at           TIMESTAMPTZ      DEFAULT NOW() NOT NULL,
  window_avg           DOUBLE PRECISION DEFAULT 0,
  trend                TEXT             DEFAULT 'STABLE',
  data_points          INTEGER          DEFAULT 0,
  escalation_count     INTEGER          DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_evolution_created ON evolution_log (created_at DESC);

-- ── THREAT CLUSTERS TABLE (NEW in v3) ────────────────────────────────────
-- Stores DBSCAN clustering results for the cluster map.
CREATE TABLE IF NOT EXISTS threat_clusters (
  id                   UUID             DEFAULT gen_random_uuid() PRIMARY KEY,
  created_at           TIMESTAMPTZ      DEFAULT NOW() NOT NULL,
  cluster_id           TEXT,
  cluster_size         INTEGER          DEFAULT 0,
  dominant_type        TEXT,
  avg_sophistication   DOUBLE PRECISION DEFAULT 0,
  sample_text          TEXT,
  types                JSONB            DEFAULT '[]'::jsonb
);

-- ── RED TEAM SESSIONS TABLE ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS red_team_sessions (
  id            UUID             DEFAULT gen_random_uuid() PRIMARY KEY,
  created_at    TIMESTAMPTZ      DEFAULT NOW(),
  attack_text   TEXT,
  attack_type   TEXT,
  tier          INTEGER          DEFAULT 1,
  bypassed      BOOLEAN          DEFAULT false,
  auto_patched  BOOLEAN          DEFAULT false,
  cycle         INTEGER          DEFAULT 0,
  score         DOUBLE PRECISION DEFAULT 0,
  latency_ms    DOUBLE PRECISION DEFAULT 0
);

-- ── ATTACK FINGERPRINTS TABLE ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS attack_fingerprints (
  id                UUID         DEFAULT gen_random_uuid() PRIMARY KEY,
  fingerprint_id    TEXT         UNIQUE NOT NULL,
  threat_type       TEXT,
  sophistication    INTEGER      DEFAULT 1,
  first_seen        TIMESTAMPTZ  DEFAULT NOW(),
  last_seen         TIMESTAMPTZ  DEFAULT NOW(),
  occurrence_count  INTEGER      DEFAULT 1,
  explanation       TEXT,
  auto_blocked      BOOLEAN      DEFAULT false
);

CREATE OR REPLACE FUNCTION upsert_fingerprint(
  p_fingerprint_id TEXT, p_threat_type TEXT, p_sophistication INTEGER, p_explanation TEXT
) RETURNS void LANGUAGE plpgsql
SET search_path = public
AS $$
BEGIN
  INSERT INTO attack_fingerprints (fingerprint_id, threat_type, sophistication, explanation)
  VALUES (p_fingerprint_id, p_threat_type, p_sophistication, p_explanation)
  ON CONFLICT (fingerprint_id)
  DO UPDATE SET occurrence_count = attack_fingerprints.occurrence_count + 1,
                last_seen = NOW();
END;
$$;

-- ── CAMPAIGNS TABLE ───────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS campaigns (
  id               UUID         DEFAULT gen_random_uuid() PRIMARY KEY,
  campaign_id      TEXT         UNIQUE,
  type             TEXT,
  threat_type      TEXT,
  severity         TEXT         CHECK (severity IN ('LOW','MEDIUM','HIGH','CRITICAL')),
  events_count     INTEGER      DEFAULT 0,
  unique_users     INTEGER      DEFAULT 0,
  unique_sessions  INTEGER      DEFAULT 0,
  window_minutes   INTEGER,
  detected_at      TIMESTAMPTZ  DEFAULT NOW(),
  resolved_at      TIMESTAMPTZ,
  status           TEXT         DEFAULT 'ACTIVE' CHECK (status IN ('ACTIVE','RESOLVED','FALSE_POSITIVE'))
);

-- ── DYNAMIC RULES TABLE ───────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS dynamic_rules (
  id            UUID         DEFAULT gen_random_uuid() PRIMARY KEY,
  created_at    TIMESTAMPTZ  DEFAULT NOW(),
  pattern       TEXT         NOT NULL UNIQUE,
  threat_type   TEXT,
  source        TEXT         CHECK (source IN ('MUTATION_ENGINE','RED_TEAM_AGENT','MANUAL')),
  active        BOOLEAN      DEFAULT true,
  trigger_count INTEGER      DEFAULT 0
);

-- ── KNOWLEDGE BASE TABLE ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS knowledge_base (
  id              UUID         DEFAULT gen_random_uuid() PRIMARY KEY,
  created_at      TIMESTAMPTZ  DEFAULT NOW(),
  threat_type     TEXT,
  fingerprint_id  TEXT,
  sophistication  INTEGER,
  pattern_vector  TEXT,
  origin          TEXT         DEFAULT 'local',
  verified        BOOLEAN      DEFAULT false
);

-- ── ROW LEVEL SECURITY ────────────────────────────────────────────────────
-- SECURITY: Every table has RLS enabled. In Supabase:
--   • anon key  → subject to RLS policies. Only tables with explicit
--     "TO anon" SELECT policies are readable.
--   • service_role → BYPASSES RLS entirely (Supabase built-in behavior).
--     Write policies below are defense-in-depth only — they document intent
--     but do NOT enforce anything for service_role.
--
-- ACCESS MODEL:
--   READABLE BY ANON:  stats, battle_state, evolution_log, campaigns,
--                       attack_fingerprints, xai_decisions, threat_clusters
--   BLOCKED FOR ANON:  events (use events_safe view), red_team_sessions,
--                       dynamic_rules, knowledge_base
--     (no anon SELECT policy = RLS denies all reads)
--
-- CRITICAL: Sensitive tables (red_team_sessions, dynamic_rules,
-- knowledge_base) are NEVER readable by anon. They contain attack
-- intelligence that adversaries could harvest.

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

-- ── ANON READ POLICIES (dashboard-safe tables only) ──────────────────────
-- These tables contain aggregated/metadata only — no raw attack payloads.
-- NOTE: DROP before CREATE is REQUIRED — PostgreSQL errors if policy already exists.
DROP POLICY IF EXISTS "Anon read stats"        ON stats;
CREATE POLICY "Anon read stats"          ON stats               FOR SELECT TO anon USING (true);
DROP POLICY IF EXISTS "Anon read battle"       ON battle_state;
CREATE POLICY "Anon read battle"         ON battle_state        FOR SELECT TO anon USING (true);
DROP POLICY IF EXISTS "Anon read evolution"    ON evolution_log;
CREATE POLICY "Anon read evolution"      ON evolution_log       FOR SELECT TO anon USING (true);
DROP POLICY IF EXISTS "Anon read campaigns"    ON campaigns;
CREATE POLICY "Anon read campaigns"      ON campaigns           FOR SELECT TO anon USING (true);
DROP POLICY IF EXISTS "Anon read fingerprints" ON attack_fingerprints;
CREATE POLICY "Anon read fingerprints"   ON attack_fingerprints FOR SELECT TO anon USING (true);

-- SECURITY: Events table — anon reads through a secure view that strips
-- raw preview text. Direct table access is service_role only.
DROP POLICY IF EXISTS "Service only events"    ON events;
CREATE POLICY "Service only events"      ON events              FOR SELECT TO authenticated USING (auth.role() = 'service_role');

-- Create a safe view for anon reads (no raw payload previews)
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

-- Grant anon access to the safe view only
GRANT SELECT ON events_safe TO anon;

-- XAI decisions: anon can read reasoning (no raw payloads after chat.py redaction)
DROP POLICY IF EXISTS "Anon read xai"          ON xai_decisions;
CREATE POLICY "Anon read xai"            ON xai_decisions       FOR SELECT TO anon USING (true);

-- Threat clusters: anon can read (sample_text is now hashed after clusterer fix)
DROP POLICY IF EXISTS "Anon read clusters"     ON threat_clusters;
CREATE POLICY "Anon read clusters"       ON threat_clusters     FOR SELECT TO anon USING (true);

-- ── SENSITIVE TABLES: SERVICE ROLE ONLY ──────────────────────────────────
-- These tables contain raw attack payloads and firewall bypass patterns.
-- NEVER expose to anon key under any circumstances.
DROP POLICY IF EXISTS "Service only red_team"  ON red_team_sessions;
CREATE POLICY "Service only red_team"    ON red_team_sessions   FOR SELECT TO authenticated USING (auth.role() = 'service_role');
DROP POLICY IF EXISTS "Service only rules"     ON dynamic_rules;
CREATE POLICY "Service only rules"       ON dynamic_rules       FOR SELECT TO authenticated USING (auth.role() = 'service_role');
DROP POLICY IF EXISTS "Service only knowledge" ON knowledge_base;
CREATE POLICY "Service only knowledge"   ON knowledge_base      FOR SELECT TO authenticated USING (auth.role() = 'service_role');

-- ── SERVICE ROLE WRITE POLICIES ──────────────────────────────────────────
-- All writes restricted to service_role. Anon CANNOT insert, update, or delete.
DROP POLICY IF EXISTS "Service write events"       ON events;
CREATE POLICY "Service write events"       ON events              FOR INSERT TO authenticated WITH CHECK (auth.role() = 'service_role');
DROP POLICY IF EXISTS "Service write stats"        ON stats;
CREATE POLICY "Service write stats"        ON stats               FOR UPDATE TO authenticated USING (auth.role() = 'service_role');
DROP POLICY IF EXISTS "Service write xai"          ON xai_decisions;
CREATE POLICY "Service write xai"          ON xai_decisions       FOR INSERT TO authenticated WITH CHECK (auth.role() = 'service_role');
DROP POLICY IF EXISTS "Service write battle"       ON battle_state;
CREATE POLICY "Service write battle"       ON battle_state        FOR ALL    TO authenticated USING (auth.role() = 'service_role');
DROP POLICY IF EXISTS "Service write evolution"    ON evolution_log;
CREATE POLICY "Service write evolution"    ON evolution_log       FOR INSERT TO authenticated WITH CHECK (auth.role() = 'service_role');
DROP POLICY IF EXISTS "Service write clusters"     ON threat_clusters;
CREATE POLICY "Service write clusters"     ON threat_clusters     FOR ALL    TO authenticated USING (auth.role() = 'service_role');
DROP POLICY IF EXISTS "Service write red_team"     ON red_team_sessions;
CREATE POLICY "Service write red_team"     ON red_team_sessions   FOR INSERT TO authenticated WITH CHECK (auth.role() = 'service_role');
DROP POLICY IF EXISTS "Service write fingerprints" ON attack_fingerprints;
CREATE POLICY "Service write fingerprints" ON attack_fingerprints FOR ALL    TO authenticated USING (auth.role() = 'service_role');
DROP POLICY IF EXISTS "Service write campaigns"    ON campaigns;
CREATE POLICY "Service write campaigns"    ON campaigns           FOR ALL    TO authenticated USING (auth.role() = 'service_role');
DROP POLICY IF EXISTS "Service write rules"        ON dynamic_rules;
CREATE POLICY "Service write rules"        ON dynamic_rules       FOR ALL    TO authenticated USING (auth.role() = 'service_role');
DROP POLICY IF EXISTS "Service write knowledge"    ON knowledge_base;
CREATE POLICY "Service write knowledge"    ON knowledge_base      FOR ALL    TO authenticated USING (auth.role() = 'service_role');

-- ── SAFE VIEW: bypass history (no raw attack_text) ───────────────────────
DROP VIEW IF EXISTS bypass_history_safe;
CREATE VIEW bypass_history_safe WITH (security_invoker = true) AS
SELECT
  id, created_at, attack_type, tier, bypassed, auto_patched,
  cycle, score, latency_ms
FROM red_team_sessions
WHERE bypassed = true AND auto_patched = true;

-- ── ENABLE REALTIME ───────────────────────────────────────────────────────
ALTER TABLE events              REPLICA IDENTITY FULL;
ALTER TABLE battle_state        REPLICA IDENTITY FULL;
ALTER TABLE campaigns           REPLICA IDENTITY FULL;
ALTER TABLE xai_decisions       REPLICA IDENTITY FULL;
ALTER TABLE red_team_sessions   REPLICA IDENTITY FULL;
ALTER TABLE attack_fingerprints REPLICA IDENTITY FULL;

-- ── ANALYTICS VIEWS ───────────────────────────────────────────────────────
CREATE OR REPLACE VIEW threat_summary AS
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

CREATE OR REPLACE VIEW attack_heatmap AS
SELECT
  EXTRACT(HOUR FROM created_at)   AS hour_of_day,
  EXTRACT(DOW  FROM created_at)   AS day_of_week,
  COUNT(*)                         AS threat_count
FROM events
WHERE action = 'BLOCKED'
  AND created_at > NOW() - INTERVAL '7 days'
GROUP BY 1, 2;

-- ── VERIFICATION ──────────────────────────────────────────────────────────
DO $$
BEGIN
  RAISE NOTICE '✅ ARGUS-X Schema v3.0 created successfully';
  RAISE NOTICE 'Tables: events, stats, xai_decisions, battle_state, evolution_log, threat_clusters, red_team_sessions, attack_fingerprints, campaigns, dynamic_rules, knowledge_base';
  RAISE NOTICE 'Next steps:';
  RAISE NOTICE '  1. Enable Realtime on: events, battle_state, campaigns, xai_decisions';
  RAISE NOTICE '  2. Create Storage bucket "argus-models" (Private)';
  RAISE NOTICE '  3. Upload ONNX model files to the bucket';
END $$;
