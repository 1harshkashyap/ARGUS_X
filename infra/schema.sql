-- ARGUS-X Complete Database Schema v2.0
-- Run once on Day 1

-- 1. Security events table
CREATE TABLE IF NOT EXISTS events (
  id                   UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  session_id           TEXT DEFAULT '',
  user_id              TEXT DEFAULT '',
  message_preview      TEXT DEFAULT '',
  blocked              BOOLEAN DEFAULT FALSE,
  sanitized            BOOLEAN DEFAULT FALSE,
  threat_type          TEXT DEFAULT 'CLEAN',
  threat_score         FLOAT DEFAULT 0.0,
  sophistication_score INTEGER DEFAULT 0,
  attack_fingerprint   TEXT DEFAULT '',
  mutations_count      INTEGER DEFAULT 0,
  llm_mode             TEXT DEFAULT '',
  latency_ms           FLOAT DEFAULT 0.0,
  created_at           TIMESTAMPTZ DEFAULT NOW()
);

-- 2. XAI decisions table
CREATE TABLE IF NOT EXISTS xai_decisions (
  id                   UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  event_id             UUID REFERENCES events(id) ON DELETE CASCADE,
  session_id           TEXT DEFAULT '',
  message_preview      TEXT DEFAULT '',
  verdict              TEXT DEFAULT 'CLEAN',
  primary_reason       TEXT DEFAULT '',
  pattern_family       TEXT DEFAULT '',
  sophistication_label TEXT DEFAULT '',
  layer_decisions      JSONB DEFAULT '[]',
  recommended_action   TEXT DEFAULT 'ALLOW',
  created_at           TIMESTAMPTZ DEFAULT NOW()
);

-- 3. Battle state table (single row, updated every 60 seconds)
CREATE TABLE IF NOT EXISTS battle_state (
  id                     INTEGER PRIMARY KEY DEFAULT 1,
  tick                   INTEGER DEFAULT 0,
  red_attacks            INTEGER DEFAULT 0,
  red_bypasses           INTEGER DEFAULT 0,
  blue_blocks            INTEGER DEFAULT 0,
  blue_patches           INTEGER DEFAULT 0,
  red_tier               INTEGER DEFAULT 1,
  red_strategy           TEXT DEFAULT 'NAIVE',
  current_attack_preview TEXT DEFAULT '',
  last_attack_result     TEXT DEFAULT 'BLOCKED',
  updated_at             TIMESTAMPTZ DEFAULT NOW()
);
INSERT INTO battle_state (id) VALUES (1) ON CONFLICT (id) DO NOTHING;

-- 4. Threat campaigns table
CREATE TABLE IF NOT EXISTS campaigns (
  id              UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  campaign_id     TEXT UNIQUE DEFAULT '',
  attack_pattern  TEXT DEFAULT '',
  source_sessions TEXT[] DEFAULT '{}',
  hit_count       INTEGER DEFAULT 1,
  severity        TEXT DEFAULT 'MEDIUM',
  first_seen      TIMESTAMPTZ DEFAULT NOW(),
  last_seen       TIMESTAMPTZ DEFAULT NOW()
);

-- 5. Dynamic firewall rules table
CREATE TABLE IF NOT EXISTS dynamic_rules (
  id            UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  pattern       TEXT UNIQUE DEFAULT '',
  threat_type   TEXT DEFAULT 'PROMPT_INJECTION',
  source_attack TEXT DEFAULT '',
  created_at    TIMESTAMPTZ DEFAULT NOW()
);

-- 6. Running statistics table (single row)
CREATE TABLE IF NOT EXISTS stats (
  id              INTEGER PRIMARY KEY DEFAULT 1,
  total_events    INTEGER DEFAULT 0,
  total_blocked   INTEGER DEFAULT 0,
  total_bypasses  INTEGER DEFAULT 0,
  total_mutations INTEGER DEFAULT 0,
  updated_at      TIMESTAMPTZ DEFAULT NOW()
);
INSERT INTO stats (id) VALUES (1) ON CONFLICT (id) DO NOTHING;

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_events_created_at   ON events(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_events_session_id   ON events(session_id);
CREATE INDEX IF NOT EXISTS idx_events_blocked       ON events(blocked);
CREATE INDEX IF NOT EXISTS idx_events_threat_type   ON events(threat_type);
CREATE INDEX IF NOT EXISTS idx_xai_event_id         ON xai_decisions(event_id);
CREATE INDEX IF NOT EXISTS idx_campaigns_pattern    ON campaigns(attack_pattern);
CREATE INDEX IF NOT EXISTS idx_dynamic_rules_type   ON dynamic_rules(threat_type);

-- Enable Realtime on critical tables
ALTER PUBLICATION supabase_realtime ADD TABLE events;
ALTER PUBLICATION supabase_realtime ADD TABLE battle_state;
ALTER PUBLICATION supabase_realtime ADD TABLE campaigns;
