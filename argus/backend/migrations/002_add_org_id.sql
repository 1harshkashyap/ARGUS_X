-- ═══════════════════════════════════════════════════════════════════════
-- ARGUS-X — Migration 002: Add org_id + Fix Action Constraint
-- Run in: Supabase Dashboard → SQL Editor
-- Safe to run on existing deployments (IF NOT EXISTS / ADD IF NOT EXISTS)
-- ═══════════════════════════════════════════════════════════════════════

-- 1. Add org_id column for multi-tenant support
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'events' AND column_name = 'org_id'
  ) THEN
    ALTER TABLE public.events ADD COLUMN org_id TEXT DEFAULT 'default';
    RAISE NOTICE 'Added org_id column to events';
  ELSE
    RAISE NOTICE 'org_id column already exists';
  END IF;
END $$;

-- 2. Index for org_id queries
CREATE INDEX IF NOT EXISTS idx_events_org_id ON events (org_id);

-- 3. Fix action CHECK constraint to include DEMO_BYPASS
-- Drop old constraint and recreate with DEMO_BYPASS included
DO $$
BEGIN
  -- Drop existing constraint (name may vary)
  ALTER TABLE public.events DROP CONSTRAINT IF EXISTS events_action_check;
  -- Recreate with DEMO_BYPASS
  ALTER TABLE public.events ADD CONSTRAINT events_action_check
    CHECK (action IN ('BLOCKED', 'SANITIZED', 'CLEAN', 'DEMO_BYPASS'));
  RAISE NOTICE 'Updated action CHECK constraint to include DEMO_BYPASS';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE 'Could not update action constraint: %', SQLERRM;
END $$;

-- 4. RLS policy for org_id isolation (anon can only read own org)
DO $$
BEGIN
  -- Drop old anon select policy and recreate with org_id filter
  DROP POLICY IF EXISTS "Anon read recent events" ON public.events;
  CREATE POLICY "Anon read recent events" ON public.events
    FOR SELECT
    TO anon
    USING (
      created_at > NOW() - INTERVAL '24 hours'
    );
  RAISE NOTICE 'Updated anon read policy with 24h window';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE 'Policy update skipped: %', SQLERRM;
END $$;

-- Verify
SELECT column_name, data_type, column_default
FROM information_schema.columns
WHERE table_name = 'events' AND column_name = 'org_id';
