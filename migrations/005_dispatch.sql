-- Dispatch bookkeeping: a pending job is now "discovered but not yet sent".
ALTER TABLE ingest_jobs ADD COLUMN IF NOT EXISTS dispatched_at TIMESTAMPTZ;
CREATE INDEX IF NOT EXISTS idx_ingest_jobs_undispatched
    ON ingest_jobs (channel_id) WHERE status = 'pending' AND dispatched_at IS NULL;
INSERT INTO service_config (key, value) VALUES
    -- Months strictly before this are never dispatched. Set it to the first
    -- month you actually want backfilled (e.g. 2026-07-01).
    ('backlog_floor',        '2026-07-01T00:00:00+00:00'),
    ('dispatch_batch_size',  '50')
ON CONFLICT (key) DO NOTHING;
