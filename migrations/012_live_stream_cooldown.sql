-- Keep active/upcoming streams out of the hot download loop.  The dispatcher
-- is the only component which releases a cooled job once it becomes eligible.
ALTER TABLE ingest_jobs
    ADD COLUMN IF NOT EXISTS next_attempt_at TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS idx_ingest_jobs_pending_ready
    ON ingest_jobs (next_attempt_at, enqueued_at)
    WHERE status = 'pending' AND dispatched_at IS NULL;

INSERT INTO service_config (key, value, updated_at) VALUES
    ('live_retry_cooldown_minutes', '120', NOW()),
    ('live_cooldown_max_attempts',  '6',   NOW()),
    ('upcoming_start_grace_hours',  '12',  NOW()),
    ('future_stream_max_days',      '30',  NOW()),
    ('max_live_stream_hours',       '48',  NOW())
ON CONFLICT (key) DO NOTHING;

-- Recover rows produced by the former rapid-retry behavior.  Do not enqueue
-- here: the dispatcher will release them after the same durable cooldown used
-- for newly detected live responses.
UPDATE ingest_jobs
   SET status='pending', attempts=0, dispatched_at=NULL, lease_id=NULL,
       completed_at=NULL, skip_reason=NULL,
       next_attempt_at=NOW() + (
           COALESCE((SELECT value::int FROM service_config
                     WHERE key='live_retry_cooldown_minutes'), 120)
           * INTERVAL '1 minute'),
       updated_at=NOW()
 WHERE status='failed'
   AND (last_error ILIKE '%expecting property name enclosed in double quotes%'
        OR last_error ILIKE '%expecting value: line 1 column 1%');
