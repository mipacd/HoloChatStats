-- A pending job can retain dispatched_at after its last SQS delivery reaches
-- the DLQ. The scheduled reaper clears these markers after this grace period,
-- allowing the month-ordered dispatcher to recreate the delivery.
INSERT INTO service_config (key, value, updated_at)
VALUES ('stale_dispatched_minutes', '15', NOW())
ON CONFLICT (key) DO NOTHING;
