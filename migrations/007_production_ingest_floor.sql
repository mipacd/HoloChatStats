-- This rewrite intentionally begins at exactly 2026-07-01 00:00:00 UTC.
-- Unlike the original seed migration, this upgrades restored installations
-- which already have older service_config values.
INSERT INTO service_config (key, value) VALUES
    ('ingest_start_date', '2026-07-01T00:00:00+00:00'),
    ('backlog_floor',     '2026-07-01T00:00:00+00:00')
ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value;
