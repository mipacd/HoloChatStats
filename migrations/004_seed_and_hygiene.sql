-- ===========================================================================
-- Runtime defaults and standing data hygiene. Safe and cheap to re-run: every
-- INSERT is ON CONFLICT DO NOTHING (so your tuned values are never clobbered)
-- and every cleanup is a no-op on a fresh database.
-- ===========================================================================
INSERT INTO service_config (key, value) VALUES
    -- discovery
    ('ingest_start_date',         '2026-07-01T00:00:00+00:00'),
                                                 -- required UTC ETL floor
                                                 -- with NO history at all
    ('max_channels_per_run',      '25'),
    ('min_scan_interval_minutes', '360'),
    ('discovery_lookback_hours',  '72'),         -- re-check window behind watermark
    ('discovery_queue_threshold', '50'),         -- backpressure
    ('scan_page_delay_ms',        '250'),
    -- concurrency
    ('max_concurrent_downloads',  '1'),
    ('max_concurrent_scans',      '3'),
    ('max_retries',               '5'),
    -- monthly staging / merge
    ('current_month_staging',     'true'),
    ('merge_grace_hours',         '24'),
    ('merge_ignore_failed',       'false'),
    -- stall recovery (handlers/reap.py); the download heartbeat is ~5s, so
    -- 15 minutes of silence is 180x slack
    ('stale_download_minutes',    '15'),
    ('stale_ingest_minutes',      '30'),
    ('stale_downloaded_minutes',  '30'),
    ('max_reaps_per_job',         '5'),
    ('reap_batch_size',           '25'),
    -- kill switch, flipped by the admin page
    ('paused',                    'false')
ON CONFLICT (key) DO NOTHING;
-- ---------------------------------------------------------------------------
-- Underscore-prefixed channels are placeholders: never scanned.
-- (The trigger in 001 enforces this going forward; this fixes existing rows.)
-- ---------------------------------------------------------------------------
UPDATE channels SET active = FALSE, updated_at = NOW()
WHERE (LEFT(channel_name, 1) = '_' OR LEFT(COALESCE(channel_group, ''), 1) = '_')
  AND active;
-- Cancel non-terminal work for inactive or unknown channels. Deletes JOB ROWS
-- only: videos, user_data, user_data_current and terminal jobs are untouched,
-- so no chat data is removed. handlers/reap.py repeats this every 5 minutes.
DELETE FROM ingest_jobs j
USING channels c
WHERE c.channel_id = j.channel_id
  AND NOT c.active
  AND j.status IN ('pending', 'downloading', 'downloaded', 'ingesting');
DELETE FROM ingest_jobs j
WHERE NOT EXISTS (SELECT 1 FROM channels c WHERE c.channel_id = j.channel_id)
  AND j.status IN ('pending', 'downloading', 'downloaded', 'ingesting');
-- ---------------------------------------------------------------------------
-- Seed discovery watermarks from whatever data already exists, so a restored
-- database never re-walks history. handlers/scan.py recomputes this baseline
-- from `videos` anyway; this is so channel_watermarks and the admin page tell
-- the truth immediately after a restore.
-- ---------------------------------------------------------------------------
INSERT INTO channel_watermarks (channel_id, last_seen_end_time)
SELECT v.channel_id, MAX(v.end_time)
FROM videos v
LEFT JOIN ingest_jobs j USING (video_id)
WHERE v.channel_id IS NOT NULL
  AND (v.has_chat_log OR j.status IN ('done', 'skipped'))
GROUP BY v.channel_id
ON CONFLICT (channel_id) DO UPDATE
  SET last_seen_end_time = GREATEST(
          COALESCE(channel_watermarks.last_seen_end_time,
                   EXCLUDED.last_seen_end_time),
          EXCLUDED.last_seen_end_time);
