-- The watermark is "newest video we are DONE with", not "newest video we know
-- of". backfill_jobs() previously wrote the latter, which pins discovery's
-- floor above un-ingested months and makes them permanently undiscoverable.
-- GREATEST() cannot repair this; it only ever moves the watermark forward.
UPDATE channel_watermarks w
SET last_seen_end_time = p.processed_through
FROM (
    SELECT v.channel_id,
           MAX(v.end_time) FILTER (
               WHERE v.has_chat_log OR j.status IN ('done','skipped')
           ) AS processed_through
    FROM videos v LEFT JOIN ingest_jobs j USING (video_id)
    GROUP BY v.channel_id
) p
WHERE p.channel_id = w.channel_id
  AND w.last_seen_end_time IS DISTINCT FROM p.processed_through;
-- Channels whose entire history is unprocessed: drop the watermark so the
-- per-channel baseline falls through to service_config.ingest_start_date.
DELETE FROM channel_watermarks WHERE last_seen_end_time IS NULL;