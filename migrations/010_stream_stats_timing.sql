-- Stable replay-timeline diagnostics for per-stream statistics.  These fields
-- contain aggregate timing only; no message or user data is retained.
ALTER TABLE video_stream_stats
    ADD COLUMN IF NOT EXISTS timing_source TEXT,
    ADD COLUMN IF NOT EXISTS first_offset_seconds DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS last_offset_seconds DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS out_of_range_messages BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS quiet_tail_seconds DOUBLE PRECISION;

CREATE INDEX IF NOT EXISTS idx_video_stream_stats_timing_audit
    ON video_stream_stats(status, schema_version, video_id);
