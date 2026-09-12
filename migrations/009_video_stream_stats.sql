-- Aggregate-only per-stream statistics. No message text or user identifiers
-- are persisted in this table.
CREATE TABLE IF NOT EXISTS video_stream_stats (
    video_id                 TEXT PRIMARY KEY REFERENCES videos(video_id) ON DELETE CASCADE,
    schema_version           SMALLINT NOT NULL DEFAULT 1,
    status                   TEXT NOT NULL DEFAULT 'pending',
    message_count            BIGINT,
    unique_chatters          BIGINT,
    member_chatters          BIGINT,
    member_percentage        NUMERIC(6,3),
    category_counts          JSONB,
    membership_rank_counts   JSONB,
    histogram_bin_seconds    INTEGER,
    histogram_counts         JSONB,
    funny_moments            JSONB,
    word_counts              JSONB,
    first_message_at         TIMESTAMPTZ,
    attempts                 INTEGER NOT NULL DEFAULT 0,
    last_error               TEXT,
    computed_at              TIMESTAMPTZ,
    updated_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT video_stream_stats_status CHECK (
        status IN ('pending', 'queued', 'processing', 'ready',
                   'unavailable', 'failed'))
);
ALTER TABLE video_stream_stats
    ADD COLUMN IF NOT EXISTS member_percentage NUMERIC(6,3);
CREATE INDEX IF NOT EXISTS idx_video_stream_stats_status
    ON video_stream_stats(status, updated_at);
CREATE INDEX IF NOT EXISTS idx_videos_end_time_video_id
    ON videos(end_time DESC, video_id);
