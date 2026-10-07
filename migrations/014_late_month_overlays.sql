-- Month-scoped overlays for late data in already-published months.
--
-- Rebuilding each full-history materialized view took longer than Lambda's
-- 900-second ceiling on the production host.  These tables contain a complete
-- replacement for only the months explicitly listed in late_month_overrides.
-- Reader-facing views select the materialized-view base for ordinary months
-- and the replacement rows for overridden months.

CREATE TABLE IF NOT EXISTS late_month_overrides (
    observed_month DATE PRIMARY KEY,
    published_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS late_mv_user_monthly_activity (
    user_id TEXT NOT NULL,
    channel_id TEXT NOT NULL,
    observed_month TIMESTAMPTZ NOT NULL,
    monthly_message_count BIGINT NOT NULL,
    PRIMARY KEY (user_id, channel_id, observed_month)
);
CREATE INDEX IF NOT EXISTS idx_late_monthly_activity_month_channel
    ON late_mv_user_monthly_activity (observed_month, channel_id);

CREATE TABLE IF NOT EXISTS late_mv_user_activity (
    user_id TEXT NOT NULL,
    channel_id TEXT NOT NULL,
    channel_group TEXT,
    activity_month TIMESTAMPTZ NOT NULL,
    video_id TEXT NOT NULL,
    last_message_at TIMESTAMPTZ NOT NULL,
    message_count BIGINT NOT NULL,
    PRIMARY KEY (user_id, channel_id, last_message_at, video_id)
);
CREATE INDEX IF NOT EXISTS idx_late_user_activity_channel
    ON late_mv_user_activity (channel_id);
CREATE INDEX IF NOT EXISTS idx_late_user_activity_group
    ON late_mv_user_activity (channel_group);
CREATE INDEX IF NOT EXISTS idx_late_user_activity_month
    ON late_mv_user_activity (activity_month);
CREATE INDEX IF NOT EXISTS idx_late_user_activity_user
    ON late_mv_user_activity (user_id);

CREATE TABLE IF NOT EXISTS late_chat_language_stats (
    channel_id TEXT NOT NULL,
    observed_month TIMESTAMPTZ NOT NULL,
    jp_count BIGINT NOT NULL,
    kr_count BIGINT NOT NULL,
    ru_count BIGINT NOT NULL,
    emoji_count BIGINT NOT NULL,
    es_en_id_count BIGINT NOT NULL,
    total_messages BIGINT NOT NULL,
    PRIMARY KEY (channel_id, observed_month)
);

CREATE TABLE IF NOT EXISTS late_mv_user_language_per_month (
    user_id TEXT NOT NULL,
    channel_id TEXT NOT NULL,
    month TIMESTAMPTZ NOT NULL,
    total_jp_messages BIGINT NOT NULL,
    total_non_emoji_messages BIGINT NOT NULL,
    PRIMARY KEY (user_id, channel_id, month)
);

CREATE OR REPLACE PROCEDURE refresh_late_month_overlay(
    target_month DATE, target_stage INTEGER)
LANGUAGE plpgsql
AS $$
DECLARE
    month_end DATE := (target_month + INTERVAL '1 month')::date;
BEGIN
    IF target_stage = 0 THEN
        DELETE FROM late_mv_user_monthly_activity
         WHERE observed_month::date = target_month;
        INSERT INTO late_mv_user_monthly_activity
            (user_id, channel_id, observed_month, monthly_message_count)
        SELECT user_id, channel_id,
               DATE_TRUNC('month', last_message_at),
               SUM(total_message_count)
          FROM user_data_all
         WHERE total_message_count > 0
           AND last_message_at >= target_month
           AND last_message_at < month_end
         GROUP BY user_id, channel_id, DATE_TRUNC('month', last_message_at);
    ELSIF target_stage = 1 THEN
        DELETE FROM late_mv_user_activity
         WHERE activity_month::date = target_month;
        INSERT INTO late_mv_user_activity
            (user_id, channel_id, channel_group, activity_month, video_id,
             last_message_at, message_count)
        SELECT ud.user_id, ud.channel_id, c.channel_group,
               DATE_TRUNC('month', ud.last_message_at), ud.video_id,
               ud.last_message_at, ud.total_message_count
          FROM user_data_all ud
          JOIN channels c ON c.channel_id = ud.channel_id
         WHERE ud.total_message_count > 0
           AND ud.last_message_at >= target_month
           AND ud.last_message_at < month_end;
    ELSIF target_stage = 2 THEN
        DELETE FROM late_chat_language_stats
         WHERE observed_month::date = target_month;
        INSERT INTO late_chat_language_stats
            (channel_id, observed_month, jp_count, kr_count, ru_count,
             emoji_count, es_en_id_count, total_messages)
        SELECT channel_id, DATE_TRUNC('month', last_message_at),
               SUM(jp_count), SUM(kr_count), SUM(ru_count), SUM(emoji_count),
               SUM(es_en_id_count), SUM(total_message_count)
          FROM user_data_all
         WHERE total_message_count > 0
           AND last_message_at >= target_month
           AND last_message_at < month_end
         GROUP BY channel_id, DATE_TRUNC('month', last_message_at);
    ELSIF target_stage = 3 THEN
        DELETE FROM late_mv_user_language_per_month
         WHERE month::date = target_month;
        INSERT INTO late_mv_user_language_per_month
            (user_id, channel_id, month, total_jp_messages,
             total_non_emoji_messages)
        SELECT user_id, channel_id, DATE_TRUNC('month', last_message_at),
               SUM(jp_count), SUM(total_message_count - emoji_count)
          FROM user_data_all
         WHERE total_message_count > 0
           AND last_message_at >= target_month
           AND last_message_at < month_end
         GROUP BY user_id, channel_id, DATE_TRUNC('month', last_message_at);
    ELSE
        RAISE EXCEPTION 'unknown late-month overlay stage: %', target_stage;
    END IF;
END;
$$;

CREATE OR REPLACE VIEW mv_user_monthly_activity_live AS
SELECT b.user_id, b.channel_id, b.observed_month, b.monthly_message_count
  FROM mv_user_monthly_activity b
 WHERE NOT EXISTS (
       SELECT 1 FROM late_month_overrides o
        WHERE o.observed_month = b.observed_month::date)
UNION ALL
SELECT l.user_id, l.channel_id, l.observed_month, l.monthly_message_count
  FROM late_mv_user_monthly_activity l
  JOIN late_month_overrides o
    ON o.observed_month = l.observed_month::date;

CREATE OR REPLACE VIEW mv_user_activity_live AS
SELECT b.user_id, b.channel_id, b.channel_group, b.activity_month, b.video_id,
       b.last_message_at, b.message_count
  FROM mv_user_activity b
 WHERE NOT EXISTS (
       SELECT 1 FROM late_month_overrides o
        WHERE o.observed_month = b.activity_month::date)
UNION ALL
SELECT l.user_id, l.channel_id, l.channel_group, l.activity_month, l.video_id,
       l.last_message_at, l.message_count
  FROM late_mv_user_activity l
  JOIN late_month_overrides o
    ON o.observed_month = l.activity_month::date;

CREATE OR REPLACE VIEW chat_language_stats_live AS
SELECT b.channel_id, b.observed_month, b.jp_count, b.kr_count, b.ru_count,
       b.emoji_count, b.es_en_id_count, b.total_messages
  FROM chat_language_stats_mv b
 WHERE NOT EXISTS (
       SELECT 1 FROM late_month_overrides o
        WHERE o.observed_month = b.observed_month::date)
UNION ALL
SELECT l.channel_id, l.observed_month, l.jp_count, l.kr_count, l.ru_count,
       l.emoji_count, l.es_en_id_count, l.total_messages
  FROM late_chat_language_stats l
  JOIN late_month_overrides o
    ON o.observed_month = l.observed_month::date;

CREATE OR REPLACE VIEW mv_user_language_per_month_live AS
SELECT b.user_id, b.channel_id, b.month, b.total_jp_messages,
       b.total_non_emoji_messages
  FROM mv_user_language_per_month b
 WHERE NOT EXISTS (
       SELECT 1 FROM late_month_overrides o
        WHERE o.observed_month = b.month::date)
UNION ALL
SELECT l.user_id, l.channel_id, l.month, l.total_jp_messages,
       l.total_non_emoji_messages
  FROM late_mv_user_language_per_month l
  JOIN late_month_overrides o ON o.observed_month = l.month::date;

CREATE OR REPLACE VIEW v_user_active_months AS
SELECT DISTINCT user_id, channel_id, channel_group, activity_month
FROM mv_user_activity_live;
