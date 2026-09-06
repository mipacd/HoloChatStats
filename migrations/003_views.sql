-- ===========================================================================
-- The reader-facing union and the four materialized views.
--
-- Every MV reads user_data_all, so the current-month staging split is invisible
-- to consumers. Every MV's unique index is DERIVED from its GROUP BY or from the
-- source table's key -- never imposed by deduplicating rows the data really has.
-- That distinction is why an earlier revision's SELECT DISTINCT on
-- mv_user_activity was wrong: it silently changed the view's grain in order to
-- satisfy REFRESH ... CONCURRENTLY.
-- ===========================================================================
DROP VIEW IF EXISTS v_user_active_months;
DROP MATERIALIZED VIEW IF EXISTS mv_user_activity;
DROP MATERIALIZED VIEW IF EXISTS mv_user_monthly_activity;
DROP MATERIALIZED VIEW IF EXISTS chat_language_stats_mv;
DROP MATERIALIZED VIEW IF EXISTS mv_user_language_per_month;
CREATE OR REPLACE VIEW user_data_all AS
SELECT user_id, channel_id, last_message_at, video_id, membership_rank,
       jp_count, kr_count, ru_count, emoji_count, es_en_id_count,
       total_message_count, is_gift, FALSE AS is_staged
FROM user_data
UNION ALL
SELECT user_id, channel_id, last_message_at, video_id, membership_rank,
       jp_count, kr_count, ru_count, emoji_count, es_en_id_count,
       total_message_count, is_gift, TRUE AS is_staged
FROM user_data_current;
-- ---------------------------------------------------------------------------
-- Grain: one row per user per VIDEO. COUNT(*) therefore means "videos this user
-- chatted in". For "did this user appear at all this month", use the
-- v_user_active_months view below.
-- ---------------------------------------------------------------------------
CREATE MATERIALIZED VIEW mv_user_activity AS
SELECT
    ud.user_id,
    ud.channel_id,
    c.channel_group,
    DATE_TRUNC('month', ud.last_message_at) AS activity_month,
    ud.video_id,
    ud.last_message_at,
    ud.total_message_count AS message_count
FROM user_data_all ud
JOIN channels c ON c.channel_id = ud.channel_id
WHERE ud.total_message_count > 0
WITH NO DATA;   -- excludes gift-only membership rows
-- The key of user_data / user_data_current: a true key, not an assertion.
CREATE UNIQUE INDEX uq_mv_user_activity
    ON mv_user_activity (user_id, channel_id, last_message_at, video_id);
CREATE INDEX idx_mv_user_activity_channel ON mv_user_activity (channel_id);
CREATE INDEX idx_mv_user_activity_group   ON mv_user_activity (channel_group);
CREATE INDEX idx_mv_user_activity_month   ON mv_user_activity (activity_month);
CREATE INDEX idx_mv_user_activity_user_id ON mv_user_activity (user_id);
CREATE INDEX idx_mv_user_activity_video   ON mv_user_activity (video_id);
CREATE INDEX idx_activity_month_user      ON mv_user_activity (activity_month, user_id);
CREATE MATERIALIZED VIEW mv_user_monthly_activity AS
SELECT user_id, channel_id,
       DATE_TRUNC('month', last_message_at) AS observed_month,
       SUM(total_message_count) AS monthly_message_count
FROM user_data_all
WHERE total_message_count > 0
GROUP BY user_id, channel_id, observed_month
WITH NO DATA;
CREATE UNIQUE INDEX uq_mv_user_monthly_activity
    ON mv_user_monthly_activity (user_id, channel_id, observed_month);
CREATE INDEX idx_mv_user_monthly_activity
    ON mv_user_monthly_activity (observed_month, channel_id);
CREATE MATERIALIZED VIEW chat_language_stats_mv AS
SELECT channel_id,
       DATE_TRUNC('month', last_message_at) AS observed_month,
       SUM(jp_count) AS jp_count, SUM(kr_count) AS kr_count,
       SUM(ru_count) AS ru_count, SUM(emoji_count) AS emoji_count,
       SUM(es_en_id_count) AS es_en_id_count,
       SUM(total_message_count) AS total_messages
FROM user_data_all
WHERE total_message_count > 0
GROUP BY channel_id, DATE_TRUNC('month', last_message_at)
WITH NO DATA;
CREATE UNIQUE INDEX uq_chat_language_stats_mv
    ON chat_language_stats_mv (channel_id, observed_month);
CREATE MATERIALIZED VIEW mv_user_language_per_month AS
SELECT ud.user_id, ud.channel_id,
       DATE_TRUNC('month', ud.last_message_at) AS month,
       SUM(ud.jp_count) AS total_jp_messages,
       SUM(ud.total_message_count - ud.emoji_count) AS total_non_emoji_messages
FROM user_data_all ud
WHERE ud.total_message_count > 0
GROUP BY ud.user_id, ud.channel_id, month
WITH NO DATA;
CREATE UNIQUE INDEX uq_mv_user_language_per_month
    ON mv_user_language_per_month (user_id, channel_id, month);
-- Presence map: what the old DISTINCT version was accidentally answering.
CREATE VIEW v_user_active_months AS
SELECT DISTINCT user_id, channel_id, channel_group, activity_month
FROM mv_user_activity;