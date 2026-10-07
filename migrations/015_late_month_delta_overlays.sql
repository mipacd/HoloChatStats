-- Replace full-month late-data rebuilds with small late-video deltas.
--
-- A completed late stream is one whose ingest finished after the month's
-- original publication.  If a global materialized-view refresh already
-- captured that video, it is excluded from the delta.  This makes the common
-- case (a handful of late logs) proportional to those logs rather than to all
-- historical user_data or even every row in the target month.

CREATE OR REPLACE PROCEDURE refresh_late_month_delta(
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
        WITH late_videos AS (
            SELECT DISTINCT j.video_id
              FROM ingest_jobs j
              JOIN videos v USING (video_id)
              JOIN monthly_merge_state s
                ON s.observed_month = target_month AND s.status = 'merged'
             WHERE j.status = 'done'
               AND j.completed_at > s.merged_at
               AND v.end_time >= target_month
               AND v.end_time < month_end
               AND NOT EXISTS (
                   SELECT 1 FROM mv_user_activity b
                    WHERE b.video_id = j.video_id)
        )
        SELECT ud.user_id, ud.channel_id,
               DATE_TRUNC('month', ud.last_message_at),
               SUM(ud.total_message_count)
          FROM user_data_all ud
          JOIN late_videos lv USING (video_id)
         WHERE ud.total_message_count > 0
         GROUP BY ud.user_id, ud.channel_id,
                  DATE_TRUNC('month', ud.last_message_at);
    ELSIF target_stage = 1 THEN
        DELETE FROM late_mv_user_activity
         WHERE activity_month::date = target_month;
        INSERT INTO late_mv_user_activity
            (user_id, channel_id, channel_group, activity_month, video_id,
             last_message_at, message_count)
        WITH late_videos AS (
            SELECT DISTINCT j.video_id
              FROM ingest_jobs j
              JOIN videos v USING (video_id)
              JOIN monthly_merge_state s
                ON s.observed_month = target_month AND s.status = 'merged'
             WHERE j.status = 'done'
               AND j.completed_at > s.merged_at
               AND v.end_time >= target_month
               AND v.end_time < month_end
               AND NOT EXISTS (
                   SELECT 1 FROM mv_user_activity b
                    WHERE b.video_id = j.video_id)
        )
        SELECT ud.user_id, ud.channel_id, c.channel_group,
               DATE_TRUNC('month', ud.last_message_at), ud.video_id,
               ud.last_message_at, ud.total_message_count
          FROM user_data_all ud
          JOIN late_videos lv USING (video_id)
          JOIN channels c ON c.channel_id = ud.channel_id
         WHERE ud.total_message_count > 0;
    ELSIF target_stage = 2 THEN
        DELETE FROM late_chat_language_stats
         WHERE observed_month::date = target_month;
        INSERT INTO late_chat_language_stats
            (channel_id, observed_month, jp_count, kr_count, ru_count,
             emoji_count, es_en_id_count, total_messages)
        WITH late_videos AS (
            SELECT DISTINCT j.video_id
              FROM ingest_jobs j
              JOIN videos v USING (video_id)
              JOIN monthly_merge_state s
                ON s.observed_month = target_month AND s.status = 'merged'
             WHERE j.status = 'done'
               AND j.completed_at > s.merged_at
               AND v.end_time >= target_month
               AND v.end_time < month_end
               AND NOT EXISTS (
                   SELECT 1 FROM mv_user_activity b
                    WHERE b.video_id = j.video_id)
        )
        SELECT ud.channel_id, DATE_TRUNC('month', ud.last_message_at),
               SUM(ud.jp_count), SUM(ud.kr_count), SUM(ud.ru_count),
               SUM(ud.emoji_count), SUM(ud.es_en_id_count),
               SUM(ud.total_message_count)
          FROM user_data_all ud
          JOIN late_videos lv USING (video_id)
         WHERE ud.total_message_count > 0
         GROUP BY ud.channel_id, DATE_TRUNC('month', ud.last_message_at);
    ELSIF target_stage = 3 THEN
        DELETE FROM late_mv_user_language_per_month
         WHERE month::date = target_month;
        INSERT INTO late_mv_user_language_per_month
            (user_id, channel_id, month, total_jp_messages,
             total_non_emoji_messages)
        WITH late_videos AS (
            SELECT DISTINCT j.video_id
              FROM ingest_jobs j
              JOIN videos v USING (video_id)
              JOIN monthly_merge_state s
                ON s.observed_month = target_month AND s.status = 'merged'
             WHERE j.status = 'done'
               AND j.completed_at > s.merged_at
               AND v.end_time >= target_month
               AND v.end_time < month_end
               AND NOT EXISTS (
                   SELECT 1 FROM mv_user_activity b
                    WHERE b.video_id = j.video_id)
        )
        SELECT ud.user_id, ud.channel_id,
               DATE_TRUNC('month', ud.last_message_at), SUM(ud.jp_count),
               SUM(ud.total_message_count - ud.emoji_count)
          FROM user_data_all ud
          JOIN late_videos lv USING (video_id)
         WHERE ud.total_message_count > 0
         GROUP BY ud.user_id, ud.channel_id,
                  DATE_TRUNC('month', ud.last_message_at);
    ELSE
        RAISE EXCEPTION 'unknown late-month delta stage: %', target_stage;
    END IF;
END;
$$;

-- Aggregate base + activated delta only for overridden months. Other months
-- remain direct base rows and retain their existing index-friendly grain.
CREATE OR REPLACE VIEW mv_user_monthly_activity_live AS
SELECT b.user_id, b.channel_id, b.observed_month, b.monthly_message_count
  FROM mv_user_monthly_activity b
 WHERE NOT EXISTS (
       SELECT 1 FROM late_month_overrides o
        WHERE o.observed_month = b.observed_month::date)
UNION ALL
SELECT x.user_id, x.channel_id, x.observed_month,
       SUM(x.monthly_message_count)::bigint AS monthly_message_count
  FROM (
        SELECT b.user_id, b.channel_id, b.observed_month,
               b.monthly_message_count
          FROM mv_user_monthly_activity b
          JOIN late_month_overrides o
            ON o.observed_month = b.observed_month::date
        UNION ALL
        SELECT l.user_id, l.channel_id, l.observed_month,
               l.monthly_message_count
          FROM late_mv_user_monthly_activity l
          JOIN late_month_overrides o
            ON o.observed_month = l.observed_month::date
  ) x
 GROUP BY x.user_id, x.channel_id, x.observed_month;

CREATE OR REPLACE VIEW mv_user_activity_live AS
SELECT b.user_id, b.channel_id, b.channel_group, b.activity_month, b.video_id,
       b.last_message_at, b.message_count
  FROM mv_user_activity b
 WHERE NOT EXISTS (
       SELECT 1
         FROM late_mv_user_activity l
         JOIN late_month_overrides o
           ON o.observed_month = l.activity_month::date
        WHERE l.user_id = b.user_id AND l.channel_id = b.channel_id
          AND l.last_message_at = b.last_message_at
          AND l.video_id = b.video_id)
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
SELECT x.channel_id, x.observed_month,
       SUM(x.jp_count)::bigint AS jp_count,
       SUM(x.kr_count)::bigint AS kr_count,
       SUM(x.ru_count)::bigint AS ru_count,
       SUM(x.emoji_count)::bigint AS emoji_count,
       SUM(x.es_en_id_count)::bigint AS es_en_id_count,
       SUM(x.total_messages)::bigint AS total_messages
  FROM (
        SELECT b.channel_id, b.observed_month, b.jp_count, b.kr_count,
               b.ru_count, b.emoji_count, b.es_en_id_count, b.total_messages
          FROM chat_language_stats_mv b
          JOIN late_month_overrides o
            ON o.observed_month = b.observed_month::date
        UNION ALL
        SELECT l.channel_id, l.observed_month, l.jp_count, l.kr_count,
               l.ru_count, l.emoji_count, l.es_en_id_count, l.total_messages
          FROM late_chat_language_stats l
          JOIN late_month_overrides o
            ON o.observed_month = l.observed_month::date
  ) x
 GROUP BY x.channel_id, x.observed_month;

CREATE OR REPLACE VIEW mv_user_language_per_month_live AS
SELECT b.user_id, b.channel_id, b.month, b.total_jp_messages,
       b.total_non_emoji_messages
  FROM mv_user_language_per_month b
 WHERE NOT EXISTS (
       SELECT 1 FROM late_month_overrides o
        WHERE o.observed_month = b.month::date)
UNION ALL
SELECT x.user_id, x.channel_id, x.month,
       SUM(x.total_jp_messages)::bigint AS total_jp_messages,
       SUM(x.total_non_emoji_messages)::bigint
           AS total_non_emoji_messages
  FROM (
        SELECT b.user_id, b.channel_id, b.month, b.total_jp_messages,
               b.total_non_emoji_messages
          FROM mv_user_language_per_month b
          JOIN late_month_overrides o ON o.observed_month = b.month::date
        UNION ALL
        SELECT l.user_id, l.channel_id, l.month, l.total_jp_messages,
               l.total_non_emoji_messages
          FROM late_mv_user_language_per_month l
          JOIN late_month_overrides o ON o.observed_month = l.month::date
  ) x
 GROUP BY x.user_id, x.channel_id, x.month;

CREATE OR REPLACE VIEW v_user_active_months AS
SELECT DISTINCT user_id, channel_id, channel_group, activity_month
FROM mv_user_activity_live;
