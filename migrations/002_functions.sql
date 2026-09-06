-- ===========================================================================
-- Business logic: the month-close merge and the membership summary rebuild.
-- Both read user_data_all, so they are agnostic to which table a month is in.
-- ===========================================================================
-- Move a month's rows from the staging table into user_data.
CREATE OR REPLACE FUNCTION merge_month_into_user_data(target_month DATE)
RETURNS BIGINT LANGUAGE plpgsql AS $$
DECLARE
    moved BIGINT;
BEGIN
    -- Single statement => single snapshot => no window in which a row is
    -- visible in both tables (or in neither) through user_data_all.
    WITH src AS (
        DELETE FROM user_data_current
        WHERE observed_month = target_month
        RETURNING user_id, channel_id, last_message_at, video_id,
                  membership_rank, jp_count, kr_count, ru_count, emoji_count,
                  es_en_id_count, total_message_count, is_gift
    ), ins AS (
        INSERT INTO user_data (user_id, channel_id, last_message_at, video_id,
            membership_rank, jp_count, kr_count, ru_count, emoji_count,
            es_en_id_count, total_message_count, is_gift)
        SELECT * FROM src
        ON CONFLICT (user_id, channel_id, last_message_at, video_id) DO UPDATE
        SET membership_rank = COALESCE(EXCLUDED.membership_rank,
                                       user_data.membership_rank),
            jp_count            = EXCLUDED.jp_count,
            kr_count            = EXCLUDED.kr_count,
            ru_count            = EXCLUDED.ru_count,
            emoji_count         = EXCLUDED.emoji_count,
            es_en_id_count      = EXCLUDED.es_en_id_count,
            total_message_count = EXCLUDED.total_message_count,
            is_gift             = EXCLUDED.is_gift
        RETURNING 1
    )
    SELECT COUNT(*) INTO moved FROM ins;
    INSERT INTO monthly_merge_state (observed_month, status, rows_merged,
                                     merged_at, updated_at)
    VALUES (target_month, 'merged', moved, NOW(), NOW())
    ON CONFLICT (observed_month) DO UPDATE
      SET status      = 'merged',
          rows_merged = COALESCE(monthly_merge_state.rows_merged, 0)
                        + EXCLUDED.rows_merged,
          merged_at   = NOW(),
          updated_at  = NOW();
    RETURN moved;
END;
$$;
-- Rebuild membership_data_summary for one month.
CREATE OR REPLACE PROCEDURE refresh_membership_data_for_month(target_month DATE)
LANGUAGE plpgsql AS $$
BEGIN
    DELETE FROM membership_data_summary WHERE observed_month = target_month;
    INSERT INTO membership_data_summary
    WITH ranked_memberships AS (
        SELECT
            m.user_id,
            m.channel_id,
            DATE_TRUNC('month', m.last_message_at)::DATE AS observed_month,
            m.membership_rank,
            m.is_gift,
            m.last_message_at,
            ROW_NUMBER() OVER (
                PARTITION BY m.user_id, m.channel_id
                ORDER BY
                    CASE WHEN m.membership_rank >= 0 THEN 0
                         WHEN m.membership_rank = -2 THEN 1
                         ELSE 2 END ASC,
                    m.last_message_at DESC
            ) AS row_num
        FROM user_data_all m
        -- Range-scan one month only; do not scan the whole table.
        WHERE m.last_message_at >= target_month
          AND m.last_message_at <  target_month + INTERVAL '1 month'
    ),
    latest_memberships AS (
        SELECT user_id, channel_id, observed_month, membership_rank
        FROM ranked_memberships WHERE row_num = 1
    )
    SELECT
        c.channel_group,
        c.channel_name,
        lm.observed_month,
        lm.membership_rank,
        COUNT(lm.user_id) AS membership_count,
        ROUND(COUNT(lm.user_id)::DECIMAL
              / NULLIF(SUM(COUNT(*)) OVER (PARTITION BY c.channel_name,
                                                        lm.observed_month), 0)
              * 100, 2) AS percentage_total,
        NOW()
    FROM latest_memberships lm
    JOIN channels c ON lm.channel_id = c.channel_id
    GROUP BY c.channel_group, c.channel_name, lm.observed_month,
             lm.membership_rank;
END;
$$;