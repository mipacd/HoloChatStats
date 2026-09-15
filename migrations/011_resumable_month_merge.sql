-- Move a closed month in bounded transactions.  The public publication
-- boundary remains monthly_merge_state.status = 'merged'; while status is
-- 'merging', user_data_all continues to expose the logical union of both
-- physical tables to internal maintenance jobs.
CREATE OR REPLACE FUNCTION merge_month_batch(
    target_month DATE,
    requested_batch_size INTEGER DEFAULT 10000
)
RETURNS BIGINT LANGUAGE plpgsql AS $$
DECLARE
    moved BIGINT;
    batch_size INTEGER := LEAST(GREATEST(requested_batch_size, 1), 50000);
BEGIN
    WITH picked AS (
        SELECT ctid
        FROM user_data_current
        WHERE observed_month = target_month
        LIMIT batch_size
        FOR UPDATE SKIP LOCKED
    ), src AS (
        DELETE FROM user_data_current current_row
        USING picked
        WHERE current_row.ctid = picked.ctid
        RETURNING current_row.user_id, current_row.channel_id,
                  current_row.last_message_at, current_row.video_id,
                  current_row.membership_rank, current_row.jp_count,
                  current_row.kr_count, current_row.ru_count,
                  current_row.emoji_count, current_row.es_en_id_count,
                  current_row.total_message_count, current_row.is_gift
    ), ins AS (
        INSERT INTO user_data (
            user_id, channel_id, last_message_at, video_id, membership_rank,
            jp_count, kr_count, ru_count, emoji_count, es_en_id_count,
            total_message_count, is_gift
        )
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
    RETURN moved;
END;
$$;

COMMENT ON FUNCTION merge_month_batch(DATE, INTEGER) IS
    'Idempotently moves one bounded batch from monthly staging to user_data';

COMMENT ON COLUMN monthly_merge_state.status IS
    'open | merging | merged; only merged is publicly visible';
