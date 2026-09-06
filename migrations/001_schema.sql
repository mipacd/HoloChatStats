-- ===========================================================================
-- Tables, columns, constraints, indexes.
--
-- Convergent by design: CREATE TABLE IF NOT EXISTS cannot fix a table that
-- already exists in the wrong shape (the exact failure mode that broke
-- ingestion on a pg_restore'd `users` with no primary key). So every table is
-- followed by explicit ALTER ... ADD COLUMN IF NOT EXISTS, and section 6 walks
-- every ON CONFLICT target in the codebase and installs any missing key.
-- ===========================================================================
DO $$
BEGIN
    CREATE EXTENSION IF NOT EXISTS vector;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'pgvector not available on this server: % -- vector-store '
                 'tables will be unusable here', SQLERRM;
END $$;
-- ---------------------------------------------------------------------------
-- 1. reference data
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS channels (
    channel_id    TEXT PRIMARY KEY,
    channel_name  TEXT NOT NULL,
    channel_group TEXT,
    -- Owned by the admin UI, NOT by channels.json. The JSON file is an import
    -- seed: it adds and renames, it never deactivates.
    active        BOOLEAN     NOT NULL DEFAULT TRUE,
    added_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
ALTER TABLE channels ADD COLUMN IF NOT EXISTS active     BOOLEAN     NOT NULL DEFAULT TRUE;
ALTER TABLE channels ADD COLUMN IF NOT EXISTS added_at   TIMESTAMPTZ NOT NULL DEFAULT NOW();
ALTER TABLE channels ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();
CREATE TABLE IF NOT EXISTS users (
    user_id  TEXT PRIMARY KEY,
    username TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS videos (
    video_id           TEXT PRIMARY KEY,
    channel_id         TEXT REFERENCES channels,
    title              TEXT NOT NULL,
    end_time           TIMESTAMPTZ NOT NULL,
    duration           INTERVAL,
    processed_at       TIMESTAMP DEFAULT NOW(),
    has_chat_log       BOOLEAN DEFAULT FALSE,
    funniest_timestamp INT
);
-- ---------------------------------------------------------------------------
-- 2. chat data: closed months in user_data, the open month in user_data_current
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS user_data (
    user_id             TEXT,
    channel_id          TEXT,
    last_message_at     TIMESTAMPTZ NOT NULL,
    video_id            TEXT,
    membership_rank     INT,
    jp_count            INT DEFAULT 0,
    kr_count            INT DEFAULT 0,
    ru_count            INT DEFAULT 0,
    emoji_count         INT DEFAULT 0,
    es_en_id_count      INT DEFAULT 0,
    total_message_count INT DEFAULT 0,
    is_gift             BOOLEAN DEFAULT FALSE,
    PRIMARY KEY (user_id, channel_id, last_message_at, video_id)
);
CREATE TABLE IF NOT EXISTS user_data_current (
    user_id             TEXT,
    channel_id          TEXT,
    last_message_at     TIMESTAMPTZ NOT NULL,
    video_id            TEXT,
    membership_rank     INT,
    jp_count            INT DEFAULT 0,
    kr_count            INT DEFAULT 0,
    ru_count            INT DEFAULT 0,
    emoji_count         INT DEFAULT 0,
    es_en_id_count      INT DEFAULT 0,
    total_message_count INT DEFAULT 0,
    is_gift             BOOLEAN DEFAULT FALSE,
    -- Month the *video* belongs to (UTC) = the merge unit. Deliberately not
    -- derived from last_message_at: a stream crossing midnight on the 1st must
    -- merge as one unit with the rest of its month.
    observed_month      DATE NOT NULL,
    PRIMARY KEY (user_id, channel_id, last_message_at, video_id)
);
CREATE TABLE IF NOT EXISTS monthly_merge_state (
    observed_month DATE PRIMARY KEY,
    status         TEXT NOT NULL DEFAULT 'open',   -- open | merged
    rows_merged    BIGINT DEFAULT 0,
    merged_at      TIMESTAMPTZ,
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS membership_data_summary (
    channel_group    TEXT,
    channel_name     TEXT,
    observed_month   DATE,
    membership_rank  INT,
    membership_count BIGINT,
    percentage_total DECIMAL(5, 2),
    updated_at       TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (channel_name, observed_month, membership_rank)
);
-- ---------------------------------------------------------------------------
-- 3. forecasting (unchanged; read by the site, not written by this pipeline)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS streaming_forecasts (
    forecast_id       SERIAL PRIMARY KEY,
    channel_id        TEXT NOT NULL REFERENCES channels(channel_id),
    forecast_month    DATE NOT NULL,
    forecasted_hours  NUMERIC(10, 2) NOT NULL,
    confidence_lower  NUMERIC(10, 2),
    confidence_upper  NUMERIC(10, 2),
    confidence_p25    NUMERIC(10, 2),
    confidence_p75    NUMERIC(10, 2),
    model_version     VARCHAR(50),
    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (channel_id, forecast_month, created_at)
);
CREATE TABLE IF NOT EXISTS forecast_model_metrics (
    metric_id     SERIAL PRIMARY KEY,
    channel_id    TEXT REFERENCES channels(channel_id),
    mae           NUMERIC(10, 4),
    rmse          NUMERIC(10, 4),
    mape          NUMERIC(10, 4),
    model_version VARCHAR(50),
    training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
-- ---------------------------------------------------------------------------
-- 4. pipeline control surface
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS service_config (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS channel_watermarks (
    channel_id         TEXT PRIMARY KEY REFERENCES channels(channel_id),
    last_scanned_at    TIMESTAMPTZ,
    last_seen_end_time TIMESTAMPTZ,
    last_error         TEXT
);
-- One row per video we intend to ingest. This table *is* the monitoring surface.
CREATE TABLE IF NOT EXISTS ingest_jobs (
    video_id            TEXT PRIMARY KEY,
    channel_id          TEXT NOT NULL,
    status              TEXT NOT NULL DEFAULT 'pending',
      -- pending | downloading | downloaded | ingesting | done | failed | skipped
    attempts            INT  NOT NULL DEFAULT 0,
    continuation        TEXT,              -- resume token, advanced only after a
                                           -- part lands in S3
    part_count          INT  NOT NULL DEFAULT 0,
    last_offset_s       DOUBLE PRECISION DEFAULT 0,
    video_duration_s    DOUBLE PRECISION,
    messages_downloaded BIGINT NOT NULL DEFAULT 0,   -- live progress counter
    lease_id            TEXT,              -- fencing token; see handlers/download.py
    reaped_count        INT  NOT NULL DEFAULT 0,
    s3_prefix           TEXT,
    message_count       BIGINT,
    skip_reason         TEXT,
    last_error          TEXT,
    enqueued_at         TIMESTAMPTZ DEFAULT NOW(),
    started_at          TIMESTAMPTZ,
    updated_at          TIMESTAMPTZ DEFAULT NOW(),
    completed_at        TIMESTAMPTZ
);
ALTER TABLE ingest_jobs ADD COLUMN IF NOT EXISTS messages_downloaded BIGINT NOT NULL DEFAULT 0;
ALTER TABLE ingest_jobs ADD COLUMN IF NOT EXISTS lease_id            TEXT;
ALTER TABLE ingest_jobs ADD COLUMN IF NOT EXISTS reaped_count        INT NOT NULL DEFAULT 0;
-- ---------------------------------------------------------------------------
-- 5. invariants
-- ---------------------------------------------------------------------------
-- An underscore-prefixed name/group is a placeholder or comment entry in
-- channels.json. Enforced here so no import path, admin call, or future handler
-- can activate one by accident.
CREATE OR REPLACE FUNCTION channels_enforce_underscore_inactive()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF LEFT(COALESCE(NEW.channel_name, ''), 1) = '_'
       OR LEFT(COALESCE(NEW.channel_group, ''), 1) = '_' THEN
        NEW.active := FALSE;
    END IF;
    RETURN NEW;
END;
$$;
DROP TRIGGER IF EXISTS trg_channels_underscore ON channels;
CREATE TRIGGER trg_channels_underscore
    BEFORE INSERT OR UPDATE ON channels
    FOR EACH ROW EXECUTE FUNCTION channels_enforce_underscore_inactive();
-- ---------------------------------------------------------------------------
-- 6. key convergence for pre-existing (restored) tables
-- ---------------------------------------------------------------------------
DROP INDEX IF EXISTS idx_user_data;

-- pg_attribute.attname is `name`, not `text`: both sides of the array
-- comparison are cast explicitly, because `name[] = text[]` has no operator.
-- indkey is an int2vector, so it is routed through its text form to get a real
-- int[]; indnkeyatts (not indnatts) so a covering index's INCLUDE columns are
-- not mistaken for key columns.
CREATE OR REPLACE FUNCTION has_unique_key(p_table text, p_cols text[])
RETURNS boolean LANGUAGE sql STABLE AS $$
    SELECT EXISTS (
        SELECT 1
        FROM pg_index i
        WHERE i.indrelid = to_regclass(p_table)
          AND (i.indisunique OR i.indisprimary)
          AND i.indpred IS NULL
          AND i.indnkeyatts = cardinality(p_cols)
          AND (
                SELECT array_agg(a.attname::text ORDER BY a.attname::text)
                FROM unnest(string_to_array(i.indkey::text, ' ')::int[])
                     WITH ORDINALITY AS k(attnum, pos)
                JOIN pg_attribute a
                  ON a.attrelid = i.indrelid AND a.attnum = k.attnum
                 AND NOT a.attisdropped
                WHERE k.pos <= i.indnkeyatts
              )
              IS NOT DISTINCT FROM
              (SELECT array_agg(c::text ORDER BY c::text) FROM unnest(p_cols) AS c)
    );
$$;
CREATE OR REPLACE FUNCTION ensure_unique_key(p_table   text,
                                             p_cols    text[],
                                             p_make_pk boolean DEFAULT true)
RETURNS text LANGUAGE plpgsql AS $$
DECLARE
    rel       regclass := p_table::regclass;
    has_pk    boolean;
    all_nn    boolean;
    col_list  text := (SELECT string_agg(quote_ident(c), ', ' ORDER BY i)
                       FROM unnest(p_cols) WITH ORDINALITY t(c, i));
    null_pred text := (SELECT string_agg(format('%I IS NULL', c), ' OR ')
                       FROM unnest(p_cols) AS c);
    reject    text := left(format('%s_key_rejects',
                                  replace(p_table, '.', '_')), 63);
    idx_name  text;
    nulls     bigint := 0;
    dups      bigint := 0;
BEGIN
    IF has_unique_key(p_table, p_cols) THEN
        RETURN format('%s (%s): ok', p_table, col_list);
    END IF;
    SELECT EXISTS (SELECT 1 FROM pg_index i
                   WHERE i.indrelid = rel AND i.indisprimary) INTO has_pk;
    -- If every key column is already NOT NULL, the null sweep is a guaranteed
    -- zero-row seq scan. Skip it.
    SELECT bool_and(a.attnotnull) INTO all_nn
    FROM pg_attribute a
    WHERE a.attrelid = rel AND a.attname = ANY (p_cols)
      AND a.attnum > 0 AND NOT a.attisdropped;
    -- Quarantine, do not destroy. On a restored production database
    -- "unreachable by ON CONFLICT" is not the same as "worthless", and a
    -- DELETE is not reviewable after the fact.
    EXECUTE format('CREATE TABLE IF NOT EXISTS %I (LIKE %s)', reject, rel);
    IF NOT COALESCE(all_nn, false) THEN
        EXECUTE format('WITH d AS (DELETE FROM %1$s WHERE %2$s RETURNING *) '
                       'INSERT INTO %3$I SELECT * FROM d',
                       rel, null_pred, reject);
        GET DIAGNOSTICS nulls = ROW_COUNT;
    END IF;
    -- One sequential scan + one sort. The previous revision self-joined the
    -- table against itself (a.ctid < b.ctid AND a.k = b.k), which the planner
    -- executes as a hash join of the table with itself: on a 40M-row user_data
    -- that is ~30 minutes and tens of GB of temp files. It is the direct cause
    -- of the 300 s Lambda timeout.
    --
    -- ORDER BY ctid DESC keeps the physically newest row of each group: last
    -- write wins, exactly what ON CONFLICT DO UPDATE would have produced.
    EXECUTE format($f$
        WITH losers AS MATERIALIZED (
            SELECT ctid FROM (
                SELECT ctid,
                       row_number() OVER (PARTITION BY %2$s
                                          ORDER BY ctid DESC) AS rn
                FROM %1$s
            ) s WHERE s.rn > 1
        ), d AS (
            DELETE FROM %1$s t USING losers l
             WHERE t.ctid = l.ctid
            RETURNING t.*
        )
        INSERT INTO %3$I SELECT * FROM d
    $f$, rel, col_list, reject);
    GET DIAGNOSTICS dups = ROW_COUNT;
    IF nulls = 0 AND dups = 0 THEN
        EXECUTE format('DROP TABLE %I', reject);
    ELSE
        RAISE NOTICE '% : quarantined % null-key and % duplicate row(s) into %',
                     p_table, nulls, dups, reject;
    END IF;
    IF p_make_pk AND NOT has_pk THEN
        EXECUTE format('ALTER TABLE %s ADD PRIMARY KEY (%s)', rel, col_list);
        RETURN format('%s (%s): PRIMARY KEY added; %s null-key + %s duplicate '
                      'row(s) quarantined', p_table, col_list, nulls, dups);
    END IF;
    idx_name := left(format('uq_%s_%s', replace(p_table, '.', '_'),
                            array_to_string(p_cols, '_')), 63);
    EXECUTE format('CREATE UNIQUE INDEX %I ON %s (%s)', idx_name, rel, col_list);
    RETURN format('%s (%s): UNIQUE INDEX %s added; %s null-key + %s duplicate '
                  'row(s) quarantined', p_table, col_list, idx_name, nulls, dups);
END;
$$;
DO $$
DECLARE
    spec  record;
    fails text[] := '{}';
BEGIN
    FOR spec IN
        SELECT * FROM (VALUES
            ('channels',                ARRAY['channel_id']),
            ('users',                   ARRAY['user_id']),
            ('videos',                  ARRAY['video_id']),
            ('ingest_jobs',             ARRAY['video_id']),
            ('channel_watermarks',      ARRAY['channel_id']),
            ('service_config',          ARRAY['key']),
            ('schema_migrations',       ARRAY['filename']),
            ('monthly_merge_state',     ARRAY['observed_month']),
            ('membership_data_summary', ARRAY['channel_name','observed_month',
                                              'membership_rank']),
            ('user_data',               ARRAY['user_id','channel_id',
                                              'last_message_at','video_id']),
            ('user_data_current',       ARRAY['user_id','channel_id',
                                              'last_message_at','video_id'])
        ) AS t(tbl, cols)
    LOOP
        BEGIN
            IF to_regclass(spec.tbl) IS NULL THEN
                RAISE NOTICE '% : table absent, skipped', spec.tbl;
                CONTINUE;
            END IF;
            RAISE NOTICE '%', ensure_unique_key(spec.tbl, spec.cols);
        EXCEPTION WHEN others THEN
            fails := fails || format('%s: %s', spec.tbl, SQLERRM);
        END;
    END LOOP;
    IF array_length(fails, 1) > 0 THEN
        RAISE EXCEPTION 'key repair failed: %', array_to_string(fails, ' | ');
    END IF;
END $$;

-- ---------------------------------------------------------------------------
-- 7. indexes
-- ---------------------------------------------------------------------------
-- Obsolete: a UNIQUE index on (user_id, channel_id) contradicts user_data's own
-- primary key -- it permitted only one row per user per channel, so the second
-- video a user ever chatted in failed to insert.
DROP INDEX IF EXISTS idx_user_data;
CREATE INDEX IF NOT EXISTS idx_user_data_user_id_last_message_at ON user_data (user_id, last_message_at);
CREATE INDEX IF NOT EXISTS idx_user_data_user_id    ON user_data (user_id);
CREATE INDEX IF NOT EXISTS idx_user_data_channel_id ON user_data (channel_id);
CREATE INDEX IF NOT EXISTS idx_user_data_video_id   ON user_data (video_id);
CREATE INDEX IF NOT EXISTS idx_user_data_is_gift    ON user_data (is_gift);
CREATE INDEX IF NOT EXISTS idx_user_data_total_message_count ON user_data(total_message_count);
CREATE INDEX IF NOT EXISTS idx_user_data_recommendations
    ON user_data (last_message_at, channel_id, user_id, total_message_count)
    WHERE total_message_count > 0;
CREATE INDEX IF NOT EXISTS idx_udc_month   ON user_data_current (observed_month);
CREATE INDEX IF NOT EXISTS idx_udc_video   ON user_data_current (video_id);
CREATE INDEX IF NOT EXISTS idx_udc_channel ON user_data_current (channel_id);
CREATE INDEX IF NOT EXISTS idx_channels_channel_id   ON channels (channel_id);
CREATE INDEX IF NOT EXISTS idx_channels_channel_name ON channels (channel_name);
CREATE INDEX IF NOT EXISTS idx_channels_active       ON channels (active);
CREATE INDEX IF NOT EXISTS idx_videos_end_time         ON videos (end_time);
CREATE INDEX IF NOT EXISTS idx_videos_channel_end_time ON videos (channel_id, end_time);
CREATE INDEX IF NOT EXISTS idx_membership_summary_group_month
    ON membership_data_summary (channel_group, observed_month);
CREATE INDEX IF NOT EXISTS idx_ingest_jobs_status     ON ingest_jobs (status, updated_at);
CREATE INDEX IF NOT EXISTS idx_ingest_jobs_updated_at ON ingest_jobs (updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_ingest_jobs_channel    ON ingest_jobs (channel_id, status);
CREATE INDEX IF NOT EXISTS idx_ingest_jobs_started    ON ingest_jobs (started_at NULLS FIRST, enqueued_at);
CREATE INDEX IF NOT EXISTS idx_ingest_jobs_stalled    ON ingest_jobs (status, updated_at)
    WHERE status IN ('downloading', 'ingesting', 'downloaded');

-- ---------------------------------------------------------------------------
-- 8. videos.end_time NOT NULL   (unchanged DO $$ block)
-- ---------------------------------------------------------------------------

-- end_time drives month bucketing and the discovery watermark, so it must not
-- be NULL. Tighten only if the data already allows it; otherwise report and let
-- {"action":"repair_end_times"} resolve the rows from YouTube.
DO $$
DECLARE n bigint;
BEGIN
    SELECT COUNT(*) INTO n FROM videos WHERE end_time IS NULL;
    IF n = 0 THEN
        BEGIN
            ALTER TABLE videos ALTER COLUMN end_time SET NOT NULL;
        EXCEPTION WHEN others THEN
            RAISE NOTICE 'videos.end_time: could not set NOT NULL (%)', SQLERRM;
        END;
    ELSE
        RAISE NOTICE 'videos.end_time: % NULL row(s) -- run '
                     '{"action":"repair_end_times"}, then re-apply this file',
                     n;
    END IF;
END $$;

CREATE MATERIALIZED VIEW IF NOT EXISTS mv_global_totals AS
SELECT 1 AS id,
       sum(total_message_count)::bigint AS total_messages,
       count(*)::bigint                 AS total_users
FROM user_data;
-- required for REFRESH MATERIALIZED VIEW CONCURRENTLY
CREATE UNIQUE INDEX IF NOT EXISTS mv_global_totals_pkey
    ON mv_global_totals (id);

