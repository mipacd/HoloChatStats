"""Runtime enforcement for strict oldest-month-first ETL processing."""


def work_months(conn, video_id):
    """Return ``(video_month, active_month)`` as UTC dates.

    Queue contents are only hints: old deployments, retries, and manual sends
    can leave future-month messages behind. The database is authoritative.
    ``user_data_current`` is included so month N+1 cannot start while month N
    is completely ingested but still waiting for its publication merge.
    """
    with conn.cursor() as cur:
        cur.execute("""
            WITH floor AS (
                SELECT COALESCE(
                    NULLIF((SELECT value FROM service_config
                            WHERE key='backlog_floor'), '')::timestamptz,
                    '1970-01-01 00:00:00+00'::timestamptz) AS ts
            ), target AS (
                SELECT date_trunc('month', end_time AT TIME ZONE 'UTC')::date m
                FROM videos WHERE video_id=%s
            ), unfinished AS (
                SELECT MIN(date_trunc(
                           'month', v.end_time AT TIME ZONE 'UTC')::date) m
                FROM ingest_jobs j JOIN videos v USING (video_id), floor f
                WHERE v.end_time >= f.ts
                  AND j.status NOT IN ('done', 'skipped')
            ), unpublished AS (
                SELECT MIN(u.observed_month) m
                FROM user_data_current u, floor f
                WHERE u.observed_month >= date_trunc('month', f.ts)::date
                  AND NOT EXISTS (
                      SELECT 1 FROM monthly_merge_state s
                      WHERE s.observed_month=u.observed_month
                        AND s.status='merged')
            )
            SELECT target.m,
                   LEAST(unfinished.m, unpublished.m)
            FROM target CROSS JOIN unfinished CROSS JOIN unpublished
        """, (video_id,))
        row = cur.fetchone()
    conn.rollback()
    return row if row else (None, None)

