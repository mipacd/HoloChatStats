from datetime import datetime, timezone, timedelta
from common.db import get_conn
from common.logging_utils import get_logger
from common.metrics import emit, COUNT, SECONDS
from common.cache_invalidation import invalidate_finalized_month_caches
import time
import psycopg2
log = get_logger("refresh")
def handler(event, context):
    event = event or {}
    conn = get_conn()
    conn.autocommit = True           # REFRESH CONCURRENTLY cannot run in a txn block
    t0 = time.time()
    # Only refresh months that actually received data since the last refresh.
    with conn.cursor() as cur:
        cur.execute("""
            SELECT DISTINCT date_trunc('month', v.end_time)::date
            FROM ingest_jobs j JOIN videos v USING (video_id)
            WHERE j.status = 'done'
              AND j.completed_at > NOW() - INTERVAL '2 days'
              AND NOT EXISTS (
                  SELECT 1 FROM monthly_merge_state s
                  WHERE s.observed_month =
                        date_trunc('month', v.end_time)::date
                    AND s.status = 'merged')""")
        months = {r[0] for r in cur.fetchall()}
        cur.execute("""SELECT key, split_part(key, ':', 2)::date, updated_at
                       FROM service_config
                       WHERE key LIKE 'late_data_month:%' AND value='pending'""")
        late_pending = {r[1]: (r[0], r[2]) for r in cur.fetchall()}
        late = {}
        if event.get("publish_months"):
            requested = {datetime.fromisoformat(str(m)).date().replace(day=1)
                         for m in event["publish_months"]}
            late = {m: marker for m, marker in late_pending.items()
                    if m in requested}
        months.update(late)
        for mv in ("mv_user_monthly_activity", "mv_user_activity",
                   "chat_language_stats_mv", "mv_user_language_per_month"):
            _refresh(cur, mv, log)
            log.info("refreshed", extra={"view": mv})
        for m in sorted(months):
            cur.execute("CALL refresh_membership_data_for_month(%s::date)", (m,))
            log.info("membership summary refreshed", extra={"month": str(m)})
        # Cache entries have no TTL.  A late finalized-month correction must
        # therefore be invalidated explicitly, after all derived SQL data is
        # current.  Deleting the marker last makes Redis failures retryable.
        for m, (key, marker_updated_at) in sorted(late.items()):
            removed = invalidate_finalized_month_caches(finalized_month=m)
            # If another late ingest touched the marker while views were being
            # rebuilt, leave it pending for the next run rather than losing
            # that correction in a refresh/delete race.
            cur.execute("""DELETE FROM service_config
                           WHERE key=%s AND value='pending' AND updated_at=%s""",
                        (key, marker_updated_at))
            marker_cleared = cur.rowcount == 1
            if marker_cleared:
                cur.execute("""INSERT INTO service_config (key, value, updated_at)
                               VALUES (%s, %s, NOW())
                               ON CONFLICT (key) DO UPDATE
                                 SET value=EXCLUDED.value, updated_at=NOW()""",
                            (f"late_data_published:{m}",
                             datetime.now(timezone.utc).isoformat()))
            log.info("late finalized month published",
                     extra={"month": str(m), "cache_keys_removed": removed,
                            "marker_cleared": marker_cleared})
    conn.autocommit = False
    emit({"RefreshSeconds": (time.time() - t0, SECONDS),
          "MonthsRefreshed": (len(months), COUNT)})
    return {"months": [str(m) for m in sorted(months)],
            "late_months": [str(m) for m in sorted(late)],
            "late_months_pending": [str(m) for m in sorted(late_pending)]}

def _refresh(cur, mv, log):
    """CONCURRENTLY needs a unique index and a populated MV. If either is
    missing, fall back to a blocking refresh rather than failing the run --
    and never let that requirement drive the view's definition."""
    try:
        cur.execute("SELECT relispopulated FROM pg_class WHERE oid = to_regclass(%s)", (mv,))
        populated = cur.fetchone()[0]
        cur.execute(f'REFRESH MATERIALIZED VIEW {"CONCURRENTLY " if populated else ""}"{mv}"')
        log.info("refreshed", extra={"view": mv, "mode": "concurrent"})
    except psycopg2.Error as e:
        log.warning("concurrent refresh unavailable; refreshing with lock",
                    extra={"view": mv, "error": str(e)[:200]})
        cur.execute(f"REFRESH MATERIALIZED VIEW {mv}")
        log.info("refreshed", extra={"view": mv, "mode": "blocking"})
