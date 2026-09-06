from datetime import datetime, timezone, timedelta
from common.db import get_conn
from common.logging_utils import get_logger
from common.metrics import emit, COUNT, SECONDS
import time
import psycopg2
log = get_logger("refresh")
def handler(event, context):
    conn = get_conn()
    conn.autocommit = True           # REFRESH CONCURRENTLY cannot run in a txn block
    t0 = time.time()
    # Only refresh months that actually received data since the last refresh.
    with conn.cursor() as cur:
        cur.execute("""
            SELECT DISTINCT date_trunc('month', v.end_time)::date
            FROM ingest_jobs j JOIN videos v USING (video_id)
            WHERE j.status = 'done'
              AND j.completed_at > NOW() - INTERVAL '2 days'""")
        months = [r[0] for r in cur.fetchall()]
        for mv in ("mv_user_monthly_activity", "mv_user_activity",
                   "chat_language_stats_mv", "mv_user_language_per_month"):
            _refresh(cur, mv, log)
            log.info("refreshed", extra={"view": mv})
        for m in months:
            cur.execute("CALL refresh_membership_data_for_month(%s::date)", (m,))
            log.info("membership summary refreshed", extra={"month": str(m)})
    conn.autocommit = False
    emit({"RefreshSeconds": (time.time() - t0, SECONDS),
          "MonthsRefreshed": (len(months), COUNT)})
    return {"months": [str(m) for m in months]}

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