from datetime import datetime, timezone, timedelta
from common.db import get_conn
from common.logging_utils import get_logger
from common.metrics import emit, COUNT, SECONDS
from common.cache_invalidation import invalidate_finalized_month_caches
import time
import psycopg2
log = get_logger("refresh")

REPUBLISH_STAGES = (
    "monthly activity",
    "user activity",
    "language totals",
    "user language totals",
)


def handler(event, context):
    event = event or {}
    if event.get("publish_months"):
        return _resume_republish(event["publish_months"])
    return _scheduled_refresh()


def _resume_republish(values):
    """Advance one durable late-data publication stage.

    Each derived-data stage rebuilds only the selected month into an overlay.
    The reader-facing SQL views substitute that complete month for the stale
    full-history materialized-view rows once every stage has succeeded.
    """
    if not isinstance(values, list) or not values:
        raise ValueError("publish_months must contain at least one month")
    month = datetime.fromisoformat(str(values[0])).date().replace(day=1)
    conn = get_conn()
    conn.autocommit = True
    t0 = time.time()
    with conn.cursor() as cur:
        stage_key = f"late_republish_stage:{month}"
        marker_key = f"late_data_month:{month}"
        cur.execute("""SELECT marker.updated_at, stage.value, stage.updated_at,
                              EXISTS (SELECT 1 FROM monthly_merge_state s
                                       WHERE s.observed_month=%s
                                         AND s.status='merged')
                         FROM service_config marker
                         JOIN service_config stage ON stage.key=%s
                        WHERE marker.key=%s AND marker.value='pending'""",
                    (month, stage_key, marker_key))
        row = cur.fetchone()
        if not row:
            conn.close()
            return {"month": str(month), "complete": False,
                    "reason": "no queued late-data publication"}
        marker_updated_at, stage_value, workflow_started_at, published = row
        if not published:
            conn.close()
            return {"month": str(month), "complete": False,
                    "reason": "month is not currently published"}
        try:
            stage = int(stage_value)
        except (TypeError, ValueError):
            stage = 0

        if stage < len(REPUBLISH_STAGES):
            phase = REPUBLISH_STAGES[stage]
            log.info("late-data overlay stage started",
                     extra={"month": str(month), "stage": stage,
                            "phase": phase})
            if stage == 0:
                # Hide a prior overlay generation until all replacement
                # datasets have been rebuilt. Readers see the consistent base
                # generation during the refresh, never a half-updated month.
                cur.execute("DELETE FROM late_month_overrides "
                            "WHERE observed_month=%s", (month,))
            cur.execute("CALL refresh_late_month_delta(%s::date, %s)",
                        (month, stage))
            cur.execute("UPDATE service_config SET value=%s WHERE key=%s",
                        (str(stage + 1), stage_key))
            log.info("late-data overlay stage completed",
                     extra={"month": str(month), "stage": stage,
                            "phase": phase})
            emit({"RefreshSeconds": (time.time() - t0, SECONDS),
                  "MonthsRefreshed": (0, COUNT)})
            conn.close()
            return {"month": str(month), "complete": False,
                    "stage": stage + 1, "stages": len(REPUBLISH_STAGES) + 2,
                    "phase": phase}

        if stage == len(REPUBLISH_STAGES):
            cur.execute("CALL refresh_membership_data_for_month(%s::date)",
                        (month,))
            cur.execute("UPDATE service_config SET value=%s WHERE key=%s",
                        (str(stage + 1), stage_key))
            log.info("late-data membership summary refreshed",
                     extra={"month": str(month)})
            emit({"RefreshSeconds": (time.time() - t0, SECONDS),
                  "MonthsRefreshed": (0, COUNT)})
            conn.close()
            return {"month": str(month), "complete": False,
                    "stage": stage + 1, "stages": len(REPUBLISH_STAGES) + 2,
                    "phase": "membership"}

        # Make all four complete month overlays visible together.  If cache
        # invalidation fails, this idempotent upsert is harmless on retry and
        # the durable stage marker remains available to the reaper.
        cur.execute("""INSERT INTO late_month_overrides
                           (observed_month, published_at)
                       VALUES (%s, NOW())
                       ON CONFLICT (observed_month) DO UPDATE
                         SET published_at=EXCLUDED.published_at""", (month,))
        removed = invalidate_finalized_month_caches(finalized_month=month)
        # A log ingested after this workflow began updates the marker. Do not
        # consume that newer work; leave it pending for another explicit pass.
        cur.execute("""DELETE FROM service_config
                        WHERE key=%s AND value='pending' AND updated_at <= %s""",
                    (marker_key, workflow_started_at))
        marker_cleared = cur.rowcount == 1
        if marker_cleared:
            cur.execute("""INSERT INTO service_config (key, value, updated_at)
                           VALUES (%s, %s, NOW())
                           ON CONFLICT (key) DO UPDATE
                             SET value=EXCLUDED.value, updated_at=NOW()""",
                        (f"late_data_published:{month}",
                         datetime.now(timezone.utc).isoformat()))
        cur.execute("DELETE FROM service_config WHERE key=%s", (stage_key,))
        log.info("late finalized month published",
                 extra={"month": str(month), "cache_keys_removed": removed,
                        "marker_cleared": marker_cleared,
                        "newer_late_data": marker_updated_at > workflow_started_at})
    emit({"RefreshSeconds": (time.time() - t0, SECONDS),
          "MonthsRefreshed": (1 if marker_cleared else 0, COUNT)})
    conn.close()
    return {"month": str(month), "complete": marker_cleared,
            "cache_keys_removed": removed,
            "newer_late_data_pending": not marker_cleared}


def _scheduled_refresh():
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
        for mv in ("mv_user_monthly_activity", "mv_user_activity",
                   "chat_language_stats_mv", "mv_user_language_per_month"):
            _refresh(cur, mv, log)
            log.info("refreshed", extra={"view": mv})
        for m in sorted(months):
            cur.execute("CALL refresh_membership_data_for_month(%s::date)", (m,))
            log.info("membership summary refreshed", extra={"month": str(m)})
    conn.autocommit = False
    conn.close()
    emit({"RefreshSeconds": (time.time() - t0, SECONDS),
          "MonthsRefreshed": (len(months), COUNT)})
    return {"months": [str(m) for m in sorted(months)],
            "late_months": [],
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
