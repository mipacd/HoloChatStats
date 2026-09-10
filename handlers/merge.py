"""
Month-close merge.
handlers/ingest.py writes the in-progress month into `user_data_current`.
This lambda moves a month's rows into `user_data` once the month is genuinely
finished:
  1. the month is over, plus service_config.merge_grace_hours;
  2. every video whose end_time falls in the month has a terminal job
     (done/failed/skipped) -- nothing pending/downloading/downloaded/ingesting;
     failed jobs remain visible for an explicit operator retry but do not hold
     publication indefinitely;
  3. every active channel has been scanned at least once *after* the month
     ended, including channels for which discovery found no videos. This is
     the publication barrier that proves the month is complete globally.
Readers are unaffected either way: `user_data_all` unions both tables and every
materialized view reads the union, so the merge is pure data movement.
Manual use:
    {"dry_run": true}                           -- report, change nothing
    {"months": ["2025-05-01"], "force": true}   -- merge regardless of gates
"""
import json
import os
import time
from datetime import date, datetime, timedelta, timezone
from common.config import settings
from common.db import get_conn
from common.logging_utils import get_logger
from common.metrics import emit, COUNT, SECONDS
from common.cache_invalidation import invalidate_finalized_month_caches
from common.aws import client
log = get_logger("merge")
def handler(event, context):
    event = event or {}
    t0 = time.time()
    force = bool(event.get("force"))
    dry_run = bool(event.get("dry_run"))
    cfg = settings(force=True)
    grace = int(cfg.get("merge_grace_hours", 24))
    conn = get_conn()
    months = ([_as_month(m) for m in event["months"]] if event.get("months")
              else _staged_months(conn))
    merged, skipped, rows_total = [], [], 0
    for month in months:
        state = _month_state(conn, month, grace)
        unscanned_ids = state.pop("_unscanned_channel_ids")
        if not state["complete"] and not force:
            if (state["month_over"] and not state["jobs_in_flight"]
                    and unscanned_ids and not dry_run
                    and str(cfg.get("paused", "false")).lower() != "true"):
                state["barrier_scan"] = _request_barrier_scan(
                    conn, month, unscanned_ids)
            log.info("month not ready to merge",
                     extra={"month": str(month), **state})
            skipped.append({"month": str(month), **state})
            continue
        if dry_run:
            merged.append({"month": str(month), "dry_run": True, **state})
            continue
        rows = _merge_month(conn, month)
        rows_total += rows
        merged.append({"month": str(month), "rows_merged": rows, **state})
        log.info("month merged into user_data",
                 extra={"month": str(month), "rows_merged": rows,
                        "forced": force})
    cache_invalidation = ({"needed": False, "skipped": "dry_run"}
                          if dry_run else _sync_cache_invalidation(conn))
    emit({"MonthsMerged": (len([m for m in merged if not m.get("dry_run")]), COUNT),
          "MonthsPending": (len(skipped), COUNT),
          "RowsMerged": (rows_total, COUNT),
          "CacheKeysInvalidated": (cache_invalidation.get("keys_removed", 0), COUNT),
          "MergeSeconds": (time.time() - t0, SECONDS)})
    return {"merged": merged, "skipped": skipped, "rows_merged": rows_total,
            "cache_invalidation": cache_invalidation}
# --------------------------------------------------------------------------- #
def _as_month(value):
    if isinstance(value, date):
        return value.replace(day=1)
    parts = str(value).split("-")
    return date(int(parts[0]), int(parts[1]), 1)
def _bounds(month):
    """UTC [start, end) for the month. Computed in Python so every month
    predicate is a plain range scan on videos.end_time (indexed) and is not at
    the mercy of the session TimeZone."""
    start = datetime(month.year, month.month, 1, tzinfo=timezone.utc)
    end = datetime(month.year + month.month // 12,
                   month.month % 12 + 1, 1, tzinfo=timezone.utc)
    return start, end
def _staged_months(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT observed_month FROM user_data_current "
                    "ORDER BY 1")
        months = [r[0] for r in cur.fetchall()]
    conn.rollback()
    return months
def _month_state(conn, month, grace_hours):
    start, end = _bounds(month)
    month_over = datetime.now(timezone.utc) >= end + timedelta(hours=grace_hours)
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) FILTER (WHERE j.status = 'failed'),
                   COUNT(*) FILTER (WHERE j.status <> 'failed')
            FROM ingest_jobs j JOIN videos v USING (video_id)
            WHERE v.end_time >= %s AND v.end_time < %s
              AND j.status NOT IN ('done', 'skipped')
        """, (start, end))
        failed, in_flight = cur.fetchone()
        cur.execute("""
            SELECT c.channel_id FROM channels c
            LEFT JOIN channel_watermarks w USING (channel_id)
            WHERE c.active
              AND (w.last_scanned_at IS NULL OR w.last_scanned_at < %s)
            ORDER BY c.channel_id
        """, (end,))
        unscanned_ids = [r[0] for r in cur.fetchall()]
        unscanned = len(unscanned_ids)
        cur.execute("SELECT COUNT(*) FROM user_data_current "
                    "WHERE observed_month = %s", (month,))
        staged = cur.fetchone()[0]
    conn.rollback()
    reasons = []
    if not month_over:
        reasons.append("month still open (or inside merge_grace_hours)")
    if in_flight:
        reasons.append(f"{in_flight} job(s) still in flight")
    if unscanned:
        reasons.append(f"{unscanned} channel(s) not rescanned since month end")
    return {"complete": not reasons, "month_over": month_over,
            "jobs_in_flight": in_flight, "jobs_failed": failed,
            "channels_not_rescanned": unscanned, "staged_rows": staged,
            "blockers": reasons,
            "_unscanned_channel_ids": unscanned_ids}

def _request_barrier_scan(conn, month, channel_ids):
    """Kick the missing post-boundary scans without normal backlog pressure.

    The durable throttle prevents repeated merge/dispatcher invocations from
    filling SQS with duplicate channel scans while the first request runs.
    Targeted discovery scans every supplied channel and remains resumable, so
    a never-scanned restored channel still walks back through July safely.
    """
    key = f"merge_barrier_scan:{month}"
    with conn.cursor() as cur:
        cur.execute("""INSERT INTO service_config (key, value, updated_at)
                       VALUES (%s, 'requested', NOW())
                       ON CONFLICT (key) DO UPDATE
                         SET value='requested', updated_at=NOW()
                       WHERE service_config.updated_at <
                             NOW() - INTERVAL '30 minutes'
                       RETURNING updated_at""", (key,))
        should_invoke = cur.fetchone() is not None
    conn.commit()
    if not should_invoke:
        return {"requested": 0, "throttled": True}
    try:
        client("lambda").invoke(
            FunctionName=f"{os.environ.get('APP_NAME', 'chat-ingest')}-discover",
            InvocationType="Event",
            Payload=json.dumps({"force": True, "channels": channel_ids}).encode())
    except Exception:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM service_config WHERE key=%s", (key,))
        conn.commit()
        raise
    log.info("requested missing month-barrier channel scans",
             extra={"month": str(month), "channels": len(channel_ids)})
    return {"requested": len(channel_ids), "throttled": False}
def _merge_month(conn, month):
    with conn.cursor() as cur:
        cur.execute("SELECT merge_month_into_user_data(%s::date)", (month,))
        rows = cur.fetchone()[0]
        # Summary table is keyed on last_message_at's month; recompute both the
        # merged month and the next one (streams that crossed the boundary).
        cur.execute("CALL refresh_membership_data_for_month(%s::date)", (month,))
        cur.execute("DELETE FROM service_config WHERE key=%s",
                    (f"merge_barrier_scan:{month}",))
    conn.commit()
    return rows

def _sync_cache_invalidation(conn):
    """Invalidate once per newest published month, retrying until successful."""
    with conn.cursor() as cur:
        cur.execute("""SELECT MAX(observed_month) FROM monthly_merge_state
                       WHERE status = 'merged'""")
        latest = cur.fetchone()[0]
        cur.execute("SELECT value FROM service_config WHERE key = %s",
                    ("cache_finalized_month",))
        row = cur.fetchone()
    conn.rollback()
    recorded = _as_month(row[0]) if row and row[0] else None
    if latest is None or (recorded is not None and recorded >= latest):
        return {"needed": False, "month": str(latest) if latest else None,
                "keys_removed": 0}
    removed = invalidate_finalized_month_caches(finalized_month=latest)
    with conn.cursor() as cur:
        cur.execute("""INSERT INTO service_config (key, value, updated_at)
                       VALUES (%s, %s, NOW())
                       ON CONFLICT (key) DO UPDATE
                         SET value = EXCLUDED.value, updated_at = NOW()""",
                    ("cache_finalized_month", str(latest)))
    conn.commit()
    log.info("aggregate web caches invalidated after month publication",
             extra={"month": str(latest), "keys_removed": removed})
    return {"needed": True, "month": str(latest), "keys_removed": removed}
