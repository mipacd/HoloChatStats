"""
Admin surface for the chat-ingestion pipeline.
Routes (served to the admin gateway / API Gateway):
    GET  /                -> HTML dashboard ("HoloChatStats Admin Page")
    GET  /api/status      -> JSON snapshot (jobs, queues, channels, run state)
    POST /api/control     -> pipeline, per-job retry, and month publish actions
    GET/POST /api/news    -> edit the homepage news.txt stored in config S3
"stop" does three things, belt-and-braces, because an emulator may not
implement all of them:
    1. service_config.paused = true      (every worker checks this)
    2. disables the SQS event source mappings (no new invocations)
    3. disables the EventBridge discovery schedule (no new fan-out)
"start" reverses all three.
Nothing is destructive: in-flight SQS messages are re-queued with a delay by
the workers themselves (see common/control.py), so a stop/start cycle loses no
work.
"""
import json
import os
import re
from datetime import datetime, timedelta, timezone
import redis
from psycopg2.extras import execute_values
from common.aws import client
from common.config import settings
from common.db import get_conn
from common.logging_utils import get_logger
from common.metrics import emit, COUNT
from common.channels import cancel_channel_jobs

log = get_logger("admin")
APP = os.environ.get("APP_NAME", "chat-ingest")
DISCOVER_RULE = os.environ.get("DISCOVER_RULE", f"{APP}-discover-schedule")
MANAGED_ESMS = ("scan-q", "download-q", "download-retry-q", "ingest-q")
REFRESH_SECONDS = int(os.environ.get("ADMIN_REFRESH_SECONDS", "5"))
CHANNEL_ID_RE = re.compile(r"^UC[A-Za-z0-9_-]{22}$")
NEWS_KEY = "news.txt"
MAX_NEWS_BYTES = 20_000
# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #
def handler(event, context):
    method, path = _route(event)
    try:
        if method in ("GET", "HEAD") and path == "/api/status":
            return _json(200, snapshot())
        if method == "POST" and path == "/api/control":
            body = _body(event)
            return _json(200, control(body.get("action"), body))
        if method in ("GET", "HEAD") and not path.startswith("/api/"):
            return _page()            # any other GET is the dashboard
        if method == "POST" and path == "/api/channels":
            return _json(200, channels_op(_body(event)))
        if method in ("GET", "HEAD") and path == "/api/news":
            return _json(200, get_news())
        if method == "POST" and path == "/api/news":
            return _json(200, save_news(_body(event)))
        return _json(404, {"error": "not found", "path": path, "method": method})
    except Exception as e:
        log.exception("admin request failed", extra={"path": path})
        return _json(500, {"error": f"{type(e).__name__}: {e}"})

def channels_op(body):
    op = (body or {}).get("op")
    try:
        if op == "add":
            return _add_channel(body)
        if op == "update":
            return _update_channel(body)
        if op == "remove":
            return _remove_channel(body)
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    return {"ok": False, "error": f"unknown op {op!r}"}

def get_news():
    try:
        raw = client("s3").get_object(
            Bucket=os.environ["CONFIG_BUCKET"], Key=NEWS_KEY)["Body"].read()
        return {"ok": True, "text": raw.decode("utf-8")}
    except Exception as exc:
        # Missing on an upgrade is harmless; the first save creates it.
        log.warning("news object unavailable", extra={"error": str(exc)[:160]})
        return {"ok": True, "text": ""}

def save_news(body):
    text = str((body or {}).get("text") or "").replace("\r\n", "\n")
    raw = text.encode("utf-8")
    if len(raw) > MAX_NEWS_BYTES:
        return {"ok": False,
                "error": f"news is limited to {MAX_NEWS_BYTES} UTF-8 bytes"}
    invalid = [i for i, line in enumerate(text.splitlines(), 1)
               if line.strip() and ": " not in line]
    if invalid:
        return {"ok": False, "error": "each non-empty line must use "
                f"'Date: message' format (invalid line {invalid[0]})"}
    client("s3").put_object(
        Bucket=os.environ["CONFIG_BUCKET"], Key=NEWS_KEY, Body=raw,
        ContentType="text/plain; charset=utf-8",
        CacheControl="no-store")
    log.info("homepage news updated", extra={"bytes": len(raw),
                                             "lines": len(text.splitlines())})
    return {"ok": True, "bytes": len(raw), "lines": len(text.splitlines())}
def _clean(body, require_name=True):
    cid = (body.get("channel_id") or "").strip()
    name = (body.get("channel_name") or "").strip()
    group = (body.get("channel_group") or "").strip() or None
    if not CHANNEL_ID_RE.match(cid):
        raise ValueError("channel_id must look like UC + 22 chars "
                         "(the channel's ID, not its @handle)")
    if require_name and not name:
        raise ValueError("channel_name is required")
    # Same convention the channels.json importer uses: '_' marks a placeholder.
    if name.startswith("_") or (group or "").startswith("_"):
        raise ValueError("names/groups beginning with '_' are reserved for "
                         "placeholder entries and are never scanned")
    return cid, name, group
def _add_channel(body):
    cid, name, group = _clean(body)
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("SELECT active FROM channels WHERE channel_id = %s", (cid,))
        prior = cur.fetchone()
        cur.execute("""
            INSERT INTO channels (channel_id, channel_name, channel_group, active)
            VALUES (%s, %s, %s, TRUE)
            ON CONFLICT (channel_id) DO UPDATE
              SET channel_name  = EXCLUDED.channel_name,
                  channel_group = COALESCE(EXCLUDED.channel_group,
                                           channels.channel_group),
                  active        = TRUE,
                  updated_at    = NOW()
        """, (cid, name, group or "Unsorted"))
    conn.commit()
    log.info("channel added", extra={"channel_id": cid, "channel_name": name,
                                     "reactivated": bool(prior)})
    return {"ok": True, "op": "add", "channel_id": cid,
            "reactivated": bool(prior and not prior[0]),
            "note": "eligible for the next discovery scan"}
def _update_channel(body):
    cid, name, group = _clean(body, require_name=False)
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE channels
            SET channel_name  = COALESCE(NULLIF(%s, ''), channel_name),
                channel_group = COALESCE(%s, channel_group),
                updated_at    = NOW()
            WHERE channel_id = %s
            RETURNING channel_name, channel_group""", (name, group, cid))
        row = cur.fetchone()
    conn.commit()
    if row is None:
        raise ValueError(f"no such channel {cid}")
    return {"ok": True, "op": "update", "channel_id": cid,
            "channel_name": row[0], "channel_group": row[1]}
def _remove_channel(body):
    """
    Deactivate + cancel queued work. Deliberately destroys nothing:
      - `channels` row stays (every summary/MV joins it),
      - `videos`, `user_data`, `user_data_current` untouched,
      - terminal jobs (done/skipped/failed) untouched,
      - `channel_watermarks` kept, so re-adding resumes where it left off.
    Cancelling = deleting the non-terminal ingest_jobs rows. That is what makes
    the already-delivered SQS messages harmless:
      * download: "no job row; dropping" -> message deleted,
      * ingest:   UPDATE ... RETURNING finds nothing -> no-op,
      * a download mid-flight loses its lease on the next heartbeat and aborts.
    Raw S3 parts for cancelled downloads age out under the 90-day lifecycle.
    """
    cid = (body.get("channel_id") or "").strip()
    if not cid:
        raise ValueError("channel_id is required")
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("""UPDATE channels SET active = FALSE, updated_at = NOW()
                       WHERE channel_id = %s RETURNING channel_name""", (cid,))
        row = cur.fetchone()
        if row is None:
            conn.rollback()
            raise ValueError(f"no such channel {cid}")
        cancelled = cancel_channel_jobs(cid)
    conn.commit()
    log.warning("channel removed from rotation",
                extra={"channel_id": cid, "channel_name": row[0],
                       "cancelled_jobs": len(cancelled)})
    emit({"ChannelsDeactivated": (1, COUNT),
          "JobsCancelled": (len(cancelled), COUNT)})
    return {"ok": True, "op": "remove", "channel_id": cid, "channel_name": row[0],
            "cancelled_jobs": len(cancelled),
            "note": "existing chat data and completed jobs were kept"}

def _route(event):
    event = event or {}
    rc = event.get("requestContext", {}) or {}
    http = rc.get("http", {}) or {}
    method = (http.get("method") or event.get("httpMethod")
              or event.get("method") or "GET").upper()
    raw = (event.get("rawPath") or http.get("path") or event.get("path") or "/")
    proxy = (event.get("pathParameters") or {}).get("proxy") or ""
    for cand in (raw, "/" + proxy.lstrip("/")):
        i = cand.find("/api/")
        if i >= 0:
            return method, cand[i:]
    return method, "/"

def _resp(status, body, content_type):
    """
    Response shape that satisfies API Gateway v1 proxy, v2 payload format 2.0,
    and the emulators' various partial implementations of both.
    - `headers` in BOTH casings: some routers do an exact-key lookup on
      "Content-Type", others on "content-type", and the fallback default is
      application/json (which makes a browser treat the page as JSON).
    - `multiValueHeaders` for v1 routers that ignore `headers` entirely.
    - explicit `isBase64Encoded`: a missing key makes some routers assume the
      body is base64 and mangle it.
    """
    ct = content_type
    return {
        "statusCode": status,
        "headers": {"Content-Type": ct, "content-type": ct,
                    "Cache-Control": "no-store", "cache-control": "no-store",
                    "X-Content-Type-Options": "nosniff"},
        "multiValueHeaders": {"Content-Type": [ct], "Cache-Control": ["no-store"]},
        "isBase64Encoded": False,
        "body": body,
    }

def _body(event):
    raw = event.get("body") or ""
    if event.get("isBase64Encoded") and raw:
        import base64
        raw = base64.b64decode(raw).decode()
    return json.loads(raw or "{}")

def _json(code, doc):
    return _resp(code, json.dumps(doc, default=str), "application/json")

# --------------------------------------------------------------------------- #
# read side
# --------------------------------------------------------------------------- #
def snapshot():
    conn, sqs = get_conn(), client("sqs")
    out = {}
    cfg = settings(force=True)
    stale_after = int(cfg.get("stale_download_minutes", 15)) * 60
    with conn.cursor() as cur:
        cur.execute("SELECT status, COUNT(*) FROM ingest_jobs GROUP BY status")
        out["jobs"] = dict(cur.fetchall())
        cur.execute("""
            WITH floor AS (
              SELECT COALESCE(NULLIF(%s, '')::timestamptz,
                              '1970-01-01 00:00:00+00'::timestamptz) ts)
            SELECT LEAST(
              (SELECT MIN(date_trunc('month', v.end_time)::date)
                 FROM ingest_jobs j JOIN videos v USING (video_id), floor f
                WHERE j.status NOT IN ('done','failed','skipped')
                  AND v.end_time >= f.ts),
              (SELECT MIN(u.observed_month) FROM user_data_current u, floor f
                WHERE u.observed_month >= date_trunc('month', f.ts)::date
                  AND NOT EXISTS (
                  SELECT 1 FROM monthly_merge_state s
                   WHERE s.observed_month=u.observed_month
                     AND s.status='merged')))
        """, (cfg.get("backlog_floor", ""),))
        active_month = cur.fetchone()[0]
        out["active_month"] = str(active_month) if active_month else None
        cur.execute("""
            SELECT j.video_id, j.channel_id,
                   COALESCE(c.channel_name, j.channel_id)        AS channel_name,
                   j.status, j.attempts,
                   COALESCE(j.last_offset_s, 0)                  AS offset_s,
                   COALESCE(NULLIF(j.video_duration_s, 0),
                            EXTRACT(EPOCH FROM v.duration))      AS duration_s,
                   COALESCE(j.messages_downloaded, 0)            AS messages,
                   j.part_count,
                   EXTRACT(EPOCH FROM NOW() - j.updated_at)::int AS age_s,
                   v.title,
                   COALESCE(j.reaped_count, 0)                   AS reaped,
                   v.end_time,
                   j.enqueued_at
            FROM ingest_jobs j
            LEFT JOIN videos   v ON v.video_id   = j.video_id
            LEFT JOIN channels c ON c.channel_id = j.channel_id
            WHERE j.status IN ('downloading', 'downloaded', 'ingesting')
            ORDER BY j.updated_at DESC LIMIT 20""")
        out["active"] = [{
            "video_id": r[0], "channel_id": r[1], "channel_name": r[2],
            "status": r[3], "attempts": r[4],
            "offset_s": float(r[5]),
            "duration_s": float(r[6]) if r[6] else None,
            "messages": int(r[7]), "parts": r[8], "age_s": r[9], "title": r[10],
            "reaped": r[11],
            "end_time": r[12].isoformat(sep=" ", timespec="minutes") if r[12] else None,
            "end_time_approx": (r[13].isoformat(sep=" ", timespec="minutes")
                                if r[12] is None and r[13] else None),
            # Reaching the video's duration only measures timeline coverage;
            # YouTube may still have continuation pages with messages. Reserve
            # 100% for a job that has actually left the download phase.
            "pct": (round(min(99.9, 100.0 * float(r[5]) / float(r[6])), 1)
                    if r[3] == "downloading" and r[6] else
                    (100.0 if r[3] in ("downloaded", "ingesting") else None)),
            "phase": (f"held until {active_month} is published"
                      if r[3] == "downloaded" and active_month and r[12]
                      and r[12].date().replace(day=1) > active_month
                      else {"downloading": "downloading",
                            "downloaded": "queued for ingest",
                            "ingesting": "ingesting"}[r[3]]),
            "stalled": r[3] == "downloading" and (r[9] or 0) > stale_after,
        } for r in cur.fetchall()]
        cur.execute("""
            SELECT j.video_id, j.channel_id, j.attempts,
                   LEFT(j.last_error, 500), j.updated_at, v.end_time
            FROM ingest_jobs j
            LEFT JOIN videos v USING (video_id)
            WHERE j.status = 'failed'
            ORDER BY v.end_time DESC NULLS LAST, j.updated_at DESC""")
        out["failed"] = [{"video_id": r[0], "channel_id": r[1], "attempts": r[2],
                          "error": r[3], "updated_at": str(r[4]),
                          "end_time": (r[5].isoformat(sep=" ", timespec="minutes")
                                       if r[5] else None)}
                         for r in cur.fetchall()]
        cur.execute("""
            WITH bounds AS (
              SELECT date_trunc('month', COALESCE(NULLIF(%s, '')::timestamptz,
                         '2026-07-01 00:00:00+00'::timestamptz)
                         AT TIME ZONE 'UTC')::date AS first_month,
                     date_trunc('month', NOW() AT TIME ZONE 'UTC')::date
                         AS current_month
            ), months AS (
              SELECT generate_series(first_month::timestamp,
                                     current_month::timestamp,
                                     INTERVAL '1 month')::date AS month,
                     current_month
              FROM bounds
            ), job_stats AS (
              SELECT m.month, m.current_month,
                     COUNT(j.video_id) AS total,
                     COUNT(j.video_id) FILTER (
                       WHERE j.status IN ('pending', 'downloading')) AS pending,
                     COUNT(j.video_id) FILTER (
                       WHERE j.status IN ('downloaded', 'ingesting'))
                         AS downloaded,
                     COUNT(j.video_id) FILTER (WHERE j.status = 'done') AS done,
                     COUNT(j.video_id) FILTER (WHERE j.status = 'failed') AS failed,
                     COUNT(j.video_id) FILTER (WHERE j.status = 'skipped') AS skipped
              FROM months m
              LEFT JOIN videos v
                ON v.end_time >= (m.month::timestamp AT TIME ZONE 'UTC')
               AND v.end_time < ((m.month + INTERVAL '1 month')::timestamp
                                  AT TIME ZONE 'UTC')
              LEFT JOIN ingest_jobs j ON j.video_id = v.video_id
              GROUP BY m.month, m.current_month
            )
            SELECT js.month, js.total, js.pending, js.downloaded, js.done,
                   js.failed, js.skipped,
                   COALESCE(s.status,
                     CASE WHEN EXISTS (
                            SELECT 1 FROM service_config hold
                             WHERE hold.key='publication_hold:' || js.month::text
                               AND hold.value='true') THEN 'held'
                          WHEN js.month = js.current_month THEN 'open'
                          ELSE 'unpublished' END) AS merge_status,
                   s.merged_at,
                   CASE WHEN js.month < js.current_month THEN
                     (SELECT COUNT(*)
                        FROM channels c
                        LEFT JOIN channel_watermarks w USING (channel_id)
                       WHERE c.active
                         AND (w.last_scanned_at IS NULL
                              OR w.last_scanned_at <
                                 ((js.month + INTERVAL '1 month')::timestamp
                                  AT TIME ZONE 'UTC')))
                   ELSE 0 END AS channel_checks,
                   COALESCE((
                     SELECT COUNT(j2.video_id)
                       FROM service_config marker
                       LEFT JOIN service_config published
                         ON published.key = 'late_data_published:' || js.month::text
                       JOIN videos v2
                         ON v2.end_time >= (js.month::timestamp AT TIME ZONE 'UTC')
                        AND v2.end_time < ((js.month + INTERVAL '1 month')::timestamp
                                           AT TIME ZONE 'UTC')
                       JOIN ingest_jobs j2
                         ON j2.video_id = v2.video_id AND j2.status = 'done'
                        AND j2.completed_at > COALESCE(
                            NULLIF(published.value, '')::timestamptz,
                            s.merged_at, '-infinity'::timestamptz)
                      WHERE marker.key = 'late_data_month:' || js.month::text
                        AND marker.value = 'pending'
                   ), 0) AS late_logs,
                   js.current_month,
                   js.month = (SELECT MAX(observed_month)
                                 FROM monthly_merge_state
                                WHERE status='merged') AS can_unpublish
              FROM job_stats js
              LEFT JOIN monthly_merge_state s ON s.observed_month = js.month
             ORDER BY js.month DESC""", (cfg.get("backlog_floor", ""),))
        out["months"] = [{
            "month": str(r[0]), "total": int(r[1]), "pending": int(r[2]),
            "downloaded": int(r[3]), "done": int(r[4]), "failed": int(r[5]),
            "skipped": int(r[6]), "merge_status": r[7],
            "merged_at": str(r[8]) if r[8] else None,
            "channel_checks": int(r[9]), "late_logs": int(r[10]),
            "closed": r[0] < r[11], "can_unpublish": bool(r[12]),
        } for r in cur.fetchall()]
        cur.execute("""
            SELECT video_id, channel_id, message_count, completed_at
            FROM ingest_jobs WHERE status = 'done'
            ORDER BY completed_at DESC NULLS LAST LIMIT 10""")
        out["recent"] = [{"video_id": r[0], "channel_id": r[1],
                          "messages": r[2], "completed_at": str(r[3])}
                         for r in cur.fetchall()]
        cur.execute("""
            SELECT COUNT(*), COALESCE(SUM(message_count), 0)
            FROM ingest_jobs
            WHERE status = 'done' AND completed_at > NOW() - INTERVAL '1 hour'""")
        videos_1h, msgs_1h = cur.fetchone()
        out["throughput"] = {"videos_last_hour": videos_1h,
                             "messages_last_hour": int(msgs_1h)}
        cur.execute("""SELECT status, COUNT(*) FROM video_stream_stats
                       GROUP BY status""")
        stats_counts = dict(cur.fetchall())
        out["stream_stats_backfill"] = {
            "enabled": str(cfg.get("stream_stats_backfill_enabled", "false")).lower() == "true",
            "pending": int(stats_counts.get("pending", 0)),
            "queued": int(stats_counts.get("queued", 0)),
            "processing": int(stats_counts.get("processing", 0)),
            "ready": int(stats_counts.get("ready", 0)),
            "unavailable": int(stats_counts.get("unavailable", 0)),
            "failed": int(stats_counts.get("failed", 0)),
        }
        out["cache_warmer"] = {
            "enabled": str(cfg.get("cache_warmer_enabled", "true")).lower() == "true"
        }
        cur.execute("""
            SELECT c.channel_id, c.channel_name,
                   COALESCE(c.channel_group, '—') AS channel_group,
                   c.active,
                   w.last_scanned_at, w.last_seen_end_time, LEFT(w.last_error, 120),
                   (SELECT COUNT(*) FROM ingest_jobs j
                     WHERE j.channel_id = c.channel_id
                       AND j.status IN ('pending','downloading','downloaded','ingesting')),
                   (SELECT COUNT(*) FROM videos v WHERE v.channel_id = c.channel_id)
            FROM channels c
            LEFT JOIN channel_watermarks w ON w.channel_id = c.channel_id
            ORDER BY c.active DESC, c.channel_group, c.channel_name""")
        out["channels"] = [{
            "channel_id": r[0], "channel_name": r[1], "channel_group": r[2],
            "active": r[3],
            "last_scanned_at": str(r[4]) if r[4] else None,
            "last_seen_end_time": str(r[5]) if r[5] else None,
            "last_error": r[6], "queued": r[7], "videos": r[8],
        } for r in cur.fetchall()]
    conn.rollback()
    out["queues"] = {}
    for label, env in (("scan", "SCAN_QUEUE_URL"),
                       ("download", "DOWNLOAD_QUEUE_URL"),
                       ("download_retry", "DOWNLOAD_RETRY_QUEUE_URL"),
                       ("ingest", "INGEST_QUEUE_URL"),
                       ("download_dlq", "DOWNLOAD_DLQ_URL")):
        url = os.environ.get(env)
        if not url:
            continue
        try:
            a = sqs.get_queue_attributes(
                QueueUrl=url,
                AttributeNames=["ApproximateNumberOfMessages",
                                "ApproximateNumberOfMessagesNotVisible",
                                "ApproximateNumberOfMessagesDelayed"])["Attributes"]
            out["queues"][label] = {
                "visible": int(a.get("ApproximateNumberOfMessages", 0)),
                "in_flight": int(a.get("ApproximateNumberOfMessagesNotVisible", 0)),
                "delayed": int(a.get("ApproximateNumberOfMessagesDelayed", 0))}
        except Exception as e:
            out["queues"][label] = {"error": str(e)[:120]}
    paused = str(cfg.get("paused", "false")).lower() == "true"
    consumers = _esm_states()
    schedule = _rule_state()
    out["config"] = cfg
    out["control"] = {
        "paused": paused,
        "consumers": consumers,
        "schedule": schedule,
        # "running" only when nothing is holding the pipeline back.
        "running": (not paused
                    and all(v != "Disabled" for v in consumers.values())
                    and schedule != "DISABLED"),
        "stale_after_s": stale_after
    }
    out["refresh_seconds"] = REFRESH_SECONDS
    out["eri_usage"] = _eri_usage()
    
    return out


def _eri_usage():
    """Aggregate accepted Eri prompts without exposing per-user counters."""
    now = datetime.now(timezone.utc).date()
    month_start = now.replace(day=1)
    first = min(month_start, now - timedelta(days=6))
    dates = []
    day = first
    while day <= now:
        dates.append(day)
        day += timedelta(days=1)
    keys = [f"llm_usage_total:{day.isoformat()}" for day in dates]
    try:
        store = redis.Redis(
            host=os.environ.get("REDIS_HOST", "localhost"),
            port=int(os.environ.get("REDIS_PORT", "6379")),
            decode_responses=True, socket_connect_timeout=0.5,
            socket_timeout=0.5, retry_on_timeout=False)
        values = [int(v or 0) for v in store.mget(keys)]
        counts = dict(zip(dates, values))
        return {
            "available": True,
            "day": counts.get(now, 0),
            "week": sum(v for d, v in counts.items()
                        if d >= now - timedelta(days=6)),
            "month": sum(v for d, v in counts.items() if d >= month_start),
        }
    except redis.RedisError as exc:
        log.warning("Eri usage counters unavailable",
                    extra={"error": type(exc).__name__})
        return {"available": False, "day": 0, "week": 0, "month": 0}
def _esm_states():
    ssm, lam = client("ssm"), client("lambda")
    states = {}
    for q in MANAGED_ESMS:
        try:
            uuid = ssm.get_parameter(Name=f"/{APP}/esm/{q}")["Parameter"]["Value"]
            states[q] = lam.get_event_source_mapping(UUID=uuid).get("State", "unknown")
        except Exception:
            states[q] = "unknown"
    return states
def _rule_state():
    try:
        return client("events").describe_rule(Name=DISCOVER_RULE).get("State", "unknown")
    except Exception:
        return "unknown"
# --------------------------------------------------------------------------- #
# write side
# --------------------------------------------------------------------------- #
def control(action, body=None):
    body = body or {}
    if action not in ("start", "stop", "scan_now", "retry_failed",
                      "retry_job", "publish_month", "republish_month",
                      "unpublish_month", "stream_stats_backfill_start",
                      "stream_stats_backfill_pause",
                      "stream_stats_backfill_retry", "cache_warmer_enable",
                      "cache_warmer_disable"):
        return {"ok": False, "error": f"unknown action {action!r}"}
    if action in ("start", "stop"):
        want_running = action == "start"
        result = {"ok": True, "action": action,
                  "paused": _set_paused(not want_running),
                  "consumers": _set_consumers(want_running),
                  "schedule": _set_schedule(want_running)}
        log.info("ingestion %s", action, extra=result)
        return result
    if action == "scan_now":
        client("lambda").invoke(FunctionName=f"{APP}-discover",
                                InvocationType="Event",
                                Payload=json.dumps({"force": True}).encode())
        return {"ok": True, "action": action, "note": "discover invoked with force"}
    if action == "publish_month":
        return _publish_month(body.get("month"))
    if action == "republish_month":
        return _republish_month(body.get("month"))
    if action == "unpublish_month":
        return _unpublish_month(body.get("month"))
    if action.startswith("stream_stats_backfill_"):
        return _stream_stats_backfill_control(action)
    if action.startswith("cache_warmer_"):
        return _cache_warmer_control(action)
    video_id = body.get("video_id") if action == "retry_job" else None
    if action == "retry_job" and not video_id:
        return {"ok": False, "error": "retry_job requires video_id"}
    retried = _retry_failed(video_id)
    if action == "retry_job" and retried["matched"] == 0:
        return {"ok": False, "error": f"failed job {video_id!r} not found",
                **retried}
    return {"ok": True, "action": action, **retried}


def _stream_stats_backfill_control(action):
    enabled = action != "stream_stats_backfill_pause"
    retained = (_retained_stream_video_ids()
                if action == "stream_stats_backfill_start" else [])
    conn = get_conn()
    with conn.cursor() as cur:
        if action == "stream_stats_backfill_start":
            if retained:
                execute_values(cur, """
                    INSERT INTO video_stream_stats (video_id, status)
                    SELECT raw.video_id, 'pending'
                    FROM (VALUES %s) AS raw(video_id)
                    JOIN ingest_jobs j USING (video_id)
                    WHERE j.status='done' AND j.part_count > 0
                    ON CONFLICT (video_id) DO NOTHING""",
                    [(video_id,) for video_id in retained])
                seeded = cur.rowcount
            else:
                seeded = 0
        elif action == "stream_stats_backfill_retry":
            cur.execute("""UPDATE video_stream_stats SET status='pending',
                               attempts=0, last_error=NULL, updated_at=NOW()
                           WHERE status='failed'""")
            seeded = cur.rowcount
        else:
            seeded = 0
        cur.execute("""INSERT INTO service_config (key, value, updated_at)
                       VALUES ('stream_stats_backfill_enabled', %s, NOW())
                       ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value,
                                                       updated_at=NOW()""",
                    ("true" if enabled else "false",))
    conn.commit()
    conn.close()
    settings(force=True)
    if enabled:
        client("lambda").invoke(FunctionName=f"{APP}-reap",
                                InvocationType="Event", Payload=b"{}")
    log.info("stream statistics backfill control",
             extra={"action": action, "matched": seeded})
    return {"ok": True, "action": action, "enabled": enabled,
            "matched": seeded}


def _cache_warmer_control(action):
    enabled = action == "cache_warmer_enable"
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("""INSERT INTO service_config (key, value, updated_at)
                       VALUES ('cache_warmer_enabled', %s, NOW())
                       ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value,
                                                       updated_at=NOW()""",
                    ("true" if enabled else "false",))
    conn.commit()
    conn.close()
    settings(force=True)
    log.info("cache warmer control changed", extra={"enabled": enabled})
    return {"ok": True, "action": action, "enabled": enabled,
            "note": "web workers apply the setting within their polling interval"}


def _retained_stream_video_ids():
    """List video prefixes that physically remain in the raw lifecycle bucket."""
    s3 = client("s3")
    bucket = os.environ["RAW_BUCKET"]
    paginator = s3.get_paginator("list_objects_v2")
    channel_prefixes = []
    for page in paginator.paginate(Bucket=bucket, Delimiter="/"):
        channel_prefixes.extend(item["Prefix"]
                                for item in page.get("CommonPrefixes", []))
    videos = set()
    for channel_prefix in channel_prefixes:
        for page in paginator.paginate(Bucket=bucket, Prefix=channel_prefix,
                                       Delimiter="/"):
            for item in page.get("CommonPrefixes", []):
                parts = item["Prefix"].strip("/").split("/")
                if len(parts) == 2 and re.fullmatch(r"[A-Za-z0-9_-]{11}", parts[1]):
                    videos.add(parts[1])
    return sorted(videos)
def _set_paused(paused):
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO service_config (key, value, updated_at)
            VALUES ('paused', %s, NOW())
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value,
                                            updated_at = NOW()""",
                    ("true" if paused else "false",))
    conn.commit()
    settings(force=True)              # bust the 60s cache in this container
    return paused
def _set_consumers(enabled):
    ssm, lam = client("ssm"), client("lambda")
    out = {}
    for q in MANAGED_ESMS:
        try:
            uuid = ssm.get_parameter(Name=f"/{APP}/esm/{q}")["Parameter"]["Value"]
            lam.update_event_source_mapping(UUID=uuid, Enabled=enabled)
            out[q] = "Enabling" if enabled else "Disabling"
        except Exception as e:
            # Not fatal: service_config.paused still gates every worker.
            out[q] = f"unsupported ({type(e).__name__})"
    return out
def _set_schedule(enabled):
    ev = client("events")
    try:
        ev.enable_rule(Name=DISCOVER_RULE) if enabled else ev.disable_rule(Name=DISCOVER_RULE)
        return "ENABLED" if enabled else "DISABLED"
    except Exception as e:
        return f"unsupported ({type(e).__name__})"
def _retry_failed(video_id=None):
    conn, sqs = get_conn(), client("sqs")
    with conn.cursor() as cur:
        cur.execute("""SELECT video_id, channel_id, last_error,
                              part_count, continuation
                       FROM ingest_jobs
                       WHERE status='failed'
                         AND (%s IS NULL OR video_id=%s)
                       FOR UPDATE""", (video_id, video_id))
        failed_rows = cur.fetchall()
        rows = []
        reset_markers = ("corrupt raw chat part", "unterminated string",
                         "jsondecodeerror", "400 client error", "bad request")
        for job_id, channel_id, error, part_count, continuation in failed_rows:
            error_lower = (error or "").lower()
            line_corruption = ("unterminated string" in error_lower
                               or "jsondecodeerror" in error_lower)
            if line_corruption and part_count < 1:
                part_count = _raw_part_count(channel_id, job_id)
            reset = ((line_corruption and part_count < 1)
                     or (not line_corruption and any(
                         marker in error_lower for marker in reset_markers)))
            # A failure after download completion can retry ingestion directly
            # when its raw parts are known-good. Other failures return to the
            # downloader, preserving a valid continuation when possible.
            target = ("downloaded" if not reset and part_count > 0
                      and continuation is None else "pending")
            cur.execute("""UPDATE ingest_jobs
                           SET status=%s, attempts=0,
                               continuation=CASE WHEN %s THEN NULL
                                                 ELSE continuation END,
                               part_count=CASE WHEN %s THEN 0 ELSE %s END,
                               last_offset_s=CASE WHEN %s THEN 0
                                                  ELSE last_offset_s END,
                               messages_downloaded=CASE WHEN %s THEN 0
                                                        ELSE messages_downloaded END,
                               last_error=NULL, completed_at=NULL,
                               dispatched_at=NOW(), updated_at=NOW()
                           WHERE video_id=%s""",
                        (target, reset, reset, part_count, reset, reset, job_id))
            rows.append((job_id, channel_id, target, 0 if reset else part_count))
    conn.commit()
    send_failures = []
    for job_id, channel_id, target, part_count in rows:
        try:
            ingest_retry = target == "downloaded"
            sqs.send_message(
                QueueUrl=os.environ["INGEST_QUEUE_URL" if ingest_retry
                                    else "DOWNLOAD_QUEUE_URL"],
                MessageBody=json.dumps({"video_id": job_id,
                                        "channel_id": channel_id,
                                        "part_count": part_count,
                                        "attempt": 0,
                                        "source": "manual"}))
        except Exception:
            send_failures.append(job_id)
            log.exception("manual retry enqueue failed", extra={"video_id": job_id})
    if send_failures:
        with conn.cursor() as cur:
            cur.execute("""UPDATE ingest_jobs
                           SET status='failed', dispatched_at=NULL,
                               last_error='admin retry could not enqueue',
                               updated_at=NOW()
                           WHERE video_id = ANY(%s)""", (send_failures,))
        conn.commit()
    conn.close()
    return {"matched": len(rows),
            "requeued": len(rows) - len(send_failures),
            "enqueue_failed": len(send_failures)}


def _raw_part_count(channel_id, video_id):
    """Recover the contiguous raw-part count lost by the old repair path."""
    prefix = f"{channel_id}/{video_id}/part-"
    indexes = set()
    paginator = client("s3").get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=os.environ["RAW_BUCKET"], Prefix=prefix):
        for item in page.get("Contents", []):
            match = re.search(r"/part-(\d{5})\.jsonl\.gz$", item.get("Key", ""))
            if match:
                indexes.add(int(match.group(1)))
    count = 0
    while count in indexes:
        count += 1
    return count

def _valid_month(value):
    month = str(value or "")
    if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])-01", month):
        return None
    return month

def _publish_month(value):
    month = _valid_month(value)
    if month is None:
        return {"ok": False, "error": "month must use YYYY-MM-01"}
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("""SELECT %s::date < date_trunc(
                         'month', NOW() AT TIME ZONE 'UTC')::date,
                              EXISTS (SELECT 1 FROM monthly_merge_state
                               WHERE observed_month=%s::date AND status='merged')""",
                    (month, month))
        closed, merged = cur.fetchone()
    conn.rollback()
    conn.close()
    if not closed:
        return {"ok": False, "error": "the current month cannot be published"}
    if merged:
        return {"ok": False, "error": f"{month} is already published"}
    client("lambda").invoke(
        FunctionName=f"{APP}-merge", InvocationType="Event",
        Payload=json.dumps({"months": [month], "force": True}).encode())
    log.warning("manual month publication requested", extra={"month": month})
    return {"ok": True, "action": "publish_month", "month": month,
            "note": "forced publication and cache invalidation queued"}

def _republish_month(value):
    month = _valid_month(value)
    if month is None:
        return {"ok": False, "error": "month must use YYYY-MM-01"}
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("""SELECT 1 FROM service_config
                       WHERE key=%s AND value='pending'""",
                    (f"late_data_month:{month}",))
        pending = cur.fetchone() is not None
    conn.rollback()
    conn.close()
    if not pending:
        return {"ok": False, "error": f"no unpublished late logs for {month}"}
    client("lambda").invoke(
        FunctionName=f"{APP}-refresh", InvocationType="Event",
        Payload=json.dumps({"publish_months": [month]}).encode())
    log.info("late month re-publication requested", extra={"month": month})
    return {"ok": True, "action": "republish_month", "month": month,
            "note": "refresh and cache invalidation queued"}


def _unpublish_month(value):
    month = _valid_month(value)
    if month is None:
        return {"ok": False, "error": "month must use YYYY-MM-01"}
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("""SELECT EXISTS (
                         SELECT 1 FROM monthly_merge_state
                          WHERE observed_month=%s::date AND status='merged'),
                              %s::date = (SELECT MAX(observed_month)
                                FROM monthly_merge_state WHERE status='merged')""",
                    (month, month))
        published, newest = cur.fetchone()
    conn.rollback()
    conn.close()
    if not published:
        return {"ok": False, "error": f"{month} is not published"}
    if not newest:
        return {"ok": False,
                "error": "only the newest published month can be unpublished"}
    client("lambda").invoke(
        FunctionName=f"{APP}-merge", InvocationType="Event",
        Payload=json.dumps({"unpublish_months": [month]}).encode())
    log.warning("month unpublication requested", extra={"month": month})
    return {"ok": True, "action": "unpublish_month", "month": month,
            "note": "month is being returned to staging and held"}
# --------------------------------------------------------------------------- #
# the page
# --------------------------------------------------------------------------- #
def _page():
    return _resp(200, PAGE.replace("__REFRESH__", str(REFRESH_SECONDS * 1000)),
                 "text/html; charset=utf-8")
PAGE = r"""<!doctype html>
<html lang="en">
<head>
<base href="./">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HoloChatStats Admin Page</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; background:#12141a; color:#e6e8ee;
         font:14px/1.45 ui-sans-serif,system-ui,"Segoe UI",Roboto,sans-serif; }
  header { display:flex; align-items:center; gap:16px; flex-wrap:wrap;
           padding:14px 20px; background:#1b1e26; border-bottom:1px solid #2c313d; }
  h1 { font-size:18px; margin:0; font-weight:600; }
  .pill { padding:3px 10px; border-radius:999px; font-size:12px; font-weight:600; }
  .run  { background:#123b25; color:#57d68b; border:1px solid #1e6b41; }
  .stop { background:#3b1414; color:#ff8585; border:1px solid #6b1e1e; }
  .warn { background:#3b3314; color:#ffd666; border:1px solid #6b5c1e; }
  button { background:#2a2f3b; color:#e6e8ee; border:1px solid #3a4050;
           border-radius:6px; padding:7px 14px; font-size:13px; cursor:pointer; }
  button:hover:not(:disabled) { background:#343a49; }
  button:disabled { opacity:.45; cursor:not-allowed; }
  button.go   { background:#17502f; border-color:#1e6b41; }
  button.halt { background:#55191b; border-color:#7a2226; }
  main { box-sizing:border-box; max-width:1440px; margin:0 auto;
         padding:12px 14px 28px; display:grid; gap:12px;
         grid-template-columns:repeat(auto-fill,minmax(min(100%,300px),440px));
         justify-content:start; align-items:start; }
  section { box-sizing:border-box; background:#171a21; border:1px solid #262b36;
            border-radius:9px; padding:11px 13px; min-width:0; }
  section.wide { grid-column:1/-1; width:min(100%,1180px); }
  section.medium { grid-column:1/-1; width:min(100%,900px); }
  section h2 { font-size:13px; letter-spacing:.08em; text-transform:uppercase;
               color:#8a93a6; margin:0 0 10px; }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th { text-align:left; color:#8a93a6; font-weight:500; border-bottom:1px solid #262b36; padding:5px 6px; }
  td { padding:5px 6px; border-bottom:1px solid #1f242e; vertical-align:top; }
  td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; }
  .kpis { display:flex; gap:10px; flex-wrap:wrap; }
  .kpi { flex:1 1 110px; background:#1b1f28; border:1px solid #262b36;
         border-radius:8px; padding:10px 12px; }
  .kpi .v { font-size:22px; font-weight:650; font-variant-numeric:tabular-nums; }
  .kpi .k { font-size:11px; color:#8a93a6; text-transform:uppercase; letter-spacing:.06em; }
  .bar { background:#222733; border-radius:4px; height:7px; width:110px; overflow:hidden; }
  .bar > i { display:block; height:100%; background:#4f8cff; }
  .err { color:#ff9a9a; font-family:ui-monospace,Menlo,monospace; font-size:11.5px;
         word-break:break-word; }
  code { font-family:ui-monospace,Menlo,monospace; }
  #meta { margin-left:auto; color:#707a8c; font-size:12px; }
  .s-done{color:#57d68b} .s-failed{color:#ff8585} .s-pending{color:#8a93a6}
  .s-downloading,.s-ingesting,.s-downloaded{color:#ffd666} .s-skipped{color:#7f87f0}
  tr.stalled td { background:#2a1b1b; }
  .bar { display:inline-block; vertical-align:middle; margin-right:6px; }
  .chan { margin-right:6px; }
  button.copy { padding:1px 6px; font-size:11px; line-height:1.4; border-radius:4px;
                background:#222733; border-color:#333b4b; color:#8a93a6; }
  button.copy:hover { color:#e6e8ee; background:#2d3442; }
  button.copy.ok { color:#57d68b; border-color:#1e6b41; }
  .reap { margin-left:6px; color:#ffd666; font-size:11px; }
  button.small { font-size:11px; padding:2px 8px; margin-left:10px; vertical-align:middle; }
  button.danger { background:#55191b; border-color:#7a2226; }
  .inactive td { opacity:.5; }
  .tag { padding:2px 8px; border-radius:999px; font-size:11px; background:#222733;
         border:1px solid #323a49; color:#9aa3b4; }
  .tag.on  { background:#123b25; color:#57d68b; border-color:#1e6b41; }
  dialog { background:#171a21; color:#e6e8ee; border:1px solid #2c313d;
           border-radius:10px; padding:18px 20px; min-width:360px; }
  dialog::backdrop { background:rgba(0,0,0,.6); }
  dialog h3 { margin:0 0 12px; font-size:15px; }
  dialog label { display:block; margin-bottom:10px; font-size:12px; color:#8a93a6; }
  dialog input { display:block; width:100%; margin-top:4px; padding:7px 9px;
                 background:#12141a; color:#e6e8ee; border:1px solid #323a49;
                 border-radius:6px; font:13px ui-monospace,Menlo,monospace; }
  dialog input:disabled { opacity:.5; }
  textarea { box-sizing:border-box; width:100%; min-height:150px; resize:vertical;
             padding:9px 11px; background:#12141a; color:#e6e8ee;
             border:1px solid #323a49; border-radius:6px;
             font:13px/1.5 ui-monospace,Menlo,monospace; }
  .editor-actions { display:flex; align-items:center; gap:10px; margin-top:9px; }
  #news-state { color:#8a93a6; font-size:12px; }
  dialog menu { display:flex; gap:8px; justify-content:flex-end; padding:0; margin:14px 0 0; }
  .hint { color:#5d6676; text-transform:none; letter-spacing:0; }
  td.ts { font-family:ui-monospace,Menlo,monospace; font-size:12px; color:#9aa3b4;
          white-space:nowrap; }
  .approx { color:#ffd666; }
  .table-scroll { overflow:auto; border:1px solid #222733; border-radius:6px; }
  .table-scroll.months { max-height:330px; }
  .table-scroll.channels { max-height:520px; }
  .table-scroll.failed { max-height:420px; }
  .table-scroll table { min-width:820px; }
  .table-scroll th { position:sticky; top:0; z-index:1; background:#171a21; }
  .actions { display:flex; align-items:center; gap:5px; white-space:nowrap; }
  .actions button.small { margin-left:0; }
  @media (max-width:700px) {
    header { padding:10px 12px; gap:8px; }
    main { padding:9px; }
    section.wide, section.medium { grid-column:auto; width:100%; }
  }
</style>
</head>
<body>
<header>
  <h1>HoloChatStats Admin Page</h1>
  <span id="state" class="pill warn">connecting…</span>
  <button id="btn-start" class="go">▶ Start ingestion</button>
  <button id="btn-stop" class="halt">■ Stop ingestion</button>
  <button id="btn-scan">Scan now</button>
  <span id="meta">—</span>
</header>
<main>
  <section class="wide">
    <h2>Pipeline</h2>
    <div class="kpis" id="kpis"></div>
  </section>
  <section>
    <h2>Queues</h2>
    <table><thead><tr><th>Queue</th><th class="num">Visible</th>
      <th class="num">In flight</th><th class="num">Delayed</th></tr></thead>
      <tbody id="queues"></tbody></table>
  </section>
  <section>
    <h2>Jobs by status</h2>
    <table><thead><tr><th>Status</th><th class="num">Count</th></tr></thead>
      <tbody id="jobs"></tbody></table>
  </section>
  <section>
    <h2>Eri usage <span class="hint">accepted prompts, UTC</span></h2>
    <div class="kpis" id="eri-usage"></div>
  </section>
  <section class="medium">
    <h2>Stream stats backfill</h2>
    <p class="hint">Low-priority aggregate-only processing of retained raw
      logs. At most one job uses the ingest worker at a time.</p>
    <div class="kpis" id="stream-stats-kpis"></div>
    <div class="editor-actions">
      <button id="btn-stats-start" class="go">Start</button>
      <button id="btn-stats-pause" class="halt">Pause</button>
      <button id="btn-stats-retry">Retry failed</button>
    </div>
  </section>
  <section>
    <h2>Cache warmer</h2>
    <p class="hint">Pre-computes expensive bounded analytics during quiet
      periods. Disabling it does not delete existing cached results.</p>
    <p><span id="cache-warmer-state" class="tag">loading</span></p>
    <div class="editor-actions">
      <button id="btn-warmer-enable" class="go">Enable</button>
      <button id="btn-warmer-disable" class="halt">Disable</button>
    </div>
  </section>
  <section class="wide">
    <h2>In progress</h2>
    <table><thead><tr><th>Video</th><th>Channel</th><th>Stream ended (UTC)</th>
      <th>Phase</th><th class="num">Try</th><th>Progress</th>
      <th class="num">Position</th><th class="num">Messages</th>
      <th class="num">Parts</th><th class="num">Heartbeat</th></tr></thead>
      <tbody id="active"></tbody></table>
  </section>
  <section>
    <h2>Recently completed</h2>
    <table><thead><tr><th>Video</th><th class="num">Messages</th>
      <th>Finished</th></tr></thead><tbody id="recent"></tbody></table>
  </section>
  <section class="medium">
    <h2>Failed <button id="btn-retry-all" class="small">Retry all</button></h2>
    <p class="hint">Failed jobs do not block month publication. Retry only
      after correcting the underlying problem.</p>
    <div class="table-scroll failed"><table><thead><tr><th>Video</th><th>Stream ended (UTC)</th>
      <th class="num">Try</th><th>Error</th><th>Actions</th></tr></thead>
      <tbody id="failed"></tbody></table></div>
  </section>
  <section class="wide">
    <h2>Monthly publication</h2>
    <p class="hint">Channel checks are publication-barrier scans. Manual
      publication overrides pending work and channel checks; use it only when
      accepting incomplete coverage. Unpublish returns a month to staging and
      holds automatic publication until Publish is clicked. Re-publish applies
      late logs and invalidates that month's caches.</p>
    <div class="table-scroll months"><table><thead><tr><th>Month</th><th class="num">Known</th>
      <th class="num">Pending</th><th class="num">Downloaded</th>
      <th class="num">Ingested</th><th class="num">Failed</th>
      <th class="num">Skipped</th><th class="num">Channel checks</th>
      <th>Merge status</th><th class="num">Late logs</th><th>Actions</th></tr></thead>
      <tbody id="months"></tbody></table></div>
  </section>
  <section class="wide">
    <h2>Channels <button id="btn-add-chan" class="go small">+ Add channel</button></h2>
    <div class="table-scroll channels"><table><thead><tr><th>Channel</th><th>Group</th><th>State</th>
      <th class="num">Queued</th><th class="num">Videos</th>
      <th>Last scanned</th><th>Watermark</th><th>Error</th>
      <th>Actions</th></tr></thead><tbody id="channels"></tbody></table></div>
  </section>
  <section class="medium">
    <h2>Homepage news</h2>
    <p class="hint">One item per line using <code>Date: message</code>. Changes
      are stored in S3 and appear after the homepage is refreshed.</p>
    <textarea id="news-text" maxlength="20000" spellcheck="true"
      placeholder="September 7, 2026: News message"></textarea>
    <div class="editor-actions">
      <button id="btn-news-save" class="go">Save news</button>
      <button id="btn-news-reload">Discard changes</button>
      <span id="news-state">loadingâ€¦</span>
    </div>
  </section>
</main>
<dialog id="chan-dlg">
  <form id="chan-form">
    <h3 id="chan-title">Add channel</h3>
    <label>Channel ID<input id="f-id" autocomplete="off" spellcheck="false"
        placeholder="UCxxxxxxxxxxxxxxxxxxxxxx"></label>
    <label>Name<input id="f-name" autocomplete="off"></label>
    <label>Group <span class="hint">(optional)</span><input id="f-group"
        autocomplete="off" placeholder="Unsorted"></label>
    <p id="f-err" class="err"></p>
    <menu>
      <button type="button" id="f-cancel">Cancel</button>
      <button type="submit" id="f-save" class="go">Save</button>
    </menu>
  </form>
</dialog>
<script>
const REFRESH = __REFRESH__;
const $ = id => document.getElementById(id);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c =>
  ({ "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;" }[c]));
const ago = s => s == null ? "—" : s < 60 ? s + "s" : s < 3600
  ? Math.floor(s/60) + "m" : Math.floor(s/3600) + "h";
const hms = s => {
  if (s == null) return "—";
  s = Math.max(0, Math.floor(s));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), x = s % 60;
  return (h ? h + ":" + String(m).padStart(2, "0") : String(m))
       + ":" + String(x).padStart(2, "0");
};
const chan = (name, id) =>
  `<span class="chan" title="${esc(id)}">${esc(name)}</span>` +
  `<button class="copy" data-id="${esc(id)}" title="Copy channel ID ${esc(id)}"
           aria-label="Copy channel ID">⧉</button>`;
// Delegated: the tables are replaced wholesale every refresh, so per-button
// listeners would be orphaned immediately.
document.addEventListener("click", async ev => {
  const btn = ev.target.closest("button.copy");
  if (!btn) return;
  const id = btn.dataset.id;
  try {
    await navigator.clipboard.writeText(id);     // localhost is a secure context
  } catch {
    const t = document.createElement("textarea");
    t.value = id; t.style.position = "fixed"; t.style.opacity = "0";
    document.body.appendChild(t); t.select();
    document.execCommand("copy"); t.remove();
  }
  btn.textContent = "✓"; btn.classList.add("ok");
  setTimeout(() => { btn.textContent = "⧉"; btn.classList.remove("ok"); }, 1200);
});
let busy = false;
async function act(action, fields = {}) {
  if (busy) return;
  busy = true;
  document.querySelectorAll("header button").forEach(b => b.disabled = true);
  try {
    const r = await fetch("api/control", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ action, ...fields })
    });
    const j = await r.json();
    if (!j.ok) alert("Action failed: " + (j.error || r.status));
  } catch (e) { alert("Action failed: " + e); }
  busy = false;
  await tick();
}
$("btn-start").onclick = () => act("start");
$("btn-stop").onclick  = () => act("stop");
$("btn-scan").onclick  = () => act("scan_now");
$("btn-retry-all").onclick = () => {
  if (confirm("Re-queue every failed job?")) act("retry_failed");
};
$("btn-stats-start").onclick = () => act("stream_stats_backfill_start");
$("btn-stats-pause").onclick = () => act("stream_stats_backfill_pause");
$("btn-stats-retry").onclick = () => {
  if (confirm("Retry failed stream-stat backfills?"))
    act("stream_stats_backfill_retry");
};
$("btn-warmer-enable").onclick = () => act("cache_warmer_enable");
$("btn-warmer-disable").onclick = () => act("cache_warmer_disable");
let newsDirty = false;
async function loadNews() {
  $("news-state").textContent = "loadingâ€¦";
  try {
    const r = await fetch("api/news", { cache: "no-store" });
    const j = await r.json();
    if (!r.ok || !j.ok) throw new Error(j.error || "HTTP " + r.status);
    $("news-text").value = j.text || "";
    newsDirty = false;
    $("news-state").textContent = "loaded";
  } catch (e) { $("news-state").textContent = "load failed: " + e; }
}
$("news-text").addEventListener("input", () => {
  newsDirty = true; $("news-state").textContent = "unsaved changes";
});
$("btn-news-reload").onclick = () => {
  if (!newsDirty || confirm("Discard unsaved news changes?")) loadNews();
};
$("btn-news-save").onclick = async () => {
  $("btn-news-save").disabled = true;
  $("news-state").textContent = "savingâ€¦";
  try {
    const r = await fetch("api/news", {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ text: $("news-text").value })
    });
    const j = await r.json();
    if (!r.ok || !j.ok) throw new Error(j.error || "HTTP " + r.status);
    newsDirty = false;
    $("news-state").textContent = `saved ${j.lines} line(s)`;
  } catch (e) { $("news-state").textContent = "save failed: " + e; }
  $("btn-news-save").disabled = false;
};
const dlg = $("chan-dlg");
let editingId = null;
function openChan(mode, ch) {
  editingId = mode === "edit" ? ch.channel_id : null;
  $("chan-title").textContent = mode === "edit" ? "Edit channel" : "Add channel";
  $("f-id").value = ch ? ch.channel_id : "";
  $("f-id").disabled = mode === "edit";      // the ID is the key; it never changes
  $("f-name").value = ch ? ch.channel_name : "";
  $("f-group").value = ch && ch.channel_group !== "—" ? ch.channel_group : "";
  $("f-err").textContent = "";
  dlg.showModal();
  (mode === "edit" ? $("f-name") : $("f-id")).focus();
}
$("btn-add-chan").onclick = () => openChan("add", null);
$("f-cancel").onclick = () => dlg.close();
$("chan-form").addEventListener("submit", async ev => {
  ev.preventDefault();
  const payload = {
    op: editingId ? "update" : "add",
    channel_id: editingId || $("f-id").value.trim(),
    channel_name: $("f-name").value.trim(),
    channel_group: $("f-group").value.trim(),
  };
  const j = await channelsApi(payload);
  if (!j.ok) { $("f-err").textContent = j.error; return; }
  dlg.close();
  await tick();
});
async function channelsApi(payload) {
  try {
    const r = await fetch("api/channels", {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify(payload)
    });
    return await r.json();
  } catch (e) { return { ok: false, error: String(e) }; }
}
async function removeChan(ch) {
  const msg = `Remove "${ch.channel_name}" from the scan rotation?\n\n`
    + `• ${ch.queued} queued/in-flight job(s) will be cancelled\n`
    + `• it will not be scanned again\n`
    + `• its ${ch.videos} video(s) and all chat data stay in the database`;
  if (!confirm(msg)) return;
  const j = await channelsApi({ op: "remove", channel_id: ch.channel_id });
  if (!j.ok) alert("Remove failed: " + j.error);
  await tick();
}
async function readdChan(ch) {
  const j = await channelsApi({ op: "add", channel_id: ch.channel_id,
                               channel_name: ch.channel_name,
                               channel_group: ch.channel_group === "—" ? "" : ch.channel_group });
  if (!j.ok) alert("Re-add failed: " + j.error);
  await tick();
}
// expose for the inline handlers below
let CHANNELS = [];
let FAILED = [];
let MONTHS = [];
window.chanAction = (i, what) => {
  const ch = CHANNELS[i];
  if (what === "edit") openChan("edit", ch);
  else if (what === "remove") removeChan(ch);
  else if (what === "readd") readdChan(ch);
};
window.retryFailed = i => {
  const job = FAILED[i];
  if (job && confirm(`Retry download ${job.video_id}?`))
    act("retry_job", { video_id: job.video_id });
};
window.republishMonth = i => {
  const item = MONTHS[i];
  if (item && confirm(`Re-publish ${item.month} and invalidate its caches?`))
    act("republish_month", { month: item.month });
};
window.unpublishMonth = i => {
  const item = MONTHS[i];
  if (!item) return;
  const warning = `Unpublish ${item.month}?\n\n`
    + `Its rows will return to staging, public coverage caches will be cleared, `
    + `and automatic publication will be held. Click Publish when the month is ready.`;
  if (confirm(warning)) act("unpublish_month", { month: item.month });
};
window.publishMonth = i => {
  const item = MONTHS[i];
  if (!item) return;
  const warning = `Publish ${item.month} now?\n\n`
    + `This overrides ${item.pending} pending download(s), ${item.failed} failed job(s), `
    + `and ${item.channel_checks} outstanding channel check(s). `
    + `The month may have incomplete coverage.`;
  if (confirm(warning)) act("publish_month", { month: item.month });
};
function render(d) {
  const c = d.control;
  const st = $("state");
  st.textContent = c.running ? "RUNNING" : (c.paused ? "STOPPED (paused)" : "PARTIALLY STOPPED");
  st.className = "pill " + (c.running ? "run" : c.paused ? "stop" : "warn");
  if (!busy) {
    $("btn-start").disabled = c.running;
    $("btn-stop").disabled  = c.paused;
    $("btn-scan").disabled  = false;
  }
  $("btn-retry-all").disabled = busy || !(d.jobs.failed > 0);
  const j = d.jobs, q = d.queues;
  const backlog = (j.pending||0) + (j.downloading||0) + (j.downloaded||0) + (j.ingesting||0);
  $("kpis").innerHTML = [
    ["Backlog", backlog],
    ["Done", j.done || 0],
    ["Failed", j.failed || 0],
    ["Skipped", j.skipped || 0],
    ["Videos / hr", d.throughput.videos_last_hour],
    ["Msgs / hr", d.throughput.messages_last_hour.toLocaleString()],
    ["DLQ", (q.download_dlq && q.download_dlq.visible) || 0],
  ].map(([k, v]) => `<div class="kpi"><div class="v">${esc(v)}</div><div class="k">${k}</div></div>`).join("");
  const eu = d.eri_usage || { available:false, day:0, week:0, month:0 };
  $("eri-usage").innerHTML = eu.available
    ? [["Today", eu.day], ["Last 7 days", eu.week], ["This month", eu.month]]
        .map(([k, v]) => `<div class="kpi"><div class="v">${Number(v).toLocaleString()}</div><div class="k">${k}</div></div>`).join("")
    : `<span class="err">usage store unavailable</span>`;
  const sb = d.stream_stats_backfill || {};
  $("stream-stats-kpis").innerHTML = [
    ["Pending", sb.pending || 0],
    ["Active", (sb.queued || 0) + (sb.processing || 0)],
    ["Completed", sb.ready || 0],
    ["Unavailable", sb.unavailable || 0],
    ["Failed", sb.failed || 0],
  ].map(([k, v]) => `<div class="kpi"><div class="v">${Number(v).toLocaleString()}</div><div class="k">${k}</div></div>`).join("");
  $("btn-stats-start").disabled = busy || Boolean(sb.enabled);
  $("btn-stats-pause").disabled = busy || !sb.enabled;
  $("btn-stats-retry").disabled = busy || !(sb.failed > 0);
  const cw = d.cache_warmer || { enabled:true };
  $("cache-warmer-state").textContent = cw.enabled ? "enabled" : "disabled";
  $("cache-warmer-state").className = "tag " + (cw.enabled ? "on" : "");
  $("btn-warmer-enable").disabled = busy || cw.enabled;
  $("btn-warmer-disable").disabled = busy || !cw.enabled;
  $("queues").innerHTML = Object.entries(q).map(([n, v]) => v.error
    ? `<tr><td>${esc(n)}</td><td colspan="3" class="err">${esc(v.error)}</td></tr>`
    : `<tr><td>${esc(n)}</td><td class="num">${v.visible}</td>
        <td class="num">${v.in_flight}</td><td class="num">${v.delayed}</td></tr>`).join("");
  $("jobs").innerHTML = Object.entries(j).sort().map(([s, n]) =>
    `<tr><td class="s-${esc(s)}">${esc(s)}</td><td class="num">${n}</td></tr>`).join("")
    || `<tr><td colspan="2">no jobs</td></tr>`;
  $("active").innerHTML = d.active.map(a => {
    const bar = a.pct == null
      ? `<span class="s-pending">unknown length</span>`
      : `<div class="bar"><i style="width:${a.pct}%"></i></div>
         <span class="num">${a.pct.toFixed(1)}%</span>`;
    return `<tr${a.stalled ? ' class="stalled"' : ""}>
      <td title="${esc(a.title || "")}"><code>${esc(a.video_id)}</code></td>
      <td>${chan(a.channel_name, a.channel_id)}</td>
      <td class="ts">${a.end_time ? esc(a.end_time)
        : a.end_time_approx
          ? `<span class="approx" title="videos.end_time is NULL — run migrate repair_end_times">?&nbsp;${esc(a.end_time_approx)}</span>`
          : `<span class="approx" title="no end_time recorded">unknown</span>`}</td>
      <td class="s-${esc(a.status)}">${esc(a.phase)}${a.stalled ? " ⚠ stalled" : ""}${
        a.reaped ? `<span class="reap" title="auto-recovered ${a.reaped}×">↻${a.reaped}</span>` : ""}</td>
      <td class="num">${a.attempts}</td>
      <td>${bar}</td>
      <td class="num">${hms(a.offset_s)}${a.duration_s ? " / " + hms(a.duration_s) : ""}</td>
      <td class="num">${a.messages.toLocaleString()}</td>
      <td class="num">${a.parts}</td>
      <td class="num">${ago(a.age_s)} ago</td></tr>`;
  }).join("") || `<tr><td colspan="10">idle</td></tr>`;
  $("recent").innerHTML = d.recent.map(r => `<tr><td><code>${esc(r.video_id)}</code></td>
      <td class="num">${(r.messages ?? 0).toLocaleString()}</td>
      <td>${esc((r.completed_at || "").slice(0, 19))}</td></tr>`).join("")
    || `<tr><td colspan="3">nothing yet</td></tr>`;
  FAILED = d.failed || [];
  $("failed").innerHTML = FAILED.map((f, i) => `<tr>
      <td><code>${esc(f.video_id)}</code></td>
      <td class="ts">${esc(f.end_time || "unknown")}</td>
      <td class="num">${f.attempts}</td><td class="err">${esc(f.error)}</td>
      <td><button class="small" onclick="retryFailed(${i})">Retry</button></td></tr>`).join("")
    || `<tr><td colspan="5">none 🎉</td></tr>`;
  MONTHS = d.months || [];
  $("months").innerHTML = MONTHS.map((m, i) => {
    const publish = m.closed && m.merge_status !== "merged"
      ? `<button class="small go" onclick="publishMonth(${i})">Publish</button>` : "";
    const republish = m.merge_status === "merged" && m.late_logs > 0
      ? `<button class="small go" onclick="republishMonth(${i})">Re-publish</button>` : "";
    const unpublish = m.merge_status === "merged" && m.can_unpublish
      ? `<button class="small danger" onclick="unpublishMonth(${i})">Unpublish</button>` : "";
    const actions = publish + republish + unpublish;
    return `<tr><td class="ts">${esc(m.month)}</td>
      <td class="num">${m.total.toLocaleString()}</td>
      <td class="num">${m.pending.toLocaleString()}</td>
      <td class="num">${m.downloaded.toLocaleString()}</td>
      <td class="num">${m.done.toLocaleString()}</td>
      <td class="num">${m.failed.toLocaleString()}</td>
      <td class="num">${m.skipped.toLocaleString()}</td>
      <td class="num">${m.channel_checks.toLocaleString()}</td>
      <td class="s-${esc(m.merge_status)}" title="${esc(m.merged_at || "")}">${esc(m.merge_status)}</td>
      <td class="num">${m.late_logs.toLocaleString()}</td>
      <td><div class="actions">${actions || "—"}</div></td></tr>`;
  }).join("") || `<tr><td colspan="11">no months configured</td></tr>`;
  CHANNELS = d.channels;
  $("channels").innerHTML = d.channels.map((ch, i) => `
    <tr class="${ch.active ? "" : "inactive"}">
      <td>${chan(ch.channel_name, ch.channel_id)}</td>
      <td>${esc(ch.channel_group)}</td>
      <td><span class="tag ${ch.active ? "on" : ""}">${ch.active ? "active" : "inactive"}</span></td>
      <td class="num">${ch.queued}</td>
      <td class="num">${ch.videos}</td>
      <td>${esc((ch.last_scanned_at || "never").slice(0, 19))}</td>
      <td>${esc((ch.last_seen_end_time || "—").slice(0, 19))}</td>
      <td class="err">${esc(ch.last_error || "")}</td>
      <td>
        <button class="small" onclick="chanAction(${i},'edit')">Edit</button>
        ${ch.active
          ? `<button class="small danger" onclick="chanAction(${i},'remove')">Remove</button>`
          : `<button class="small go" onclick="chanAction(${i},'readd')">Re-add</button>`}
      </td></tr>`).join("")
    || `<tr><td colspan="9">no channels — click “Add channel”</td></tr>`;
  const esm = Object.entries(c.consumers).map(([k, v]) => `${k}=${v}`).join("  ");
  $("meta").textContent = `updated ${new Date().toLocaleTimeString()} · `
    + `schedule=${c.schedule} · ${esm} · refresh ${REFRESH / 1000}s`;
}
async function tick() {
  if (document.querySelector("button.copy.ok")) return;
  // Never re-render while a modal is open or a "copied" tick is showing.
  if (dlg.open || document.querySelector("button.copy.ok")) return;
  try {
    const r = await fetch("api/status", { cache: "no-store" });
    if (!r.ok) throw new Error("HTTP " + r.status);
    render(await r.json());
  } catch (e) {
    $("state").textContent = "DISCONNECTED";
    $("state").className = "pill warn";
    $("meta").textContent = String(e);
  }
}
tick();
loadNews();
setInterval(tick, REFRESH);   // page updates itself, no reload
</script>
</body>
</html>
"""
