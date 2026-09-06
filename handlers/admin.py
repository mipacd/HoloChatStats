"""
Admin surface for the chat-ingestion pipeline.
Routes (served to the admin gateway / API Gateway):
    GET  /                -> HTML dashboard ("HoloChatStats Admin Page")
    GET  /api/status      -> JSON snapshot (jobs, queues, channels, run state)
    POST /api/control     -> {"action": "start" | "stop" | "scan_now" | "retry_failed"}
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
from common.aws import client
from common.config import settings
from common.db import get_conn
from common.logging_utils import get_logger
from common.metrics import emit, COUNT
from common.channels import cancel_channel_jobs

log = get_logger("admin")
APP = os.environ.get("APP_NAME", "chat-ingest")
DISCOVER_RULE = os.environ.get("DISCOVER_RULE", f"{APP}-discover-schedule")
MANAGED_ESMS = ("scan-q", "download-q", "ingest-q")
REFRESH_SECONDS = int(os.environ.get("ADMIN_REFRESH_SECONDS", "5"))
CHANNEL_ID_RE = re.compile(r"^UC[A-Za-z0-9_-]{22}$")
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
            return _json(200, control(body.get("action")))
        if method in ("GET", "HEAD") and not path.startswith("/api/"):
            return _page()            # any other GET is the dashboard
        if method == "POST" and path == "/api/channels":
            return _json(200, channels_op(_body(event)))
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
            "pct": (round(min(100.0, 100.0 * float(r[5]) / float(r[6])), 1)
                    if r[3] == "downloading" and r[6] else
                    (100.0 if r[3] in ("downloaded", "ingesting") else None)),
            "phase": {"downloading": "downloading",
                      "downloaded": "queued for ingest",
                      "ingesting": "ingesting"}[r[3]],
            "stalled": r[3] == "downloading" and (r[9] or 0) > stale_after,
        } for r in cur.fetchall()]
        cur.execute("""
            SELECT video_id, channel_id, attempts, LEFT(last_error, 160), updated_at
            FROM ingest_jobs WHERE status = 'failed'
            ORDER BY updated_at DESC LIMIT 15""")
        out["failed"] = [{"video_id": r[0], "channel_id": r[1], "attempts": r[2],
                          "error": r[3], "updated_at": str(r[4])}
                         for r in cur.fetchall()]
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
    
    return out
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
def control(action):
    if action not in ("start", "stop", "scan_now", "retry_failed"):
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
    return {"ok": True, "action": action, "requeued": _retry_failed()}
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
def _retry_failed():
    conn, sqs = get_conn(), client("sqs")
    with conn.cursor() as cur:
        cur.execute("""UPDATE ingest_jobs
                       SET status = 'pending', attempts = 0, last_error = NULL,
                           updated_at = NOW()
                       WHERE status = 'failed'
                       RETURNING video_id, channel_id""")
        rows = cur.fetchall()
    conn.commit()
    url = os.environ["DOWNLOAD_QUEUE_URL"]
    for video_id, channel_id in rows:
        sqs.send_message(QueueUrl=url,
                         MessageBody=json.dumps({"video_id": video_id,
                                                 "channel_id": channel_id,
                                                 "attempt": 0}))
    return len(rows)
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
  main { padding:18px 20px 40px; display:grid; gap:18px;
         grid-template-columns:repeat(auto-fit,minmax(420px,1fr)); }
  section { background:#171a21; border:1px solid #262b36; border-radius:10px; padding:14px 16px; }
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
  dialog menu { display:flex; gap:8px; justify-content:flex-end; padding:0; margin:14px 0 0; }
  .hint { color:#5d6button; }
  .hint { color:#5d6676; text-transform:none; letter-spacing:0; }
  td.ts { font-family:ui-monospace,Menlo,monospace; font-size:12px; color:#9aa3b4;
          white-space:nowrap; }
  .approx { color:#ffd666; }
</style>
</head>
<body>
<header>
  <h1>HoloChatStats Admin Page</h1>
  <span id="state" class="pill warn">connecting…</span>
  <button id="btn-start" class="go">▶ Start ingestion</button>
  <button id="btn-stop" class="halt">■ Stop ingestion</button>
  <button id="btn-scan">Scan now</button>
  <button id="btn-retry">Retry failed</button>
  <span id="meta">—</span>
</header>
<main>
  <section style="grid-column:1/-1">
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
  <section style="grid-column:1/-1">
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
  <section>
    <h2>Failed</h2>
    <table><thead><tr><th>Video</th><th class="num">Try</th>
      <th>Error</th></tr></thead><tbody id="failed"></tbody></table>
  </section>
  <section style="grid-column:1/-1">
    <h2>Channels <button id="btn-add-chan" class="go small">+ Add channel</button></h2>
    <table><thead><tr><th>Channel</th><th>Group</th><th>State</th>
      <th class="num">Queued</th><th class="num">Videos</th>
      <th>Last scanned</th><th>Watermark</th><th>Error</th>
      <th>Actions</th></tr></thead><tbody id="channels"></tbody></table>
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
async function act(action) {
  if (busy) return;
  busy = true;
  document.querySelectorAll("header button").forEach(b => b.disabled = true);
  try {
    const r = await fetch("api/control", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ action })
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
$("btn-retry").onclick = () => { if (confirm("Re-queue every failed job?")) act("retry_failed"); };
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
window.chanAction = (i, what) => {
  const ch = CHANNELS[i];
  if (what === "edit") openChan("edit", ch);
  else if (what === "remove") removeChan(ch);
  else if (what === "readd") readdChan(ch);
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
    $("btn-retry").disabled = !(d.jobs.failed > 0);
  }
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
  $("failed").innerHTML = d.failed.map(f => `<tr><td><code>${esc(f.video_id)}</code></td>
      <td class="num">${f.attempts}</td><td class="err">${esc(f.error)}</td></tr>`).join("")
    || `<tr><td colspan="3">none 🎉</td></tr>`;
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
setInterval(tick, REFRESH);   // page updates itself, no reload
</script>
</body>
</html>
"""
