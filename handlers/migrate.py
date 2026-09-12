"""
Administrative Lambda: schema migrations, channel seeding, and backfill
of ingest_jobs from a pre-existing `videos` table.
Invoke with e.g.:
    {"action": "migrate"}
    {"action": "seed_channels"}
    {"action": "backfill_jobs"}
    {"action": "enqueue_pending", "limit": 500}
    {"action": "all"}
"""
import json
import os
import socket
import time
from pathlib import Path
from common.aws import client
from common.db import get_conn
from common.logging_utils import get_logger
from datetime import datetime, timezone
from common import dispatch
from common.config import settings
log = get_logger("migrate")
MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"
# Arbitrary constant; pg_advisory_lock key so two concurrent invocations
# (e.g. deploy script + manual invoke) can't interleave DDL.
ADVISORY_LOCK_KEY = 744_211_987
HEAVY_ROW_THRESHOLD = 2_000_000

# Every (table, columns) that some handler targets with ON CONFLICT.
UPSERT_KEYS = [
    ("channels", ["channel_id"]),
    ("users", ["user_id"]),
    ("videos", ["video_id"]),
    ("ingest_jobs", ["video_id"]),
    ("channel_watermarks", ["channel_id"]),
    ("service_config", ["key"]),
    ("schema_migrations", ["filename"]),
    ("video_stream_stats", ["video_id"]),
    ("monthly_merge_state", ["observed_month"]),
    ("membership_data_summary",
     ["channel_name", "observed_month", "membership_rank"]),
    ("user_data", ["user_id", "channel_id", "last_message_at", "video_id"]),
    ("user_data_current",
     ["user_id", "channel_id", "last_message_at", "video_id"]),
]

RESTORE_HISTORY_DDL = """
CREATE TABLE IF NOT EXISTS restore_history (
    id          SERIAL PRIMARY KEY,
    source      TEXT,
    sha256      TEXT,
    dbname      TEXT,
    restored_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    row_counts  JSONB
)"""
_PROBE_TABLES = ("videos", "user_data", "user_data_current", "users",
                 "channels", "ingest_jobs")

def handler(event, context):
    action = (event or {}).get("action", "migrate")
    result = {"action": action}

    if action == "ping":
        return ping()
    if action in ("migrate", "all"):
        budget = (context.get_remaining_time_in_millis()
                  if context and hasattr(context, "get_remaining_time_in_millis")
                  else None)
        result["migrations"] = apply_migrations(
            budget_ms=budget, force=bool((event or {}).get("force")))
    if action in ("seed_channels", "all"):
        result["channels"] = seed_channels()
    if action in ("backfill_jobs", "all"):
        result["backfill"] = backfill_jobs()
    if action == "enqueue_pending":
        result["enqueued"] = enqueue_pending(
            limit=int((event or {}).get("limit", 10_000)),
            since=(event or {}).get("since"),
            include_history=bool((event or {}).get("include_history")))
    if action == "enqueue_video":
        return enqueue_video(event)
    if action == "job_status":
        return job_status(event["video_id"])  
    if action == "verify_schema":
        return verify_schema()
    if action == "repair_end_times":
        return repair_end_times(limit=int((event or {}).get("limit", 200)))
    if action == "restore_state":
        return restore_state()
    if action == "mark_restored":
        return mark_restored(event)
    if action == "reset_migration_log":
        return reset_migration_log()
    if action == "set_config":
        return set_config(event["key"], str(event["value"]))
    if action == "retry_cookie_failures":
        return retry_cookie_failures()
    if action == "drain_retry_queue":
        return drain_retry_queue()
    if action == "retire_backlog":
        return retire_backlog(event)
    if action == "dispatch":        
        return dispatch.run(settings(force=True))
    if action == "refresh_views":
        return refresh_views()
    if len(result) == 1:
        raise ValueError(f"unknown action: {action!r}")
    log.info("migrate handler finished", extra=result)
    return result

def _config(conn, key, default=None):
    with conn.cursor() as cur:
        cur.execute("SELECT value FROM service_config WHERE key = %s", (key,))
        row = cur.fetchone()
    conn.rollback()
    return row[0] if row and row[0] else default

def _computed_floor(conn):
    """Month after the newest merged month; else the month of the newest
    ingested data. Mirrors common/dispatch._floor so retire and dispatch can
    never disagree about what 'too old' means."""
    with conn.cursor() as cur:
        cur.execute("""SELECT COALESCE(
            (SELECT (MAX(observed_month) + INTERVAL '1 month')
               FROM monthly_merge_state WHERE status = 'merged'),
            (SELECT date_trunc('month', MAX(v.end_time))
               FROM videos v JOIN ingest_jobs j USING (video_id)
              WHERE j.status = 'done'))""")
        row = cur.fetchone()[0]
    conn.rollback()
    return row


def retire_backlog(event):
    """
    Mark undispatched pending jobs for videos older than the floor as skipped.
    Idempotent and safe to run on every deploy: `dispatched_at IS NULL` means
    this can never cancel work already on the wire, and a job that is already
    skipped/done/failed is untouched.
    """
    conn = get_conn()
    floor = (event.get("floor")
             or _config(conn, "backlog_floor")
             or _computed_floor(conn))
    if not floor:
        return {"retired": 0, "note": "no floor: set backlog_floor via "
                                      "set_config, or pass floor="}
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE ingest_jobs j
            SET status = 'skipped',
                skip_reason = 'pre-floor backlog (' || %s::date || ')',
                completed_at = NOW(), updated_at = NOW()
            FROM videos v
            WHERE v.video_id = j.video_id
              AND j.status = 'pending'
              AND j.dispatched_at IS NULL
              AND v.end_time < %s::timestamptz
            RETURNING j.video_id""", (str(floor), str(floor)))
        n = cur.rowcount
    conn.commit()
    log.info("backlog retired", extra={"retired": n, "floor": str(floor)})
    return {"retired": n, "floor": str(floor)}

def refresh_views():
    """A never-populated matview cannot be refreshed CONCURRENTLY."""
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("""SELECT c.relname, c.relispopulated
                       FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
                       WHERE c.relkind = 'm' AND n.nspname = 'public'
                       ORDER BY c.relname""")
        mvs = cur.fetchall()
    conn.rollback()
    out = []
    for name, populated in mvs:
        with conn.cursor() as cur:
            cur.execute(f'REFRESH MATERIALIZED VIEW '
                        f'{"CONCURRENTLY " if populated else ""}"{name}"')
        conn.commit()
        out.append({"view": name, "concurrently": populated})
    return {"refreshed": out}

def _row_count(cur, table):
    cur.execute("SELECT GREATEST(reltuples, 0)::bigint FROM pg_class "
                "WHERE oid = to_regclass(%s)", (table,))
    est = (cur.fetchone() or [None])[0]
    if est is None:
        return None
    if est > 5_000_000:                       # exact COUNT(*) is a seq scan
        return {"estimate": int(est)}
    cur.execute(f"SELECT COUNT(*) FROM {table}")
    return cur.fetchone()[0]

def restore_state():
    """
    Has this database already been restored into?
    Two independent signals, because either can be missing:
      * restore_history -- written by the deploy after a successful pg_restore;
      * actual rows     -- protects a database that was populated some other way
                           (a manual pg_restore, a promoted snapshot) from being
                           silently overwritten.
    Row counts are estimates from pg_class: this runs on every deploy and must
    not sequential-scan user_data.
    """
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(RESTORE_HISTORY_DDL)
    conn.commit()
    out = {"history": [], "row_estimates": {}, "non_empty": []}
    with conn.cursor() as cur:
        cur.execute("""SELECT source, sha256, dbname, restored_at
                       FROM restore_history ORDER BY id DESC LIMIT 5""")
        out["history"] = [{"source": r[0], "sha256": r[1], "dbname": r[2],
                           "restored_at": str(r[3])} for r in cur.fetchall()]
        for t in _PROBE_TABLES:
            cur.execute("SELECT to_regclass(%s)", (t,))
            if cur.fetchone()[0] is None:
                continue
            cur.execute("""SELECT GREATEST(reltuples::bigint, 0)
                           FROM pg_class WHERE oid = to_regclass(%s)""", (t,))
            out["row_estimates"][t] = cur.fetchone()[0]
            cur.execute(f"SELECT EXISTS (SELECT 1 FROM {t} LIMIT 1)")
            if cur.fetchone()[0]:
                out["non_empty"].append(t)
    conn.rollback()
    out["restored"] = bool(out["history"])
    out["has_data"] = any(t in out["non_empty"]
                          for t in ("videos", "user_data", "users"))
    return out
def mark_restored(event):
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(RESTORE_HISTORY_DDL)
        counts = {}
        for t in _PROBE_TABLES:
            counts[t] = _row_count(cur, t)
        cur.execute("""INSERT INTO restore_history (source, sha256, dbname,
                                                    row_counts)
                       VALUES (%s, %s, %s, %s::jsonb) RETURNING id""",
                    (event.get("source"), event.get("sha256"),
                     event.get("dbname"), json.dumps(counts)))
        rid = cur.fetchone()[0]
    conn.commit()
    log.info("restore recorded", extra={"restore_id": rid, "row_counts": counts})
    return {"restore_id": rid, "row_counts": counts}
def reset_migration_log():
    """
    A restored dump may carry its own schema_migrations rows, which would make
    this deployment skip migrations it has never actually applied. The
    consolidated files are convergent, so clearing the log and re-running them
    is always safe -- and is the only way to guarantee the schema matches the
    code after a restore.
    """
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('schema_migrations')")
        if cur.fetchone()[0] is None:
            conn.rollback()
            return {"cleared": 0, "note": "table absent"}
        cur.execute("DELETE FROM schema_migrations RETURNING filename")
        cleared = [r[0] for r in cur.fetchall()]
    conn.commit()
    log.warning("migration log cleared after restore",
                extra={"cleared": cleared})
    return {"cleared": len(cleared), "filenames": cleared}
def set_config(key, value):
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("""INSERT INTO service_config (key, value, updated_at)
                       VALUES (%s, %s, NOW())
                       ON CONFLICT (key) DO UPDATE
                         SET value = EXCLUDED.value, updated_at = NOW()""",
                    (key, value))
    conn.commit()
    return {key: value}

def retry_cookie_failures():
    """Requeue terminal downloads fixed by auth/response-reader updates."""
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE ingest_jobs
            SET status='pending', attempts=0,
                continuation = CASE
                  WHEN last_error ILIKE '%%unterminated string%%'
                    OR last_error ILIKE '%%JSONDecodeError%%'
                    OR last_error ILIKE '%%400 Client Error%%'
                    OR last_error ILIKE '%%Bad Request%%'
                  THEN NULL ELSE continuation END,
                part_count = CASE
                  WHEN last_error ILIKE '%%unterminated string%%'
                    OR last_error ILIKE '%%JSONDecodeError%%'
                    OR last_error ILIKE '%%400 Client Error%%'
                    OR last_error ILIKE '%%Bad Request%%'
                  THEN 0 ELSE part_count END,
                last_offset_s = CASE
                  WHEN last_error ILIKE '%%unterminated string%%'
                    OR last_error ILIKE '%%JSONDecodeError%%'
                    OR last_error ILIKE '%%400 Client Error%%'
                    OR last_error ILIKE '%%Bad Request%%'
                  THEN 0 ELSE last_offset_s END,
                messages_downloaded = CASE
                  WHEN last_error ILIKE '%%unterminated string%%'
                    OR last_error ILIKE '%%JSONDecodeError%%'
                    OR last_error ILIKE '%%400 Client Error%%'
                    OR last_error ILIKE '%%Bad Request%%'
                  THEN 0 ELSE messages_downloaded END,
                last_error=NULL,
                completed_at=NULL, lease_id=NULL, updated_at=NOW()
            WHERE status='failed'
              AND (last_error ILIKE '%%sign in to confirm%%not a bot%%'
                   OR last_error ILIKE '%%cookie%%'
                   OR last_error ILIKE '%%unterminated string%%'
                   OR last_error ILIKE '%%JSONDecodeError%%'
                   OR last_error ILIKE '%%503 Server Error%%'
                   OR last_error ILIKE '%%Service Unavailable%%'
                   OR last_error ILIKE '%%truncated: last message%%'
                   OR last_error ILIKE '%%live event%%'
                   OR last_error ILIKE '%%will begin%%'
                   OR last_error ILIKE '%%age-restricted%%'
                   OR last_error ILIKE '%%age restricted%%'
                   OR last_error ILIKE '%%confirm your age%%'
                   OR last_error ILIKE '%%page needs to be reloaded%%'
                   OR last_error ILIKE '%%400 Client Error%%'
                   OR last_error ILIKE '%%Bad Request%%')
            RETURNING video_id, channel_id
        """)
        rows = cur.fetchall()
    conn.commit()
    sqs = client("sqs")
    queue = os.environ["DOWNLOAD_QUEUE_URL"]
    for video_id, channel_id in rows:
        sqs.send_message(QueueUrl=queue, MessageBody=json.dumps({
            "video_id": video_id, "channel_id": channel_id,
            "attempt": 0, "source": "retry"}))
    log.info("retryable YouTube failures requeued", extra={"count": len(rows)})
    return {"requeued": len(rows)}

def drain_retry_queue(limit=10_000):
    """Move messages left by the former two-lane design to the main queue.

    Floci can let the main ESM monopolize a function with reserved concurrency
    one, permanently starving the retry ESM. This operation is idempotent at
    the job layer: duplicate messages are rejected by the row-state claim.
    """
    source = os.environ.get("DOWNLOAD_RETRY_QUEUE_URL")
    target = os.environ["DOWNLOAD_QUEUE_URL"]
    if not source or source == target:
        return {"moved": 0}
    sqs, moved, empty_polls = client("sqs"), 0, 0
    while moved < limit and empty_polls < 3:
        response = sqs.receive_message(QueueUrl=source, MaxNumberOfMessages=10,
                                       WaitTimeSeconds=0,
                                       VisibilityTimeout=60)
        messages = response.get("Messages", [])
        if not messages:
            empty_polls += 1
            time.sleep(.2)
            continue
        empty_polls = 0
        for message in messages:
            body = json.loads(message["Body"])
            body["source"] = "retry"
            sqs.send_message(QueueUrl=target, MessageBody=json.dumps(body))
            sqs.delete_message(QueueUrl=source,
                               ReceiptHandle=message["ReceiptHandle"])
            moved += 1
            if moved >= limit:
                break
    log.info("legacy retry queue drained", extra={"moved": moved})
    return {"moved": moved}

def verify_schema():
    """
    Reports constraint drift. CREATE TABLE IF NOT EXISTS cannot detect it: a
    restored database has the tables but may lack their keys, and the failure
    only surfaces at the first ON CONFLICT, mid-ingest.
    """
    conn = get_conn()
    missing, present, unknown = [], [], []
    with conn.cursor() as cur:
        cur.execute("SELECT to_regproc('has_unique_key') IS NOT NULL")
        have_fn = cur.fetchone()[0]
        for table, cols in UPSERT_KEYS:
            cur.execute("SELECT to_regclass(%s) IS NOT NULL", (table,))
            exists = cur.fetchone()[0]
            entry = {"table": table, "key": cols, "exists": exists}
            if not exists:
                missing.append({**entry, "reason": "table absent"})
                continue
            if not have_fn:
                unknown.append(entry)
                continue
            cur.execute("SELECT has_unique_key(%s, %s::text[])", (table, cols))
            (present if cur.fetchone()[0] else missing).append(entry)
        cur.execute("""SELECT COUNT(*) FILTER (WHERE end_time IS NULL),
                              COUNT(*) FILTER (WHERE channel_id IS NULL),
                              COUNT(*) FROM videos""")
        no_end, no_chan, total = cur.fetchone()
    conn.rollback()
    out = {"keys_ok": [e["table"] for e in present],
           "keys_missing": missing,
           "videos": {"total": total, "null_end_time": no_end,
                      "null_channel_id": no_chan}}
    if unknown:
        out["keys_unknown"] = [e["table"] for e in unknown]
        out["remedy"] = 'run {"action":"migrate"} first -- 012 installs has_unique_key()'
    elif missing:
        out["remedy"] = ('run {"action":"migrate"} -- 012_repair_keys.sql adds '
                         'the missing keys (and removes unreachable null-key / '
                         'duplicate rows)')
    if no_end:
        out["remedy_end_time"] = ('run {"action":"repair_end_times"} to resolve '
                                  'them from YouTube')
    log.info("schema verification", extra=out)
    return out

def repair_end_times(limit=200):
    """
    Fill videos.end_time where it is NULL, from YouTube. Not guessed from
    processed_at: end_time drives month bucketing (handlers/merge.py) and the
    discovery watermark, so a wrong value silently misfiles a whole month.
    """
    from chat_downloader import sites
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("""SELECT video_id FROM videos WHERE end_time IS NULL
                       ORDER BY processed_at NULLS LAST LIMIT %s""", (limit,))
        ids = [r[0] for r in cur.fetchall()]
    conn.rollback()
    cd = sites.YouTubeChatDownloader()
    fixed, failed = 0, []
    for video_id in ids:
        try:
            data = cd.get_video_data(video_id=video_id)
            raw = data.get("end_time")
            if not raw:
                failed.append({"video_id": video_id,
                               "reason": "no end_time (still live or upload)"})
                continue
            end = datetime.fromtimestamp(raw / 1_000_000, timezone.utc)
            with conn.cursor() as cur:
                cur.execute("""UPDATE videos SET end_time = %s
                               WHERE video_id = %s AND end_time IS NULL""",
                            (end, video_id))
            conn.commit()
            fixed += 1
        except Exception as e:
            conn.rollback()
            failed.append({"video_id": video_id, "reason": str(e)[:160]})
    log.info("end_time repair", extra={"fixed": fixed, "failed": len(failed)})
    return {"candidates": len(ids), "fixed": fixed, "failed": failed[:20]}


# --------------------------------------------------------------------------- #
# schema migrations
# --------------------------------------------------------------------------- #
def apply_migrations(budget_ms=None, force=False):
    conn = get_conn()
    applied = []
    cur = conn.cursor()
    # SET does not accept bind parameters; set_config() does.
    cur.execute("SELECT set_config('lock_timeout', '5s', false)")
    if budget_ms:
        # Surface a Postgres cancellation ~15 s before the sandbox is killed.
        # Otherwise you get an opaque Function.TimedOut and no idea which
        # statement was running.
        cur.execute("SELECT set_config('statement_timeout', %s, false)",
                    (f"{max(5_000, int(budget_ms) - 15_000)}ms",))
    cur.execute("""CREATE TABLE IF NOT EXISTS schema_migrations (
                       filename   TEXT PRIMARY KEY,
                       applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW())""")
    conn.commit()
    cur.execute("SELECT filename FROM schema_migrations")
    done = {r[0] for r in cur.fetchall()}
    pending = [p for p in sorted(MIGRATIONS_DIR.glob("*.sql"))
               if p.name not in done]
    conn.rollback()
    if not pending:
        return {"applied": [], "already_applied": sorted(done)}
    if not force:
        cur.execute("""SELECT COALESCE(GREATEST(
              (SELECT reltuples FROM pg_class WHERE oid = to_regclass('user_data')),
              (SELECT reltuples FROM pg_class WHERE oid = to_regclass('user_data_current'))
            ), 0)::bigint""")
        rows = cur.fetchone()[0]
        conn.rollback()
        if rows > HEAVY_ROW_THRESHOLD:
            return {
                "refused": "pending migrations against a large database",
                "estimated_rows": rows, "pending": [p.name for p in pending],
                "why": ("building primary keys, ~15 indexes and four "
                        "materialized views over this table does not fit in "
                        "Lambda's 900 s ceiling"),
                "remedy": ("python infra/deploy.py --code-only "
                           "--schema-mode docker --endpoint http://localhost:4566"),
                "override": 'invoke again with {"action":"migrate","force":true}',
            }
    # pg_try_advisory_lock, not pg_advisory_lock: a timed-out invocation leaves
    # an orphaned backend holding a SESSION lock until TCP reaps it, and every
    # retry then blocks silently until *it* times out too.
    cur.execute("SELECT pg_try_advisory_lock(%s)", (ADVISORY_LOCK_KEY,))
    if not cur.fetchone()[0]:
        conn.rollback()
        return {"skipped": "another migration holds the advisory lock"}
    try:
        for path in pending:
            sql = path.read_text(encoding="utf-8")
            log.info("applying migration", extra={"file": path.name})
            try:
                cur.execute(sql)
                notices = [n.strip() for n in conn.notices]
                del conn.notices[:]
                cur.execute("INSERT INTO schema_migrations (filename) VALUES (%s)",
                            (path.name,))
                conn.commit()          # one txn per file; a failure stops here
                applied.append({"file": path.name, "notices": notices})
            except Exception:
                conn.rollback()
                log.exception("migration failed", extra={"file": path.name})
                raise
    finally:
        cur.execute("SELECT pg_advisory_unlock(%s)", (ADVISORY_LOCK_KEY,))
        conn.commit()
        cur.close()
    return {"applied": applied, "already_applied": sorted(done)}
# --------------------------------------------------------------------------- #
# channels.json  ->  channels table
# --------------------------------------------------------------------------- #
def seed_channels():
    """
    channels.json shape:
        { "Group1": { "Channel1": "YT_ID_1", ... }, "Group2": { ... } }
    Names (and groups) beginning with '_' are placeholders/comments and are
    skipped entirely -- they are never added to the channel list.
    Upserts name/group; never touches `active`, which the admin UI owns, and
    never deletes. Removing a channel from the file no longer stops discovery:
    deactivate it in the admin page instead.
    """
    s3 = client("s3")
    obj = s3.get_object(Bucket=os.environ["CONFIG_BUCKET"], Key="channels.json")
    data = json.loads(obj["Body"].read())
    rows, skipped = [], []
    for group, members in data.items():
        if group.startswith("_"):
            skipped.append(group)
            continue
        if not isinstance(members, dict):
            raise ValueError(f"channels.json: group {group!r} must map names to IDs")
        for name, channel_id in members.items():
            if name.startswith("_"):
                skipped.append(f"{group}/{name}")
                continue
            rows.append((channel_id, name, group))
    seen = {}
    for cid, name, group in rows:
        if cid in seen and seen[cid] != (name, group):
            raise ValueError(f"channels.json: duplicate channel_id {cid} "
                             f"({seen[cid]} vs {(name, group)})")
        seen[cid] = (name, group)
    conn = get_conn()
    with conn.cursor() as cur:
        cur.executemany("""
            INSERT INTO channels (channel_id, channel_name, channel_group)
            VALUES (%s, %s, %s)
            ON CONFLICT (channel_id) DO UPDATE
              SET channel_name  = EXCLUDED.channel_name,
                  channel_group = EXCLUDED.channel_group,
                  updated_at    = NOW()
        """, rows)
    conn.commit()
    if skipped:
        log.info("skipped underscore-prefixed entries", extra={"skipped": skipped})
    return {"upserted": len(rows), "groups": len(data) - len([g for g in skipped if "/" not in g]),
            "skipped": skipped}
# --------------------------------------------------------------------------- #
# old-world videos  ->  ingest_jobs
# --------------------------------------------------------------------------- #
def backfill_jobs():
    """
    Idempotent: ON CONFLICT DO NOTHING, so re-running after a partial
    restore is safe. Videos with has_chat_log=TRUE become 'done' (their
    user_data already exists from the old pipeline); the rest become
    'pending' but are NOT enqueued automatically -- run enqueue_pending
    explicitly once you've set ingest_start_date, so a restored DB doesn't
    instantly trigger thousands of downloads.
    """
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO ingest_jobs
                (video_id, channel_id, status, video_duration_s, s3_prefix,
                 completed_at, enqueued_at)
            SELECT
                v.video_id,
                v.channel_id,
                CASE WHEN v.has_chat_log THEN 'done' ELSE 'pending' END,
                EXTRACT(EPOCH FROM v.duration),
                v.channel_id || '/' || v.video_id,
                CASE WHEN v.has_chat_log THEN v.processed_at END,
                NOW()
            FROM videos v
            JOIN channels c ON c.channel_id = v.channel_id AND c.active
            ON CONFLICT (video_id) DO NOTHING
            RETURNING status
        """)
        inserted = [r[0] for r in cur.fetchall()]
        # Seed watermarks from existing data so discovery doesn't re-walk
        # every channel's entire history on its first run.
        cur.execute("""
            INSERT INTO channel_watermarks (channel_id, last_seen_end_time)
            -- Only PROCESSED videos may advance the watermark. A video whose
            -- metadata we have but whose chat we have not ingested is exactly
            -- the work discovery must still find.
            SELECT v.channel_id, MAX(v.end_time)
            FROM videos v LEFT JOIN ingest_jobs j USING (video_id)
            WHERE v.channel_id IS NOT NULL
              AND (v.has_chat_log OR j.status IN ('done','skipped'))
            GROUP BY v.channel_id
             ON CONFLICT (channel_id) DO UPDATE
               SET last_seen_end_time = GREATEST(
                       channel_watermarks.last_seen_end_time,
                       EXCLUDED.last_seen_end_time)
        """)
    conn.commit()
    return {"inserted": len(inserted),
            "done": inserted.count("done"),
            "pending": inserted.count("pending")}
def enqueue_pending(limit=10_000, since=None, include_history=False):
    """
    Push pending jobs onto the download queue.
    Default: per channel, only videos NEWER than that channel's most recent
    already-ingested (or deliberately skipped) video -- the same baseline
    handlers/scan.py uses. This is what keeps a restored database from
    enqueueing two years of downloads.
    {"action": "enqueue_pending", "include_history": true}   -> ignore the
        per-channel baseline and fall back to service_config.ingest_start_date
        (or "since": "YYYY-MM-DD") as an absolute floor.
    """
    conn = get_conn()
    sqs = client("sqs")
    queue_url = os.environ["DOWNLOAD_QUEUE_URL"]
    with conn.cursor() as cur:
        if since is None:
            cur.execute("SELECT value FROM service_config "
                        "WHERE key = 'ingest_start_date'")
            since = cur.fetchone()[0]
        cur.execute("""
            WITH baseline AS (
                SELECT v.channel_id,
                       MAX(v.end_time) FILTER (
                           WHERE v.has_chat_log OR j.status IN ('done','skipped')
                       ) AS processed_through
                FROM videos v
                LEFT JOIN ingest_jobs j USING (video_id)
                GROUP BY v.channel_id
            )
            SELECT j.video_id, j.channel_id
            FROM ingest_jobs j
            JOIN videos v USING (video_id)
            JOIN channels c ON c.channel_id = v.channel_id AND c.active
            LEFT JOIN baseline b ON b.channel_id = v.channel_id
            WHERE j.status = 'pending'
              AND v.end_time >= %s::timestamptz
              AND (%s OR v.end_time > COALESCE(b.processed_through,
                                               %s::timestamptz))
            ORDER BY v.end_time
            LIMIT %s
        """, (since, include_history, since, limit))
        jobs = cur.fetchall()
    conn.rollback()
    sent, batch = 0, []
    for i, (video_id, channel_id) in enumerate(jobs):
        batch.append({"Id": str(i % 10),
                      "MessageBody": json.dumps({"video_id": video_id,
                                                 "channel_id": channel_id,
                                                 "attempt": 0})})
        if len(batch) == 10:
            sqs.send_message_batch(QueueUrl=queue_url, Entries=batch)
            sent += len(batch)
            batch = []
    if batch:
        sqs.send_message_batch(QueueUrl=queue_url, Entries=batch)
        sent += len(batch)
    log.info("enqueued pending jobs",
             extra={"count": sent, "floor": str(since),
                    "include_history": include_history})
    return sent

def ping():
    """
    Connectivity diagnostic. Never raises: every stage is reported with its
    own timing so you can see exactly which hop is broken.
    """
    from common.aws import endpoint
    out = {
        "endpoint": endpoint(),
        "env": {k: os.environ.get(k) for k in
                ("INGEST_ENDPOINT_URL", "AWS_ENDPOINT_URL", "AWS_REGION",
                 "DB_SECRET_ID", "CONFIG_BUCKET", "RAW_BUCKET",
                 "DOWNLOAD_QUEUE_URL", "REDIS_HOST", "REDIS_PORT",
                 "APP_NAME")},
        "stages": {},
    }
    def stage(name, fn):
        t0 = time.time()
        try:
            out["stages"][name] = {"ok": True, "detail": fn(),
                                   "ms": int((time.time() - t0) * 1000)}
            return True
        except Exception as e:
            out["stages"][name] = {"ok": False,
                                   "error": f"{type(e).__name__}: {e}"[:400],
                                   "ms": int((time.time() - t0) * 1000)}
            return False
    def _secret():
        from common.config import secret
        c = secret(os.environ["DB_SECRET_ID"])
        return {"host": c["host"], "port": c.get("port", 5432),
                "dbname": c["dbname"], "username": c["username"]}
    if not stage("secretsmanager", _secret):
        out["diagnosis"] = ("cannot reach Secrets Manager from inside the "
                            "lambda container -- check INGEST_ENDPOINT_URL "
                            "(--lambda-endpoint) and container networking")
        return out
    creds = out["stages"]["secretsmanager"]["detail"]
    stage("dns", lambda: socket.gethostbyname(creds["host"]))
    stage("tcp", lambda: _tcp_probe(creds["host"], int(creds["port"])))
    def _db():
        from common.db import get_conn
        conn = get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT current_database(), version()")
            db, ver = cur.fetchone()
            cur.execute("""SELECT EXISTS (SELECT 1 FROM information_schema.tables
                           WHERE table_name = 'schema_migrations')""")
            migrated = cur.fetchone()[0]
        conn.rollback()
        return {"database": db, "version": ver.split(",")[0],
                "schema_migrations_exists": migrated}
    stage("postgres", _db)
    def _redis():
        import redis
        host = os.environ.get("REDIS_HOST") or os.environ.get("ELASTICACHE_HOST")
        port = int(os.environ.get("REDIS_PORT")
                   or os.environ.get("ELASTICACHE_PORT", "6379"))
        if not host:
            raise RuntimeError("REDIS_HOST/ELASTICACHE_HOST is not configured")
        store = redis.Redis(host=host, port=port, socket_connect_timeout=1,
                            socket_timeout=1, retry_on_timeout=False)
        return {"host": host, "port": port, "pong": bool(store.ping())}
    stage("redis", _redis)
    stage("s3", lambda: len(client("s3").list_objects_v2(
        Bucket=os.environ["CONFIG_BUCKET"]).get("Contents", [])))
    stage("sqs", lambda: client("sqs").get_queue_attributes(
        QueueUrl=os.environ["DOWNLOAD_QUEUE_URL"],
        AttributeNames=["ApproximateNumberOfMessages"])["Attributes"])
    if not out["stages"].get("dns", {}).get("ok"):
        out["diagnosis"] = (f"db host {creds['host']!r} does not resolve inside "
                            f"the lambda container -- use the compose service "
                            f"name or the emulator's RDS endpoint, not localhost")
    elif not out["stages"].get("tcp", {}).get("ok"):
        out["diagnosis"] = (f"{creds['host']}:{creds['port']} resolves but "
                            f"refuses connections -- wrong port, or the lambda "
                            f"containers are on a different docker network")
    elif not out["stages"].get("redis", {}).get("ok"):
        out["diagnosis"] = (
            f"ElastiCache is unavailable from Lambda at "
            f"{os.environ.get('REDIS_HOST')}:{os.environ.get('REDIS_PORT')}; "
            "for Floci this must be the floci-valkey-<replication-group> "
            "container DNS name, and that container must be attached to the "
            "Floci Compose network")
    elif all(s["ok"] for s in out["stages"].values()):
        out["diagnosis"] = "all dependencies reachable"
    return out
def _tcp_probe(host, port, timeout=3):
    s = socket.create_connection((host, port), timeout=timeout)
    s.close()
    return f"{host}:{port} reachable"

def enqueue_video(event):
    """
    Push one specific video through the real pipeline.
      {"action": "enqueue_video", "channel_id": "UC...", "video_id": "...",
       "force": true}          # force = reset a previous attempt and redo it
    """
    video_id = event["video_id"]
    channel_id = event["channel_id"]
    force = bool(event.get("force"))
    from chat_downloader import sites
    video_data = sites.YouTubeChatDownloader().get_video_data(video_id=video_id)
    if not video_data.get("continuation_info"):
        return {"error": "no chat replay available (not a concluded stream, "
                         "or chat replay disabled)", "video_id": video_id}
    end_raw = video_data.get("end_time")
    end_date = (datetime.fromtimestamp(end_raw / 1_000_000, timezone.utc)
                if end_raw else datetime.now(timezone.utc))
    duration = video_data.get("duration") or 0
    title = video_data.get("title") or video_id
    conn = get_conn()
    with conn.cursor() as cur:
        # The video's channel may not be in channels.json (FK on watermarks
        # isn't involved here, but joins in the views expect the row).
        cur.execute("""
            INSERT INTO channels (channel_id, channel_name, channel_group)
            VALUES (%s, %s, %s)
            ON CONFLICT (channel_id) DO NOTHING
        """, (channel_id, event.get("channel_name", channel_id),
              event.get("channel_group", "Test")))
        cur.execute("SELECT active, channel_name FROM channels WHERE channel_id = %s",
                    (channel_id,))
        active, cname = cur.fetchone()
        if not active:
            conn.rollback()
            return {"error": "channel is inactive (or underscore-prefixed); "
                             "re-add it in the admin page first",
                    "channel_id": channel_id, "channel_name": cname}
        cur.execute("""
            INSERT INTO videos (video_id, channel_id, title, end_time, duration,
                                processed_at, has_chat_log)
            VALUES (%s, %s, %s, %s, make_interval(secs => %s), NOW(), FALSE)
            ON CONFLICT (video_id) DO UPDATE
              SET title = EXCLUDED.title, end_time = EXCLUDED.end_time,
                  duration = EXCLUDED.duration, processed_at = NOW()
        """, (video_id, channel_id, title, end_date, duration))
        if force:
            cur.execute("DELETE FROM ingest_jobs WHERE video_id = %s", (video_id,))
        cur.execute("""
            INSERT INTO ingest_jobs (video_id, channel_id, status,
                                     video_duration_s, s3_prefix)
            VALUES (%s, %s, 'pending', %s, %s)
            ON CONFLICT (video_id) DO NOTHING
            RETURNING video_id
        """, (video_id, channel_id, duration, f"{channel_id}/{video_id}"))
        claimed = cur.fetchone() is not None
    conn.commit()
    if not claimed:
        # A job already exists and force wasn't given -- report, don't enqueue.
        return {"video_id": video_id, "enqueued": False,
                "note": "job already exists; pass force=true to redo",
                **job_status(video_id)}
    if force:
        _purge_raw_parts(channel_id, video_id)
    client("sqs").send_message(
        QueueUrl=os.environ["DOWNLOAD_QUEUE_URL"],
        MessageBody=json.dumps({"video_id": video_id,
                                "channel_id": channel_id,
                                "attempt": 0}))
    log.info("test video enqueued", extra={"video_id": video_id,
                                           "channel_id": channel_id,
                                           "duration_s": duration})
    return {"video_id": video_id, "enqueued": True, "title": title,
            "duration_s": duration, "end_time": end_date.isoformat()}
def job_status(video_id):
    """Progress snapshot for one video: the poll target for the test script."""
    conn = get_conn()
    out = {"video_id": video_id}
    with conn.cursor() as cur:
        cur.execute("""
            SELECT status, attempts, part_count, last_offset_s,
                   video_duration_s, message_count, last_error, skip_reason,
                   enqueued_at, started_at, completed_at, updated_at
            FROM ingest_jobs WHERE video_id = %s
        """, (video_id,))
        row = cur.fetchone()
        if row is None:
            conn.rollback()
            return {**out, "status": "unknown",
                    "note": "no ingest_jobs row -- enqueue_video first"}
        cols = ("status", "attempts", "part_count", "last_offset_s",
                "video_duration_s", "message_count", "last_error",
                "skip_reason", "enqueued_at", "started_at", "completed_at",
                "updated_at")
        out.update({k: (str(v) if k.endswith("_at") and v else v)
                    for k, v in zip(cols, row)})
        if out["video_duration_s"] and out["last_offset_s"]:
            out["download_pct"] = round(
                min(100.0, 100.0 * out["last_offset_s"] / out["video_duration_s"]), 1)
        cur.execute("""SELECT has_chat_log, funniest_timestamp, title
                       FROM videos WHERE video_id = %s""", (video_id,))
        v = cur.fetchone()
        if v:
            out["has_chat_log"], out["funniest_timestamp"], out["title"] = v
        cur.execute("""SELECT COUNT(*), COALESCE(SUM(total_message_count), 0)
                       FROM user_data WHERE video_id = %s""", (video_id,))
        out["user_data_rows"], out["messages_in_db"] = cur.fetchone()
    conn.rollback()
    return out
def _purge_raw_parts(channel_id, video_id):
    s3 = client("s3")
    bucket = os.environ["RAW_BUCKET"]
    prefix = f"{channel_id}/{video_id}/"
    while True:
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        keys = [{"Key": o["Key"]} for o in resp.get("Contents", [])]
        if not keys:
            return
        s3.delete_objects(Bucket=bucket, Delete={"Objects": keys})
        if not resp.get("IsTruncated"):
            return
