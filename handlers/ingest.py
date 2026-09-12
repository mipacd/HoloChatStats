import gzip, json, os
from collections import defaultdict
from datetime import datetime, timezone
from psycopg2.extras import execute_values
from common.aws import client
from common.chat_parser import categorize_message, parse_membership_rank
from common.config import setting
from common.feature_analysis import has_humor, get_feature_timestamps
from common.db import get_conn
from common.logging_utils import get_logger
from common.metrics import emit, COUNT
from common.control import paused, requeue_all
from common.channels import is_active, cancel_job
from common.month_order import work_months
from common.stream_stats import StreamStatsAccumulator


log = get_logger("ingest")
BUCKET = os.environ["RAW_BUCKET"]
FLUSH_EVERY = 5000
MAIN_TABLE = "user_data"
STAGING_TABLE = "user_data_current"
_COLS = ("user_id", "channel_id", "last_message_at", "video_id",
         "membership_rank", "jp_count", "kr_count", "ru_count", "emoji_count",
         "es_en_id_count", "total_message_count", "is_gift")
_TMPL = "(%s,%s,%s::timestamptz,%s,%s,%s,%s,%s,%s,%s,%s,%s"
_UPDATE = """
        membership_rank = COALESCE(EXCLUDED.membership_rank, {t}.membership_rank),
        jp_count = EXCLUDED.jp_count, kr_count = EXCLUDED.kr_count,
        ru_count = EXCLUDED.ru_count, emoji_count = EXCLUDED.emoji_count,
        es_en_id_count = EXCLUDED.es_en_id_count,
        total_message_count = EXCLUDED.total_message_count,
        is_gift = EXCLUDED.is_gift"""

class RawPartCorrupt(RuntimeError):
    """A committed download part cannot be decoded and must be rebuilt."""

def handler(event, context):
    if paused():
        requeue_all("INGEST_QUEUE_URL", event["Records"])
        return {"paused": True}
    for record in event["Records"]:
        message = json.loads(record["body"])
        if message.get("action") == "backfill_stream_stats":
            _backfill_stream_stats(message)
        else:
            _ingest(message)
    return {"ok": True}
def _iter_messages(s3, channel_id, video_id, part_count):
    for part in range(part_count):
        key = f"{channel_id}/{video_id}/part-{part:05d}.jsonl.gz"
        body = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
        try:
            lines = gzip.decompress(body).decode().splitlines()
        except (gzip.BadGzipFile, EOFError, UnicodeDecodeError) as exc:
            # Do not include the line contents: chat text is user data and can
            # be large. The object key is sufficient diagnostics.
            raise RawPartCorrupt(f"corrupt raw chat part {key}: {exc}") from exc
        malformed = 0
        for line_number, line in enumerate(lines, 1):
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                # One incomplete YouTube response must not discard the rest of
                # an otherwise valid gzip part. Never log the line itself.
                malformed += 1
                log.warning("malformed raw chat message skipped",
                            extra={"video_id": video_id, "part": part,
                                   "line": line_number,
                                   "error": str(exc)[:160]})
        if malformed:
            emit({"MalformedRawMessagesSkipped": (malformed, COUNT)},
                 {"Stage": "ingest"}, video_id=video_id, part=part)
def _month_of_ts(ts):
    return datetime.fromtimestamp(ts or 0, timezone.utc).date().replace(day=1)
def _route(cur, video_id, fallback_ts):
    """
    Which table do this video's rows belong in?
    A month stays in `user_data_current` until handlers/merge.py has merged it
    -- that is the current month, plus the tail of the previous month while its
    last streams are still draining through the queues. Once merged, late
    re-ingests of that month go straight to user_data.
    Runs inside the write transaction on purpose: the routing decision and the
    rows it produces must commit together.
    """
    cur.execute("""SELECT date_trunc('month', end_time AT TIME ZONE 'UTC')::date
                   FROM videos WHERE video_id = %s""", (video_id,))
    row = cur.fetchone()
    month = row[0] if row and row[0] else _month_of_ts(fallback_ts)
    if setting("current_month_staging", "true").lower() != "true":
        return MAIN_TABLE, month
    cur.execute("""SELECT status FROM monthly_merge_state
                   WHERE observed_month = %s FOR SHARE""", (month,))
    state = cur.fetchone()
    merged = bool(state and state[0] == "merged")
    return (MAIN_TABLE if merged else STAGING_TABLE), month

def _ingest(msg):
    video_id, channel_id = msg["video_id"], msg["channel_id"]
    if not is_active(channel_id):
        # Abort before writing anything, so there is no partial month to undo.
        cancel_job(video_id)
        log.warning("channel inactive; cancelling ingest",
                    extra={"video_id": video_id, "channel_id": channel_id})
        emit({"JobsCancelled": (1, COUNT)}, {"Stage": "ingest"}, video_id=video_id)
        return
    conn, s3 = get_conn(), client("s3")
    video_month, active_month = work_months(conn, video_id)
    if video_month and active_month and video_month > active_month:
        # Keep the completed raw parts and retry ingestion later. This closes
        # the second route by which a stale future-month queue message could
        # become visible before the active month is published.
        client("sqs").send_message(
            QueueUrl=os.environ["INGEST_QUEUE_URL"],
            MessageBody=json.dumps(msg), DelaySeconds=900)
        log.warning("future-month ingest deferred",
                    extra={"video_id": video_id,
                           "video_month": str(video_month),
                           "active_month": str(active_month)})
        emit({"IngestDeferredByMonth": (1, COUNT)}, {"Stage": "ingest"},
             video_id=video_id, active_month=str(active_month))
        conn.close()
        return
    with conn.cursor() as cur:
        cur.execute("""UPDATE ingest_jobs SET status='ingesting', updated_at=NOW()
                       WHERE video_id=%s AND status = 'downloaded'
                       RETURNING part_count""", (video_id,))
        row = cur.fetchone()
    conn.commit()
    if row is None:
        log.info("not ready / already ingested", extra={"video_id": video_id})
        conn.close()
        return
    part_count = row[0]
    duration, stream_start_ts = _video_timing(conn, video_id)
    stream_stats = StreamStatsAccumulator(duration, stream_start_ts)
    cat_by_user = defaultdict(lambda: defaultdict(int))
    rank_map, chat_counts, last_at, usernames = {}, defaultdict(int), defaultdict(float), {}
    gift_only, known_rank = set(), set()
    humor, last_ts, total = [], 0.0, 0
    try:
        for m in _iter_messages(s3, channel_id, video_id, part_count):
            total += 1
            stream_stats.add(m)
            a = m["author"]
            uid, uname, badges = a.get("id"), a.get("name"), a.get("badges")
            ts, mtype = m["timestamp"], m.get("message_type", "chat")
            text = m.get("message")
            if not uid:
                continue
            usernames[uid] = uname
            last_ts = max(last_ts, ts)
            if mtype in ("new_member", "gift_member"):
                if mtype == "gift_member" and not badges:
                    rank_map.setdefault(uid, -2)
                    gift_only.add(uid)
                else:
                    rank_map[uid] = parse_membership_rank(badges[0].lower() if badges else "")
                    known_rank.add(uid)
                last_at[uid] = max(last_at[uid], ts)
                continue
            if not text or not isinstance(text, str) or not text.strip():
                continue
            rank_map[uid] = parse_membership_rank(badges[0].lower() if badges else "")
            known_rank.add(uid)
            cat = categorize_message(text)
            if not cat:
                continue
            cat_by_user[uid][cat] += 1
            chat_counts[uid] += 1
            last_at[uid] = max(last_at[uid], ts)
            if has_humor(text):
                humor.append((ts, 1))
        humor.append((last_ts, 0))
        funniest = get_feature_timestamps({"humor": humor})["humor"]
        aggregate = stream_stats.finish()
        rows, user_rows = [], []
        with conn.cursor() as cur:
            table, month = _route(cur, video_id, last_ts)
            cur.execute("""SELECT EXISTS (
                           SELECT 1 FROM monthly_merge_state
                           WHERE observed_month=%s AND status='merged')""",
                        (month,))
            late_finalized = bool(cur.fetchone()[0])
            # Keep user_data_all (UNION ALL) duplicate-free: whichever table we
            # are about to write, this video must not have rows in the other.
            # Both deletes are index-backed on video_id.
            other = MAIN_TABLE if table == STAGING_TABLE else STAGING_TABLE
            cur.execute(f"DELETE FROM {other} WHERE video_id = %s", (video_id,))
            for uid in usernames:
                is_gift = uid in gift_only and uid not in known_rank
                c = cat_by_user[uid]
                rows.append((uid, channel_id,
                             datetime.fromtimestamp(last_at[uid] or last_ts, timezone.utc),
                             video_id, rank_map.get(uid, -2),
                             c["jp"], c["kr"], c["ru"], c["emoji"], c["es_en_id"],
                             chat_counts[uid], is_gift))
                user_rows.append((uid, usernames[uid]))
                if len(rows) >= FLUSH_EVERY:
                    _write(cur, table, month, rows, user_rows)
                    rows, user_rows = [], []
            if rows:
                _write(cur, table, month, rows, user_rows)
            cur.execute("""UPDATE videos SET has_chat_log = TRUE, funniest_timestamp = %s
                           WHERE video_id = %s""", (funniest, video_id))
            cur.execute("""UPDATE ingest_jobs
                           SET status='done', message_count=%s, completed_at=NOW(),
                               updated_at=NOW(), last_error=NULL
                           WHERE video_id=%s""", (total, video_id))
            _upsert_stream_stats(cur, video_id, aggregate)
            if late_finalized:
                # An operator-triggered re-publication consumes this durable
                # marker only after rebuilding derived data and invalidating
                # this month's permanent analytics caches.
                cur.execute("""INSERT INTO service_config (key, value, updated_at)
                               VALUES (%s, 'pending', NOW())
                               ON CONFLICT (key) DO UPDATE
                                 SET value='pending', updated_at=NOW()""",
                            (f"late_data_month:{month}",))
        conn.commit()
    except RawPartCorrupt as e:
        conn.rollback()
        with conn.cursor() as cur:
            cur.execute("""UPDATE ingest_jobs
                           SET status='failed',
                               continuation=NULL, part_count=0,
                               last_offset_s=0, messages_downloaded=0,
                               lease_id=NULL, last_error=%s,
                               completed_at=NOW(), updated_at=NOW()
                           WHERE video_id=%s""", (str(e)[:1000], video_id))
        conn.commit()
        emit({"RawPartsRejected": (1, COUNT)}, {"Stage": "ingest"},
             video_id=video_id)
        log.warning("corrupt raw part; awaiting operator retry",
                    extra={"video_id": video_id, "error": str(e)[:300]})
        return
    except Exception as e:
        conn.rollback()
        with conn.cursor() as cur:
            cur.execute("""UPDATE ingest_jobs SET status='failed', last_error=%s,
                           updated_at=NOW() WHERE video_id=%s""", (str(e)[:1000], video_id))
        conn.commit()
        emit({"IngestFailed": (1, COUNT)}, {"Stage": "ingest"}, video_id=video_id)
        raise                      # let SQS retry / eventually DLQ
    emit({"IngestCompleted": (1, COUNT), "MessagesIngested": (total, COUNT),
          "UniqueChatters": (len(usernames), COUNT),
          "StagedIngest": (1 if table == STAGING_TABLE else 0, COUNT)},
         {"Stage": "ingest"}, video_id=video_id)
    log.info("ingested", extra={"video_id": video_id, "messages": total,
                               "chatters": len(usernames),
                               "table": table, "month": str(month),
                               "late_finalized": late_finalized})


def _video_timing(conn, video_id):
    with conn.cursor() as cur:
        cur.execute("""SELECT COALESCE(EXTRACT(EPOCH FROM v.duration),
                                       j.video_duration_s, 0),
                              EXTRACT(EPOCH FROM v.end_time)
                                - COALESCE(EXTRACT(EPOCH FROM v.duration),
                                           j.video_duration_s, 0)
                       FROM videos v LEFT JOIN ingest_jobs j USING (video_id)
                       WHERE v.video_id=%s""", (video_id,))
        row = cur.fetchone()
    conn.rollback()
    if not row:
        return 0, None
    return int(row[0] or 0), float(row[1]) if row[1] is not None else None


def _upsert_stream_stats(cur, video_id, aggregate):
    """Persist only aggregate values; no message/user fields are accepted."""
    cur.execute("""INSERT INTO video_stream_stats (
                     video_id, schema_version, status, message_count,
                     unique_chatters, member_chatters, member_percentage,
                     category_counts,
                     membership_rank_counts, histogram_bin_seconds,
                     histogram_counts, funny_moments, word_counts,
                     first_message_at, last_error, computed_at, updated_at)
                   VALUES (%s,%s,'ready',%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s,
                           %s::jsonb,%s::jsonb,%s::jsonb,to_timestamp(%s),
                           NULL,NOW(),NOW())
                   ON CONFLICT (video_id) DO UPDATE SET
                     schema_version=EXCLUDED.schema_version,
                     status='ready', message_count=EXCLUDED.message_count,
                     unique_chatters=EXCLUDED.unique_chatters,
                     member_chatters=EXCLUDED.member_chatters,
                     member_percentage=EXCLUDED.member_percentage,
                     category_counts=EXCLUDED.category_counts,
                     membership_rank_counts=EXCLUDED.membership_rank_counts,
                     histogram_bin_seconds=EXCLUDED.histogram_bin_seconds,
                     histogram_counts=EXCLUDED.histogram_counts,
                     funny_moments=EXCLUDED.funny_moments,
                     word_counts=EXCLUDED.word_counts,
                     first_message_at=EXCLUDED.first_message_at,
                     last_error=NULL, computed_at=NOW(), updated_at=NOW()""",
                (video_id, aggregate["schema_version"],
                 aggregate["message_count"], aggregate["unique_chatters"],
                 aggregate["member_chatters"],
                 aggregate["member_percentage"],
                 json.dumps(aggregate["category_counts"]),
                 json.dumps(aggregate["membership_rank_counts"]),
                 aggregate["histogram_bin_seconds"],
                 json.dumps(aggregate["histogram_counts"]),
                 json.dumps(aggregate["funny_moments"]),
                 json.dumps(aggregate["word_counts"]),
                 aggregate["first_message_at"]))


def _missing_raw(exc):
    response = getattr(exc, "response", {}) or {}
    code = str((response.get("Error") or {}).get("Code", ""))
    status = (response.get("ResponseMetadata") or {}).get("HTTPStatusCode")
    return code in ("NoSuchKey", "NoSuchBucket", "404") or status == 404


def _backfill_stream_stats(msg):
    """Aggregate one retained historical raw log through ingest concurrency."""
    video_id, channel_id = msg["video_id"], msg["channel_id"]
    conn, s3 = get_conn(), client("s3")
    with conn.cursor() as cur:
        cur.execute("""SELECT value FROM service_config
                       WHERE key='stream_stats_backfill_enabled'""")
        enabled = cur.fetchone()
        if not enabled or str(enabled[0]).lower() != "true":
            # Pause also applies to work that was queued before the button was
            # pressed. Leave that stream resumable without consuming a retry.
            cur.execute("""UPDATE video_stream_stats
                           SET status='pending', updated_at=NOW()
                           WHERE video_id=%s AND status='queued'""", (video_id,))
            conn.commit()
            conn.close()
            return
        cur.execute("""SELECT s.status, j.part_count
                       FROM video_stream_stats s
                       JOIN ingest_jobs j USING (video_id)
                       WHERE s.video_id=%s AND j.status='done'
                       FOR UPDATE OF s""", (video_id,))
        row = cur.fetchone()
        if not row or row[0] not in ("queued", "pending", "failed"):
            conn.rollback()
            conn.close()
            return
        part_count = int(row[1] or 0)
        cur.execute("""UPDATE video_stream_stats
                       SET status='processing', attempts=attempts+1,
                           last_error=NULL, updated_at=NOW()
                       WHERE video_id=%s""", (video_id,))
    conn.commit()
    duration, stream_start_ts = _video_timing(conn, video_id)
    accumulator = StreamStatsAccumulator(duration, stream_start_ts)
    try:
        if part_count < 1:
            raise FileNotFoundError("no raw part checkpoints recorded")
        for message in _iter_messages(s3, channel_id, video_id, part_count):
            accumulator.add(message)
        with conn.cursor() as cur:
            _upsert_stream_stats(cur, video_id, accumulator.finish())
        conn.commit()
        emit({"StreamStatsBackfilled": (1, COUNT)}, {"Stage": "ingest"},
             video_id=video_id)
        log.info("stream statistics backfilled", extra={"video_id": video_id})
    except Exception as exc:
        conn.rollback()
        unavailable = isinstance(exc, FileNotFoundError) or _missing_raw(exc)
        with conn.cursor() as cur:
            cur.execute("""UPDATE video_stream_stats
                           SET status=%s, last_error=%s, updated_at=NOW()
                           WHERE video_id=%s""",
                        ("unavailable" if unavailable else "failed",
                         str(exc)[:500], video_id))
        conn.commit()
        log.warning("stream statistics backfill did not complete",
                    extra={"video_id": video_id,
                           "status": "unavailable" if unavailable else "failed",
                           "error": str(exc)[:200]})
    finally:
        conn.close()

def _write(cur, table, month, rows, user_rows):
    execute_values(cur, """
        INSERT INTO users (user_id, username) VALUES %s
        ON CONFLICT (user_id) DO UPDATE SET username = EXCLUDED.username""", user_rows)
    cols, tmpl, payload, extra = list(_COLS), _TMPL, rows, ""
    if table == STAGING_TABLE:
        cols.append("observed_month")
        tmpl += ",%s::date"
        payload = [tuple(r) + (month,) for r in rows]
        extra = ",\n        observed_month = EXCLUDED.observed_month"
    tmpl += ")"
    execute_values(cur, f"""
        INSERT INTO {table} ({", ".join(cols)})
        VALUES %s
        ON CONFLICT (user_id, channel_id, last_message_at, video_id) DO UPDATE
        SET {_UPDATE.format(t=table)}{extra}""",
        payload, template=tmpl)
