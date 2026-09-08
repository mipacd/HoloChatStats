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
def handler(event, context):
    if paused():
        requeue_all("INGEST_QUEUE_URL", event["Records"])
        return {"paused": True}
    for record in event["Records"]:
        _ingest(json.loads(record["body"]))
    return {"ok": True}
def _iter_messages(s3, channel_id, video_id, part_count):
    for part in range(part_count):
        key = f"{channel_id}/{video_id}/part-{part:05d}.jsonl.gz"
        body = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
        for line in gzip.decompress(body).decode().splitlines():
            if line:
                yield json.loads(line)
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
    cur.execute("""SELECT 1 FROM monthly_merge_state
                   WHERE observed_month = %s AND status = 'merged'""", (month,))
    merged = cur.fetchone() is not None
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
                       WHERE video_id=%s AND status IN ('downloaded','failed')
                       RETURNING part_count""", (video_id,))
        row = cur.fetchone()
    conn.commit()
    if row is None:
        log.info("not ready / already ingested", extra={"video_id": video_id})
        return
    part_count = row[0]
    cat_by_user = defaultdict(lambda: defaultdict(int))
    rank_map, chat_counts, last_at, usernames = {}, defaultdict(int), defaultdict(float), {}
    gift_only, known_rank = set(), set()
    humor, last_ts, total = [], 0.0, 0
    try:
        for m in _iter_messages(s3, channel_id, video_id, part_count):
            total += 1
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
            if late_finalized:
                # The scheduled refresh consumes this durable marker only
                # after rebuilding derived data and invalidating this month's
                # permanent analytics caches.
                cur.execute("""INSERT INTO service_config (key, value, updated_at)
                               VALUES (%s, 'pending', NOW())
                               ON CONFLICT (key) DO UPDATE
                                 SET value='pending', updated_at=NOW()""",
                            (f"late_data_month:{month}",))
        conn.commit()
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
