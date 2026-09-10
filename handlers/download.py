import gzip, io, json, os, time, uuid
from common.aws import client
from common.config import setting
from common.control import paused, requeue_all
from common.db import get_conn
from common.logging_utils import get_logger
from common.metrics import emit, COUNT, SECONDS
from common.youtube import ChatReplay
from common.channels import is_active, cancel_job
from common.month_order import work_months


log = get_logger("download")
RESERVE_MS = 60_000          # leave headroom to flush + checkpoint
PAGES_PER_PART = 400         # ~one S3 object per 400 pages
HEARTBEAT_SECONDS = 5        # progress write cadence (drives stall detection)
DOWNLOAD_LOCK_KEY = 744_211_988
BUCKET = os.environ["RAW_BUCKET"]
RETRY_QUEUE = os.environ.get("DOWNLOAD_RETRY_QUEUE_URL")
PERMANENT = ("members", "not available", "removed", "private", "no chat replay",
             "no continuation", "live event", "will begin", "age-restricted",
             "age restricted", "confirm your age")
# Chat replay pagination ending is the authoritative completion signal. This
# threshold only controls a diagnostic warning; quiet tails never fail a job.
REPLAY_END_SILENCE_SECONDS = 15 * 60
class LeaseLost(Exception):
    """Our row was reassigned (reaper decided we were dead). Stop immediately:
    anything we write from here on would corrupt the new owner's checkpoint."""
def handler(event, context):
    if paused():
        requeue_all("DOWNLOAD_QUEUE_URL", event["Records"])
        log.info("paused: deferred batch", extra={"n": len(event["Records"])})
        return {"paused": True}
    sqs = client("sqs")
    for record in event["Records"]:
        msg = json.loads(record["body"])
        source = msg.get("source") or msg.get("src")
        if source not in (None, "dispatch", "retry", "resume", "manual"):
            log.warning("download message from an unknown producer -- month ordering "
                "is not being honoured", extra={"video_id": msg.get("video_id"),
                                                "msg": msg})
        # The channel may have been deactivated after this message was sent.
        # Drop the job outright: greyed out in the UI must mean not running.
        if not is_active(msg.get("channel_id")):
            cancel_job(msg.get("video_id"))
            log.warning("channel inactive; cancelling queued download",
                        extra={"video_id": msg.get("video_id"),
                               "channel_id": msg.get("channel_id")})
            emit({"JobsCancelled": (1, COUNT)}, {"Stage": "download"},
                 video_id=msg.get("video_id"))
            continue
        lock_conn = get_conn()
        with lock_conn.cursor() as cur:
            cur.execute("SELECT pg_try_advisory_lock(%s)",
                        (DOWNLOAD_LOCK_KEY,))
            have_lock = cur.fetchone()[0]
        if not have_lock:
            lock_conn.close()
            sqs.send_message(QueueUrl=os.environ["DOWNLOAD_QUEUE_URL"],
                             MessageBody=json.dumps(msg),
                             DelaySeconds=15)
            log.info("another chat download is active; deferred",
                     extra={"video_id": msg.get("video_id")})
            continue
        try:
            if _defer_future_month(msg):
                continue
            try:
                _process(msg, context)
            except LeaseLost:
                # The reaper already re-enqueued this job. Returning normally
                # deletes our (now duplicate) message.
                log.warning("lease revoked; abandoning",
                            extra={"video_id": msg.get("video_id")})
                emit({"DownloadLeaseLost": (1, COUNT)}, {"Stage": "download"},
                     video_id=msg.get("video_id"))
        finally:
            with lock_conn.cursor() as cur:
                cur.execute("SELECT pg_advisory_unlock(%s)",
                            (DOWNLOAD_LOCK_KEY,))
            lock_conn.close()
    return {"ok": True}

def _defer_future_month(msg):
    """Undo stale queue dispatches that are newer than the active month."""
    conn = get_conn()
    video_month, active_month = work_months(conn, msg.get("video_id"))
    if not video_month or not active_month or video_month <= active_month:
        conn.close()
        return False
    with conn.cursor() as cur:
        # Preserve durable S3 continuation/part checkpoints. When this month
        # becomes active the dispatcher can safely resume rather than restart.
        cur.execute("""
            UPDATE ingest_jobs
            SET status='pending', dispatched_at=NULL, lease_id=NULL,
                updated_at=NOW(),
                last_error='deferred: waiting for month ' || %s::text
            WHERE video_id=%s
              AND status IN ('pending', 'downloading')
        """, (active_month, msg.get("video_id")))
        deferred = cur.rowcount > 0
    conn.commit()
    conn.close()
    if deferred:
        log.warning("future-month download returned to dispatcher",
                    extra={"video_id": msg.get("video_id"),
                           "video_month": str(video_month),
                           "active_month": str(active_month)})
        emit({"DownloadsDeferredByMonth": (1, COUNT)}, {"Stage": "download"},
             video_id=msg.get("video_id"), active_month=str(active_month))
    return deferred

def _process(msg, context):
    video_id, channel_id = msg["video_id"], msg["channel_id"]
    conn, sqs, s3 = get_conn(), client("sqs"), client("s3")
    t0 = time.time()
    lease = str(uuid.uuid4())
    with conn.cursor() as cur:
        cur.execute("""SELECT j.status, j.continuation, j.part_count, j.attempts,
                              j.last_offset_s, j.video_duration_s,
                              COALESCE(j.messages_downloaded, 0),
                              EXTRACT(EPOCH FROM (
                                  v.end_time - COALESCE(
                                      v.duration,
                                      make_interval(secs => j.video_duration_s)
                                  )
                              ))::double precision AS derived_start_ts
                       FROM ingest_jobs j
                       LEFT JOIN videos v ON v.video_id = j.video_id
                       WHERE j.video_id = %s FOR UPDATE OF j""", (video_id,))
        row = cur.fetchone()
        if row is None:
            log.warning("no job row; dropping", extra={"video_id": video_id})
            conn.rollback(); return
        (status, continuation, part_count, attempts, last_offset, duration,
         msgs_before, stored_start_ts) = row
        if status in ("done", "skipped", "ingesting", "downloaded"):
            log.info("already past download stage", extra={"video_id": video_id,
                                                           "status": status})
            conn.rollback(); return
        # Taking the lease is what makes any previous owner a zombie.
        cur.execute("""UPDATE ingest_jobs
                       SET status='downloading', attempts = attempts + 1,
                           lease_id = %s,
                           started_at = COALESCE(started_at, NOW()), updated_at = NOW()
                       WHERE video_id = %s""", (lease, video_id))
    conn.commit()
    try:
        replay = ChatReplay(video_id,
                            continuation=continuation,
                            video_start_ts=(msg.get("video_start_ts")
                                            or stored_start_ts),
                            duration=duration)
    except Exception as e:
        return _handle_error(conn, sqs, video_id, channel_id, msg, e)
    if not duration and getattr(replay, "duration", None):
        duration = duration or replay.duration
        with conn.cursor() as cur:
            cur.execute("""UPDATE ingest_jobs SET video_duration_s = %s,
                           updated_at = NOW() WHERE video_id = %s""",
                        (duration, video_id))
        conn.commit()
    buf, pages, written, next_cont = [], 0, 0, continuation
    last_beat = time.time()
    try:
        for messages, nxt in replay.pages():
            buf.extend(messages)
            pages += 1
            next_cont = nxt
            if messages:
                last_offset = max(last_offset or 0,
                                  messages[-1]["timestamp"] - replay.video_start_ts)
            budget_gone = context.get_remaining_time_in_millis() < RESERVE_MS
            if pages % PAGES_PER_PART == 0 or budget_gone:
                written += _flush(s3, channel_id, video_id, part_count, buf)
                part_count += 1
                buf = []
                # Durable checkpoint the instant the part is in S3: `next_cont`
                # is the token for the page AFTER everything just flushed, so a
                # reaper-driven resume replays nothing and skips nothing.
                _checkpoint(conn, video_id, lease, next_cont, part_count,
                            last_offset, msgs_before + written)
                last_beat = time.time()
            # Heartbeat: offset/messages only. Deliberately NOT `continuation`
            # -- pages still sitting in `buf` are not durable, so advancing the
            # resume token here would silently lose them.
            now = time.time()
            if now - last_beat >= HEARTBEAT_SECONDS:
                _progress(conn, video_id, lease, last_offset, part_count,
                          msgs_before + written + len(buf))
                last_beat = now
            if budget_gone:
                sqs.send_message(
                    QueueUrl=os.environ["DOWNLOAD_QUEUE_URL"],
                    MessageBody=json.dumps({**msg,
                                            "video_start_ts": replay.video_start_ts,
                                            "resumed": True}))
                emit({"DownloadCheckpoints": (1, COUNT),
                      "MessagesDownloaded": (written, COUNT)},
                     {"Stage": "download"}, video_id=video_id)
                log.info("checkpointed", extra={"video_id": video_id,
                                                "part": part_count,
                                                "offset_s": last_offset})
                return
    except LeaseLost:
        raise
    except Exception as e:
        if buf:
            written += _flush(s3, channel_id, video_id, part_count, buf)
            part_count += 1
            _checkpoint(conn, video_id, lease, next_cont, part_count,
                        last_offset, msgs_before + written)
        return _handle_error(
            conn, sqs, video_id, channel_id,
            {**msg, "video_start_ts": replay.video_start_ts, "resumed": True}, e)
    if buf:
        written += _flush(s3, channel_id, video_id, part_count, buf)
        part_count += 1
    if (duration and last_offset
            and last_offset < (duration - REPLAY_END_SILENCE_SECONDS)):
        log.warning("chat replay ended with a quiet tail; accepting completion",
                    extra={"video_id": video_id,
                           "last_message_s": int(last_offset),
                           "video_duration_s": int(duration),
                           "quiet_tail_s": int(duration - last_offset)})
    with conn.cursor() as cur:
        cur.execute("""UPDATE ingest_jobs
                       SET status='downloaded', continuation=NULL, part_count=%s,
                           last_offset_s=%s, messages_downloaded=%s,
                           lease_id=NULL, updated_at=NOW(), last_error=NULL
                       WHERE video_id=%s AND lease_id=%s""",
                    (part_count, last_offset, msgs_before + written, video_id, lease))
        if cur.rowcount == 0:
            conn.rollback()
            raise LeaseLost(video_id)      # do NOT enqueue a duplicate ingest
    conn.commit()
    sqs.send_message(
        QueueUrl=os.environ["INGEST_QUEUE_URL"],
        MessageBody=json.dumps({"video_id": video_id, "channel_id": channel_id,
                                "part_count": part_count}))
    emit({"DownloadsCompleted": (1, COUNT),
          "DownloadSeconds": (time.time() - t0, SECONDS)},
         {"Stage": "download"}, video_id=video_id, parts=part_count)
def _flush(s3, channel_id, video_id, part, messages):
    key = f"{channel_id}/{video_id}/part-{part:05d}.jsonl.gz"
    raw = io.BytesIO()
    with gzip.GzipFile(fileobj=raw, mode="wb") as gz:
        for m in messages:
            gz.write((json.dumps(m, ensure_ascii=False) + "\n").encode())
    s3.put_object(Bucket=BUCKET, Key=key, Body=raw.getvalue(),
                  ContentType="application/jsonl", ContentEncoding="gzip")
    return len(messages)
def _progress(conn, video_id, lease, last_offset, part_count, messages):
    with conn.cursor() as cur:
        cur.execute("""UPDATE ingest_jobs
                       SET last_offset_s = GREATEST(COALESCE(last_offset_s, 0), %s),
                           part_count = GREATEST(part_count, %s),
                           messages_downloaded = GREATEST(
                               COALESCE(messages_downloaded, 0), %s),
                           updated_at = NOW()
                       WHERE video_id = %s AND lease_id = %s""",
                    (last_offset or 0, part_count, messages, video_id, lease))
        lost = cur.rowcount == 0
    conn.commit()
    if lost:
        raise LeaseLost(video_id)
def _checkpoint(conn, video_id, lease, continuation, part_count, last_offset, messages):
    with conn.cursor() as cur:
        cur.execute("""UPDATE ingest_jobs
                       SET continuation=%s, part_count=%s, last_offset_s=%s,
                           messages_downloaded=%s, updated_at=NOW()
                       WHERE video_id=%s AND lease_id=%s""",
                    (continuation, part_count, last_offset, messages, video_id, lease))
        lost = cur.rowcount == 0
    conn.commit()
    if lost:
        raise LeaseLost(video_id)
def _handle_error(conn, sqs, video_id, channel_id, msg, exc):
    text = str(exc)
    permanent = any(k in text.lower() for k in PERMANENT)
    max_retries = setting("max_retries", 5, int)
    attempt = msg.get("attempt", 0) + 1
    if permanent or attempt >= max_retries:
        with conn.cursor() as cur:
            cur.execute("""UPDATE ingest_jobs
                           SET status=%s, last_error=%s, lease_id=NULL,
                               updated_at=NOW(), completed_at=NOW(), skip_reason=%s
                           WHERE video_id=%s""",
                        ("skipped" if permanent else "failed", text[:1000],
                         text[:200] if permanent else None, video_id))
        conn.commit()
        emit({"DownloadsSkipped" if permanent else "DownloadsFailed": (1, COUNT)},
             {"Stage": "download"}, video_id=video_id, error=text[:200])
        log.warning("terminal", extra={"video_id": video_id, "permanent": permanent,
                                       "error": text[:200]})
        return
    delay = min(900, 5 * (2 ** attempt))
    with conn.cursor() as cur:
        cur.execute("""UPDATE ingest_jobs SET status='pending', last_error=%s,
                       lease_id=NULL, updated_at=NOW() WHERE video_id=%s""",
                    (text[:1000], video_id))
    conn.commit()
    # Keep the message on whichever lane it arrived on.
    queue = os.environ["DOWNLOAD_QUEUE_URL"]
    sqs.send_message(QueueUrl=queue,
                     MessageBody=json.dumps({**msg, "attempt": attempt}),
                     DelaySeconds=delay)
    emit({"DownloadRetries": (1, COUNT)}, {"Stage": "download"},
         video_id=video_id, delay=delay, error=text[:200])
