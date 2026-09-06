"""Month-ordered dispatcher: the single place that decides processing ORDER.
Discovery (scan) finds videos newest-first because that is how the uploads
playlist paginates; this module releases them to the download queue
oldest-month-first, one month at a time, and refuses to start month N+1 while
a closed month N is still unmerged."""
import json, os
from datetime import datetime, timezone
from common.aws import client
from common.db import get_conn
from common.logging_utils import get_logger
from common.floor import backlog_floor
log = get_logger("dispatch")

def _scan_work_outstanding():
    """Do not release a month until the current discovery sweep is complete."""
    queue_url = os.environ.get("SCAN_QUEUE_URL")
    if not queue_url:
        return 0
    attrs = client("sqs").get_queue_attributes(
        QueueUrl=queue_url,
        AttributeNames=["ApproximateNumberOfMessages",
                        "ApproximateNumberOfMessagesNotVisible",
                        "ApproximateNumberOfMessagesDelayed"],
    )["Attributes"]
    return sum(int(attrs.get(k, 0)) for k in (
        "ApproximateNumberOfMessages",
        "ApproximateNumberOfMessagesNotVisible",
        "ApproximateNumberOfMessagesDelayed",
    ))
def _floor(cur, cfg):
    v = (cfg.get("backlog_floor") or "").strip()
    if v:
        return datetime.fromisoformat(v).replace(tzinfo=timezone.utc)
    # Fallback: the month after the newest merged month; else newest done data.
    cur.execute("""SELECT COALESCE(
        (SELECT (MAX(observed_month) + INTERVAL '1 month') FROM monthly_merge_state
          WHERE status='merged'),
        (SELECT date_trunc('month', MAX(v.end_time)) FROM videos v
          JOIN ingest_jobs j USING (video_id) WHERE j.status='done'))""")
    row = cur.fetchone()[0]
    return row or datetime(1970, 1, 1, tzinfo=timezone.utc)
def run(cfg):
    scans = _scan_work_outstanding()
    if scans:
        return {"dispatched": 0, "reason": "discovery sweep in progress",
                "scan_work": scans}
    conn = get_conn()
    batch = int(cfg.get("dispatch_batch_size", 50))
    this_month = datetime.now(timezone.utc).date().replace(day=1)
    floor = backlog_floor(conn, cfg)
    if floor is None:
        return {"dispatched": 0, "reason": "no backlog_floor and no processed data"}
    with conn.cursor() as cur:
        cur.execute("""
            SELECT date_trunc('month', v.end_time)::date AS month,
                   COUNT(*) FILTER (WHERE j.status='pending'
                                      AND j.dispatched_at IS NULL) AS waiting,
                   COUNT(*) FILTER (WHERE j.status='pending'
                                      AND j.dispatched_at IS NOT NULL) AS in_queue,
                   COUNT(*) FILTER (WHERE j.status IN ('downloading','downloaded',
                                                       'ingesting')) AS in_flight
            FROM ingest_jobs j JOIN videos v USING (video_id)
            WHERE v.end_time >= %s
            GROUP BY 1 HAVING COUNT(*) FILTER (
                WHERE j.status IN ('pending','downloading','downloaded','ingesting')
            ) > 0
            ORDER BY 1 LIMIT 1""", (floor,))
        row = cur.fetchone()
        if row is None:
            conn.rollback()
            return {"dispatched": 0, "floor": str(floor.date()),
                    "reason": "no unfinished work at or after floor"}
        month, waiting, in_queue, in_flight = row
        cur.execute("SELECT MIN(observed_month) FROM user_data_current "
                    "WHERE observed_month < %s", (month,))
        stuck = cur.fetchone()[0]
        if stuck and stuck < this_month:
            conn.rollback()
            client("lambda").invoke(
                FunctionName=f"{os.environ.get('APP_NAME','chat-ingest')}-merge",
                InvocationType="Event",
                Payload=json.dumps({"months": [str(stuck)]}).encode())
            return {"dispatched": 0, "floor": str(floor.date()),
                    "active_month": str(month),
                    "reason": f"blocked: {stuck} drained but not merged"}
        cur.execute("""
            SELECT j.video_id, j.channel_id
            FROM ingest_jobs j JOIN videos v USING (video_id)
            WHERE j.status='pending' AND j.dispatched_at IS NULL
              AND v.end_time >= %s::date
              AND v.end_time <  (%s::date + INTERVAL '1 month')
            ORDER BY v.end_time ASC
            LIMIT %s FOR UPDATE OF j SKIP LOCKED""", (month, month, batch))
        jobs = cur.fetchall()
        if jobs:
            cur.execute("UPDATE ingest_jobs SET dispatched_at=NOW(), updated_at=NOW() "
                        "WHERE video_id = ANY(%s)", ([v for v, _ in jobs],))
    conn.commit()
    sqs, q = client("sqs"), os.environ["DOWNLOAD_QUEUE_URL"]
    failed = []
    for video_id, channel_id in jobs:
        try:
            sqs.send_message(QueueUrl=q, MessageBody=json.dumps(
                {"video_id": video_id, "channel_id": channel_id,
                 "attempt": 0, "src": "dispatch"}))
        except Exception:
            failed.append(video_id)
            log.exception("dispatch send failed", extra={"video_id": video_id})
    if failed:
        conn = get_conn()
        with conn.cursor() as cur:
            cur.execute("UPDATE ingest_jobs SET dispatched_at=NULL, updated_at=NOW() "
                        "WHERE video_id = ANY(%s) AND status='pending'", (failed,))
        conn.commit()
    return {"dispatched": len(jobs) - len(failed), "floor": str(floor.date()),
            "active_month": str(month), "waiting": waiting,
            "in_queue": in_queue, "in_flight": in_flight,
            "send_failures": len(failed)}
