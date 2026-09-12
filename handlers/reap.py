"""
Recovers jobs whose worker vanished: container restart, OOM kill, lost SQS
message, DLQ'd message. Detection is purely "the heartbeat stopped", which is
only meaningful because handlers/download.py writes progress every few seconds.
Recovered downloads rejoin the main download queue. The database advisory lock
serializes work without relying on two competing Floci event-source mappings.
Resume is cheap and exact: `continuation` is only ever advanced immediately
after a part lands in S3, so a reaped job replays from the last durable part.
Manual use:
    {"dry_run": true}      -- report what would be recovered
    {"limit": 5}           -- cap this pass
"""
import json
import os
from common.aws import client
from common.config import settings
from common.db import get_conn
from common.logging_utils import get_logger
from common.metrics import emit, COUNT
from common import dispatch
log = get_logger("reap")
RETRY_QUEUE = os.environ.get("DOWNLOAD_RETRY_QUEUE_URL")
def handler(event, context):
    event = event or {}
    cfg = settings(force=True)
    dry = bool(event.get("dry_run"))
    limit = int(event.get("limit", cfg.get("reap_batch_size", 25)))
    out = {
        "dry_run": dry,
        "cancelled_inactive": [] if dry else _purge_inactive(),
        "downloads": _reap_downloads(cfg, limit, dry),
        "ingests": _reap_ingests(cfg, limit, dry),
        "orphaned_downloaded": _reap_downloaded(cfg, limit, dry),
        "stream_stats_backfill": (_dispatch_stream_stats_backfill(cfg, dry)
                                  if not dry else {"dispatched": 0,
                                                   "reason": "dry run"}),
    }
    total = sum(len(v["recovered"]) for v in
                (out["downloads"], out["ingests"], out["orphaned_downloaded"]))
    emit({"JobsRecovered": (total, COUNT),
          "JobsAbandoned": (len(out["downloads"]["abandoned"]), COUNT)})
    if total or out["downloads"]["abandoned"]:
        log.info("reap pass complete", extra={"recovered": total,
                                              "abandoned": out["downloads"]["abandoned"]})
    out["dispatch"] = dispatch.run(cfg)
    return out


def _dispatch_stream_stats_backfill(cfg, dry=False):
    """Admit at most one low-priority aggregate job to the ingest queue."""
    del dry
    if str(cfg.get("stream_stats_backfill_enabled", "false")).lower() != "true":
        return {"dispatched": 0, "reason": "paused"}
    if str(cfg.get("paused", "false")).lower() == "true":
        return {"dispatched": 0, "reason": "ingestion paused"}
    conn = get_conn()
    with conn.cursor() as cur:
        # Recover a worker/container that vanished while aggregating.
        cur.execute("""UPDATE video_stream_stats
                       SET status='pending', last_error='backfill worker stalled',
                           updated_at=NOW()
                       WHERE status='processing'
                         AND updated_at < NOW() - INTERVAL '30 minutes'""")
        cur.execute("""SELECT EXISTS (
                                WITH active AS (
                                  SELECT MIN(date_trunc('month', v.end_time)) month
                                  FROM ingest_jobs j JOIN videos v USING (video_id)
                                  WHERE j.status IN ('pending','downloading',
                                                     'downloaded','ingesting'))
                                SELECT 1 FROM ingest_jobs j
                                LEFT JOIN videos v USING (video_id), active a
                                WHERE j.status='ingesting'
                                   OR (j.status='downloaded'
                                       AND date_trunc('month', v.end_time)=a.month)),
                              EXISTS (SELECT 1 FROM video_stream_stats
                                      WHERE status IN ('queued','processing'))""")
        live_ingest, already_active = cur.fetchone()
        if live_ingest or already_active:
            conn.commit()
            conn.close()
            return {"dispatched": 0,
                    "reason": "normal ingest waiting" if live_ingest
                              else "backfill already active"}
        cur.execute("""SELECT s.video_id, j.channel_id
                       FROM video_stream_stats s
                       JOIN ingest_jobs j USING (video_id)
                       WHERE s.status IN ('pending','failed')
                         AND s.attempts < 3 AND j.status='done'
                       ORDER BY j.completed_at DESC NULLS LAST
                       LIMIT 1 FOR UPDATE OF s SKIP LOCKED""")
        row = cur.fetchone()
        if not row:
            conn.commit()
            conn.close()
            return {"dispatched": 0, "reason": "no eligible work"}
        video_id, channel_id = row
        cur.execute("""UPDATE video_stream_stats SET status='queued',
                           updated_at=NOW() WHERE video_id=%s""", (video_id,))
    conn.commit()
    try:
        client("sqs").send_message(
            QueueUrl=os.environ["INGEST_QUEUE_URL"], DelaySeconds=30,
            MessageBody=json.dumps({"action": "backfill_stream_stats",
                                    "video_id": video_id,
                                    "channel_id": channel_id}))
    except Exception:
        with conn.cursor() as cur:
            cur.execute("""UPDATE video_stream_stats SET status='pending',
                               last_error='could not enqueue backfill',
                               updated_at=NOW() WHERE video_id=%s""", (video_id,))
        conn.commit()
        conn.close()
        raise
    conn.close()
    log.info("stream statistics backfill queued", extra={"video_id": video_id})
    return {"dispatched": 1, "video_id": video_id}
def _stale_preview(conn, status, minutes, limit):
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT video_id, channel_id, reaped_count,
                   EXTRACT(EPOCH FROM NOW() - updated_at)::int
            FROM ingest_jobs
            WHERE status = %s AND updated_at < NOW() - (%s * INTERVAL '1 minute')
            ORDER BY COALESCE(started_at, enqueued_at) ASC LIMIT %s""",
                    (status, minutes, limit))
        rows = cur.fetchall()
    conn.rollback()
    return [{"video_id": r[0], "channel_id": r[1], "reaped_count": r[2],
             "stale_s": r[3]} for r in rows]
def _reap_downloads(cfg, limit, dry):
    minutes = int(cfg.get("stale_download_minutes", 15))
    max_reaps = int(cfg.get("max_reaps_per_job", 5))
    conn = get_conn()
    if dry:
        return {"recovered": _stale_preview(conn, "downloading", minutes, limit),
                "abandoned": []}
    with conn.cursor() as cur:
        # A job that keeps dying is broken, not unlucky. Stop re-queuing it.
        cur.execute("""
            UPDATE ingest_jobs
            SET status='failed', lease_id=NULL, updated_at=NOW(), completed_at=NOW(),
                last_error = 'stalled ' || reaped_count || ' times; giving up'
            WHERE status='downloading'
              AND updated_at < NOW() - (%s * INTERVAL '1 minute')
              AND reaped_count >= %s
            RETURNING video_id""", (minutes, max_reaps))
        abandoned = [r[0] for r in cur.fetchall()]
        # SKIP LOCKED: two overlapping reaper ticks cannot claim the same row.
        # Oldest first -- the job that has been waiting longest goes first.
        cur.execute("""
            WITH stale AS (
                SELECT video_id FROM ingest_jobs
                WHERE status='downloading'
                  AND updated_at < NOW() - (%s * INTERVAL '1 minute')
                  AND reaped_count < %s
                ORDER BY COALESCE(started_at, enqueued_at) ASC
                LIMIT %s
                FOR UPDATE SKIP LOCKED
            )
            UPDATE ingest_jobs j
            SET status='pending', lease_id=NULL, reaped_count = j.reaped_count + 1,
                updated_at=NOW(),
                last_error='auto-recovered: heartbeat stopped'
            FROM stale s WHERE j.video_id = s.video_id
            RETURNING j.video_id, j.channel_id, j.part_count, j.reaped_count,
                      (j.continuation IS NOT NULL) AS resumable,
                      COALESCE(j.started_at, j.enqueued_at) AS age_key,
                      COALESCE(j.last_offset_s, 0)""",
                    (minutes, max_reaps, limit))
        rows = cur.fetchall()
    conn.commit()
    rows.sort(key=lambda r: r[5])          # oldest first onto the wire
    sqs = client("sqs")
    # One queue plus the database advisory lock guarantees one downloader.
    # A second event-source mapping can starve under reserved concurrency in
    # Floci, so recovered work rejoins the main FIFO-like lane.
    queue = os.environ["DOWNLOAD_QUEUE_URL"]
    recovered = []
    for video_id, channel_id, parts, reaps, resumable, _, offset in rows:
        sqs.send_message(QueueUrl=queue, MessageBody=json.dumps({
            "video_id": video_id, "channel_id": channel_id,
            "attempt": 0, "source": "retry", "resumed": bool(resumable)}))
        recovered.append({"video_id": video_id, "channel_id": channel_id,
                          "resumes_from_part": parts, "offset_s": offset,
                          "reaped_count": reaps})
        log.info("recovered stalled download",
                 extra={"video_id": video_id, "part": parts,
                        "offset_s": offset, "reaped_count": reaps})
    for v in abandoned:
        log.error("abandoned job after repeated stalls", extra={"video_id": v})
    return {"recovered": recovered, "abandoned": abandoned}
def _reap_ingests(cfg, limit, dry):
    """`ingesting` has no heartbeat -- only a start. A long ingest is normal;
    30 minutes of silence is not."""
    minutes = int(cfg.get("stale_ingest_minutes", 30))
    conn = get_conn()
    if dry:
        return {"recovered": _stale_preview(conn, "ingesting", minutes, limit)}
    with conn.cursor() as cur:
        cur.execute("""
            WITH stale AS (
                SELECT video_id FROM ingest_jobs
                WHERE status='ingesting'
                  AND updated_at < NOW() - (%s * INTERVAL '1 minute')
                ORDER BY COALESCE(started_at, enqueued_at) ASC
                LIMIT %s FOR UPDATE SKIP LOCKED
            )
            UPDATE ingest_jobs j
            SET status='downloaded', updated_at=NOW(),
                last_error='auto-recovered: ingest never finished'
            FROM stale s WHERE j.video_id = s.video_id
            RETURNING j.video_id, j.channel_id, j.part_count""", (minutes, limit))
        rows = cur.fetchall()
    conn.commit()
    return {"recovered": _requeue_ingest(rows)}
def _reap_downloaded(cfg, limit, dry):
    """Download finished, ingest message lost. The row is correct; nobody is
    coming to read it."""
    minutes = int(cfg.get("stale_downloaded_minutes", 30))
    conn = get_conn()
    if dry:
        return {"recovered": _stale_preview(conn, "downloaded", minutes, limit)}
    with conn.cursor() as cur:
        cur.execute("""
            SELECT video_id, channel_id, part_count FROM ingest_jobs
            WHERE status='downloaded'
              AND updated_at < NOW() - (%s * INTERVAL '1 minute')
            ORDER BY COALESCE(started_at, enqueued_at) ASC
            LIMIT %s FOR UPDATE SKIP LOCKED""", (minutes, limit))
        rows = cur.fetchall()
        if rows:
            cur.execute("""UPDATE ingest_jobs SET updated_at=NOW()
                           WHERE video_id = ANY(%s)""", ([r[0] for r in rows],))
    conn.commit()
    return {"recovered": _requeue_ingest(rows)}
def _requeue_ingest(rows):
    sqs = client("sqs")
    url = os.environ["INGEST_QUEUE_URL"]
    out = []
    for video_id, channel_id, part_count in rows:
        sqs.send_message(QueueUrl=url, MessageBody=json.dumps(
            {"video_id": video_id, "channel_id": channel_id,
             "part_count": part_count}))
        out.append({"video_id": video_id, "channel_id": channel_id,
                    "part_count": part_count})
        log.info("re-enqueued ingest", extra={"video_id": video_id})
    return out
def _purge_inactive():
    """Standing cleanup: non-terminal jobs for inactive/unknown channels are
    not work we want, no matter how they got enqueued (a pre-deactivation
    backfill, an in-flight scan, a manual enqueue_pending)."""
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("""
            DELETE FROM ingest_jobs j USING channels c
            WHERE c.channel_id = j.channel_id AND NOT c.active
              AND j.status IN ('pending', 'downloading', 'downloaded')
            RETURNING j.video_id, j.channel_id""")
        inactive = cur.fetchall()
        cur.execute("""
            DELETE FROM ingest_jobs j
            WHERE NOT EXISTS (SELECT 1 FROM channels c
                              WHERE c.channel_id = j.channel_id)
              AND j.status IN ('pending', 'downloading', 'downloaded')
            RETURNING j.video_id, j.channel_id""")
        orphans = cur.fetchall()
    conn.commit()
    out = [{"video_id": v, "channel_id": c} for v, c in inactive + orphans]
    if out:
        log.warning("cancelled jobs for inactive/unknown channels",
                    extra={"count": len(out)})
        emit({"JobsCancelled": (len(out), COUNT)})
    return out
