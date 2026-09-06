"""
Scheduled fan-out. Does no YouTube I/O at all.
Gates, in order:
  1. service_config.paused
  2. backpressure: skip entirely while the download/ingest pipeline still has
     meaningful work outstanding. This is what makes the cycle self-regulating:
     scan -> fill download queue -> drain -> scan again.
  3. per-channel cooldown (min_scan_interval_minutes), oldest-scanned first.
Invoke manually with {"force": true} to bypass gates 1 and 2, or
{"channels": ["UC..."]} to scan a specific subset.
"""
import json
import os
from common.aws import client
from common.config import settings
from common.db import get_conn
from common.logging_utils import get_logger
from common.metrics import emit, COUNT
from common import dispatch


log = get_logger("discover")
def handler(event, context):
    event = event or {}
    force = bool(event.get("force"))
    cfg = settings(force=True)
    if cfg.get("paused", "false").lower() == "true" and not force:
        log.warning("paused via service_config")
        emit({"DiscoverySkipped": (1, COUNT)}, {"Reason": "paused"})
        return {"scanned": 0, "reason": "paused"}
    _apply_concurrency(cfg)
    _sync_channels()
    outstanding = _outstanding_work()
    threshold = int(cfg["discovery_queue_threshold"])
    if not force and outstanding["total"] > threshold:
        log.info("backpressure: skipping scan", extra=outstanding)
        emit({"DiscoverySkipped": (1, COUNT),
              "OutstandingWork": (outstanding["total"], COUNT)},
             {"Reason": "backpressure"})
        return {"scanned": 0, "reason": "backpressure", **outstanding}
    channels = _eligible_channels(cfg, only=event.get("channels"), force=force)
    if not channels:
        log.info("no channels eligible (all within cooldown)")
        emit({"DiscoverySkipped": (1, COUNT)}, {"Reason": "cooldown"})
        return {"scanned": 0, "reason": "cooldown"}
    sqs = client("sqs")
    queue_url = os.environ["SCAN_QUEUE_URL"]
    sent, batch = 0, []
    for i, (channel_id, channel_name) in enumerate(channels):
        batch.append({
            "Id": str(i % 10),
            "MessageBody": json.dumps({"channel_id": channel_id,
                                       "channel_name": channel_name}),
        })
        if len(batch) == 10:
            sent += _send(sqs, queue_url, batch)
            batch = []
    if batch:
        sent += _send(sqs, queue_url, batch)
    emit({"ChannelsEnqueued": (sent, COUNT),
          "OutstandingWork": (outstanding["total"], COUNT)})
    dispatch.run(cfg)
    log.info("fan-out complete", extra={"channels_enqueued": sent,
                                        "outstanding": outstanding["total"]})
    return {"scanned": sent, **outstanding}
def _send(sqs, queue_url, batch):
    resp = sqs.send_message_batch(QueueUrl=queue_url, Entries=batch)
    for f in resp.get("Failed", []):
        log.error("send_message_batch failure", extra={"entry": f})
    return len(resp.get("Successful", []))
def _outstanding_work():
    """
    Combine SQS depth with DB state. SQS alone is not enough: a checkpointing
    download that is mid-flight shows as in_flight, but a job that failed and
    is waiting on a DelaySeconds retry shows in neither visible nor in_flight.
    """
    sqs = client("sqs")
    depths = {}
    lanes = [("download", "DOWNLOAD_QUEUE_URL"),
             ("ingest", "INGEST_QUEUE_URL"),
             ("scan", "SCAN_QUEUE_URL")]
    if os.environ.get("DOWNLOAD_RETRY_QUEUE_URL"):
        lanes.append(("download_retry", "DOWNLOAD_RETRY_QUEUE_URL"))
    for label, env in lanes:
        attrs = sqs.get_queue_attributes(
            QueueUrl=os.environ[env],
            AttributeNames=["ApproximateNumberOfMessages",
                            "ApproximateNumberOfMessagesNotVisible",
                            "ApproximateNumberOfMessagesDelayed"])["Attributes"]
        depths[label] = (int(attrs.get("ApproximateNumberOfMessages", 0))
                         + int(attrs.get("ApproximateNumberOfMessagesNotVisible", 0))
                         + int(attrs.get("ApproximateNumberOfMessagesDelayed", 0)))
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) FROM ingest_jobs
            WHERE status IN ('downloading', 'downloaded', 'ingesting')
              OR (status = 'pending' AND dispatched_at IS NOT NULL)
        """)
        active_jobs = cur.fetchone()[0]
    conn.rollback()
    return {
        "download_depth": depths["download"],
        "download_retry_depth": depths.get("download_retry", 0),
        "ingest_depth": depths["ingest"],
        "scan_depth": depths["scan"],
        "active_jobs": active_jobs,
        "total": max(sum(depths.values()), active_jobs),
    }

def _sync_channels():
    """
    Import channels.json on every run: names and groups stay in sync with the
    file, but `active` is owned by the admin UI and is never overwritten here.
    Entries whose group or name begins with '_' are placeholders and are skipped.
    A channel that exists only in the DB (added via the admin page) is untouched.
    """
    s3 = client("s3")
    try:
        data = json.loads(s3.get_object(Bucket=os.environ["CONFIG_BUCKET"],
                                        Key="channels.json")["Body"].read())
    except Exception as e:
        log.info("no channels.json to import", extra={"error": str(e)[:120]})
        return 0
    rows = [(cid, name, group)
            for group, members in data.items() if not group.startswith("_")
            if isinstance(members, dict)
            for name, cid in members.items() if not name.startswith("_")]
    if not rows:
        return 0
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
    return len(rows)

def _eligible_channels(cfg, only=None, force=False):
    """Active channels from the DB, filtered by cooldown, oldest-scanned first."""
    cooldown = 0 if force else int(cfg["min_scan_interval_minutes"])
    limit = int(cfg["max_channels_per_run"])
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT c.channel_id, c.channel_name, w.last_scanned_at
            FROM channels c
            LEFT JOIN channel_watermarks w ON w.channel_id = c.channel_id
            WHERE c.active
              AND (w.last_scanned_at IS NULL
                   OR w.last_scanned_at <= NOW() - (%s * INTERVAL '1 minute'))
              AND (%s::text[] IS NULL OR c.channel_id = ANY(%s::text[]))
            -- never-scanned first, then least-recently-scanned
            ORDER BY w.last_scanned_at ASC NULLS FIRST
        """, (cooldown, list(only) if only else None, list(only) if only else None))
        rows = cur.fetchall()
    conn.rollback()
    eligible = [(r[0], r[1]) for r in rows]
    return eligible[:limit] if limit > 0 else eligible

def _apply_concurrency(cfg):
    """
    Push the runtime concurrency settings onto the SQS event source mappings.
    Lets you retune the pipeline by editing one service_config row; takes
    effect on the next discovery tick.
    """
    ssm, lam = client("ssm"), client("lambda")
    app = os.environ.get("APP_NAME", "chat-ingest")
    wanted = {
        "download-q": max(2, int(cfg["max_concurrent_downloads"])),
        "scan-q": max(2, int(cfg["max_concurrent_scans"])),
    }
    for queue, n in wanted.items():
        try:
            uuid = ssm.get_parameter(Name=f"/{app}/esm/{queue}")["Parameter"]["Value"]
            lam.update_event_source_mapping(
                UUID=uuid, ScalingConfig={"MaximumConcurrency": n})
        except Exception as e:
            # Unsupported by the emulator, or parameter absent: reserved
            # concurrency on the function remains the effective cap.
            log.debug("could not apply ESM concurrency",
                      extra={"queue": queue, "error": str(e)})