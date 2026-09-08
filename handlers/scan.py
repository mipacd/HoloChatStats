"""
Scans one channel's uploads playlist for concluded streams and enqueues
download jobs. One SQS message == one channel.
Resumable: if the 15-minute budget runs out mid-pagination, the message is
re-enqueued with the current page_token and the accumulated watermark
candidate. The watermark is only committed once a full pass completes, so an
interrupted scan can never skip videos.
"""
import json
import os
import time
from datetime import datetime, timezone, timedelta
import isodate
from chat_downloader import sites
from pyyoutube import Api
from common.aws import client
from common.config import secret, settings
from common.db import get_conn
from common.logging_utils import get_logger
from common.metrics import emit, COUNT, SECONDS
from common.control import paused, requeue_all
from common.channels import is_active, cancel_channel_jobs


log = get_logger("scan")
RESERVE_MS = 90_000   # headroom to re-enqueue + write state before timeout
QUOTA_MARKERS = ("quota", "quotaexceeded", "dailylimitexceeded")
# Some replay failures are only true at the time YouTube is queried.  They are
# terminal for month publication, but the first uploads-playlist page is
# reconsidered on later scans so a newly-published replay can be recovered.
SKIP_ON_LOOKUP_ERROR = ("members", "not available", "removed", "private", "deleted")
RECHECKABLE_SKIP = ("members", "not available", "private",
                    "concluded, no chat replay", "live event", "will begin")
def handler(event, context):
    if paused():
        requeue_all("SCAN_QUEUE_URL", event["Records"])
        return {"paused": True}
    for record in event["Records"]:
        _scan(json.loads(record["body"]), context)
    return {"ok": True}
def _scan(msg, context):
    channel_id = msg["channel_id"]
    channel_name = msg.get("channel_name", channel_id)
    t0 = time.time()
    cfg = settings()
    start_floor = datetime.fromisoformat(cfg["ingest_start_date"])
    if start_floor.tzinfo is None:
        start_floor = start_floor.replace(tzinfo=timezone.utc)
    lookback = timedelta(hours=int(cfg["discovery_lookback_hours"]))
    page_delay = int(cfg["scan_page_delay_ms"]) / 1000.0
    conn = get_conn()
    if not is_active(channel_id):
        cancelled = cancel_channel_jobs(channel_id)
        log.info("channel inactive; dropping scan and cancelling queued work",
                 extra={"channel_id": channel_id, "channel": channel_name,
                        "cancelled": len(cancelled)})
        emit({"ScansSkipped": (1, COUNT),
              "JobsCancelled": (len(cancelled), COUNT)}, {"Channel": channel_name})
        return
    sqs = client("sqs")
    api = Api(api_key=secret(os.environ["YT_SECRET_ID"])["api_key"])
    cd = sites.YouTubeChatDownloader()
    ignore = _ignore_list()
    # Resume state
    # ---- resume point -----------------------------------------------------
    # Resume state carried across re-enqueues
    page_token = msg.get("page_token")
    newest = _parse_dt(msg.get("newest"))
    baseline, cold_start = _channel_baseline(conn, channel_id, start_floor)
    if newest is None:
        newest = baseline
    if cold_start:
        # Nothing ingested for this channel ever: honour the global floor.
        floor = start_floor
    else:
        # Start *after* the newest video we already have, minus the lookback
        # window: a stream that was still live on the previous pass has no
        # end_time yet and must be revisited.
        floor = baseline - lookback
    log.info("scan window", extra={"channel_id": channel_id,
                                   "channel": channel_name,
                                   "baseline": baseline.isoformat(),
                                   "floor": floor.isoformat(),
                                   "cold_start": cold_start})
    playlist_id = "UU" + channel_id[2:]
    enqueued = pages = videos_seen = api_video_calls = 0
    lookup_errors = []
    reached_floor = False
    try:
        while True:
            if context.get_remaining_time_in_millis() < RESERVE_MS:
                sqs.send_message(
                    QueueUrl=os.environ["SCAN_QUEUE_URL"],
                    MessageBody=json.dumps({
                        "channel_id": channel_id,
                        "channel_name": channel_name,
                        "page_token": page_token,
                        "newest": newest.isoformat() if newest else None,
                    }))
                emit({"ScanCheckpoints": (1, COUNT)}, {"Channel": channel_name})
                log.info("scan checkpointed", extra={"channel_id": channel_id,
                                                     "pages": pages,
                                                     "enqueued": enqueued})
                return
            page = api.get_playlist_items(playlist_id=playlist_id,
                                          page_token=page_token, count=50)
            if not page or not page.items:
                break
            pages += 1
            items = [i for i in page.items
                     if i.contentDetails.videoId not in ignore]
            known = _known_videos(conn, [i.contentDetails.videoId for i in items])
            for item in items:
                video_id = item.contentDetails.videoId
                videos_seen += 1
                prior = known.get(video_id)
                # Already resolved: reuse the stored end_time for the
                # pagination decision and make zero network calls.
                recheck_skip = bool(
                    prior and prior["status"] == "skipped" and pages == 1
                    and _recheckable_skip(prior.get("skip_reason")))
                # Once page 1 crosses the ordinary discovery floor, inspect
                # the rest of that page only for eligible skipped replays.
                # Do not make metadata calls for other old entries and never
                # paginate to page 2 merely for late-arrival monitoring.
                if reached_floor and pages == 1 and not recheck_skip:
                    continue
                if (prior and prior["status"] in ("done", "skipped")
                        and prior["end_time"] and not recheck_skip):
                    if prior["end_time"] < floor:
                        reached_floor = True
                        if pages == 1:
                            continue
                        break
                    newest = max(newest, prior["end_time"])
                    continue
                try:
                    video_data = cd.get_video_data(video_id=video_id)
                except Exception as e:
                    if any(k in str(e).lower() for k in SKIP_ON_LOOKUP_ERROR):
                        _mark_skipped(conn, channel_id, video_id, str(e)[:200])
                        continue
                    log.warning("get_video_data failed",
                                extra={"video_id": video_id, "error": str(e)[:200]})
                    lookup_errors.append(f"{video_id}: {str(e)[:160]}")
                    continue
                end_date = _resolve_end_time(item, video_data)
                if end_date < floor and not recheck_skip:
                    reached_floor = True
                    if pages == 1:
                        continue
                    break
                newest = max(newest, end_date)
                # No continuation_info => plain upload, premiere without chat,
                # or a stream that has not concluded. Either way, nothing to
                # download right now; a live stream gets revisited via the
                # lookback window once it ends.
                if not video_data.get("continuation_info"):
                    if video_data.get("end_time"):
                        _mark_skipped(conn, channel_id, video_id,
                                      "concluded, no chat replay")
                    else:
                        log.info("not concluded yet, will revisit",
                                 extra={"video_id": video_id})
                    continue
                duration = video_data.get("duration")
                if not duration:
                    try:
                        d = api.get_video_by_id(video_id=video_id)
                        api_video_calls += 1
                        duration = isodate.parse_duration(
                            d.items[0].contentDetails.duration).total_seconds()
                    except Exception:
                        duration = 0
                if _upsert_and_claim(conn, channel_id, video_id,
                                     item.snippet.title, end_date, duration,
                                     reopen_skipped=recheck_skip):
                    # The job deliberately remains undispatched.  Only
                    # common.dispatch may release downloads, which guarantees
                    # strict oldest-month-first processing.
                    enqueued += 1
                    log.info("claimed video", extra={"video_id": video_id,
                             "end_time": end_date.isoformat()})
            page_token = page.nextPageToken
            if reached_floor or not page_token:
                break
            time.sleep(page_delay)
    except Exception as e:
        text = str(e)
        if any(k in text.lower() for k in QUOTA_MARKERS):
            # Don't retry into a wall: record it, let the next scheduled
            # cycle pick the channel back up after quota resets.
            _record_error(conn, channel_id, f"quota exceeded: {text[:300]}")
            emit({"QuotaExceeded": (1, COUNT)}, {"Channel": channel_name})
            log.error("youtube quota exceeded; abandoning scan",
                      extra={"channel_id": channel_id})
            return
        _record_error(conn, channel_id, text[:500])
        emit({"ScanFailed": (1, COUNT)}, {"Channel": channel_name})
        log.exception("scan failed", extra={"channel_id": channel_id})
        raise   # SQS retries, then DLQ
    # A partial lookup is not proof that this channel was checked through the
    # month boundary.  Keep its successful watermark unchanged so publication
    # remains blocked and the next discovery cycle retries it.
    if lookup_errors:
        _record_error(conn, channel_id, "; ".join(lookup_errors)[:500])
        emit({"ScanFailed": (1, COUNT)}, {"Channel": channel_name})
        log.warning("scan incomplete; successful watermark preserved",
                    extra={"channel_id": channel_id,
                           "lookup_errors": len(lookup_errors)})
        return
    # Full, error-free pass completed -> commit the watermark.
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO channel_watermarks
                (channel_id, last_scanned_at, last_seen_end_time, last_error)
            VALUES (%s, NOW(), %s, NULL)
            ON CONFLICT (channel_id) DO UPDATE
              SET last_scanned_at = NOW(),
                  last_seen_end_time = GREATEST(
                      COALESCE(channel_watermarks.last_seen_end_time,
                               EXCLUDED.last_seen_end_time),
                      EXCLUDED.last_seen_end_time),
                  last_error = NULL
        """, (channel_id, newest))
    conn.commit()
    emit({"ScansCompleted": (1, COUNT),
          "VideosEnqueued": (enqueued, COUNT),
          "PlaylistPagesFetched": (pages, COUNT),
          "VideoDetailApiCalls": (api_video_calls, COUNT),
          "ScanSeconds": (time.time() - t0, SECONDS)},
         {"Channel": channel_name})
    log.info("scan complete", extra={"channel_id": channel_id,
                                     "channel": channel_name,
                                     "pages": pages, "videos_seen": videos_seen,
                                     "enqueued": enqueued,
                                     "watermark": newest.isoformat()})
# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _parse_dt(value):
    if not value:
        return None
    dt = datetime.fromisoformat(value)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
def _resolve_end_time(item, video_data):
    end_raw = video_data.get("end_time")
    if end_raw:
        return datetime.fromtimestamp(end_raw / 1_000_000, timezone.utc)
    return datetime.fromisoformat(
        item.contentDetails.videoPublishedAt.replace("Z", "+00:00"))
def _ignore_list():
    s3 = client("s3")
    try:
        obj = s3.get_object(Bucket=os.environ["CONFIG_BUCKET"], Key="ignore.json")
        return set(json.loads(obj["Body"].read()))
    except Exception:
        return set()
def _known_videos(conn, video_ids):
    """Bulk lookup of job status + stored end_time for a playlist page."""
    if not video_ids:
        return {}
    with conn.cursor() as cur:
        cur.execute("""
            SELECT j.video_id, j.status, v.end_time, j.skip_reason
            FROM ingest_jobs j
            LEFT JOIN videos v USING (video_id)
            WHERE j.video_id = ANY(%s)
        """, (list(video_ids),))
        rows = cur.fetchall()
    conn.rollback()
    return {r[0]: {"status": r[1], "end_time": r[2], "skip_reason": r[3]}
            for r in rows}
def _recheckable_skip(reason):
    text = (reason or "").lower()
    return any(marker in text for marker in RECHECKABLE_SKIP)
def _upsert_and_claim(conn, channel_id, video_id, title, end_date, duration,
                      reopen_skipped=False):
    """
    Upsert video metadata and claim the download job. Returns True only when
    this call created a fresh pending job, so overlapping scans can't enqueue
    the same video twice.
    """
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO videos (video_id, channel_id, title, end_time, duration,
                                processed_at, has_chat_log)
            VALUES (%s, %s, %s, %s, make_interval(secs => %s), NOW(), FALSE)
            ON CONFLICT (video_id) DO UPDATE
              SET title = EXCLUDED.title,
                  end_time = EXCLUDED.end_time,
                  duration = EXCLUDED.duration,
                  processed_at = NOW()
        """, (video_id, channel_id, title, end_date, duration or 0))
        cur.execute("""
            INSERT INTO ingest_jobs (video_id, channel_id, status,
                                     video_duration_s, s3_prefix)
            VALUES (%s, %s, 'pending', %s, %s)
            ON CONFLICT (video_id) DO NOTHING
            RETURNING video_id
        """, (video_id, channel_id, duration or 0, f"{channel_id}/{video_id}"))
        claimed = cur.fetchone() is not None
        if not claimed and reopen_skipped:
            cur.execute("""
                UPDATE ingest_jobs
                   SET status='pending', attempts=0, continuation=NULL,
                       part_count=0, last_offset_s=0, messages_downloaded=0,
                       lease_id=NULL, reaped_count=0, message_count=NULL,
                       skip_reason=NULL, last_error=NULL, dispatched_at=NULL,
                       enqueued_at=NOW(), started_at=NULL, completed_at=NULL,
                       updated_at=NOW(), video_duration_s=%s, s3_prefix=%s
                 WHERE video_id=%s AND status='skipped'
                 RETURNING video_id
            """, (duration or 0, f"{channel_id}/{video_id}", video_id))
            claimed = cur.fetchone() is not None
    conn.commit()
    return claimed
def _mark_skipped(conn, channel_id, video_id, reason):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO ingest_jobs (video_id, channel_id, status, skip_reason,
                                     completed_at)
            VALUES (%s, %s, 'skipped', %s, NOW())
            ON CONFLICT (video_id) DO UPDATE
              SET status = CASE WHEN ingest_jobs.status = 'pending'
                                THEN 'skipped' ELSE ingest_jobs.status END,
                  skip_reason = COALESCE(ingest_jobs.skip_reason, EXCLUDED.skip_reason),
                  updated_at = NOW()
        """, (video_id, channel_id, reason[:200]))
    conn.commit()
def _record_error(conn, channel_id, message):
    conn.rollback()
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO channel_watermarks (channel_id, last_scanned_at, last_error)
            VALUES (%s, NULL, %s)
            ON CONFLICT (channel_id) DO UPDATE
              SET last_error = EXCLUDED.last_error
        """, (channel_id, message))
    conn.commit()
def _channel_baseline(conn, channel_id, start_floor):
    """
    Where discovery for this channel resumes from, newest-first:
      1. channel_watermarks.last_seen_end_time  (previous scans)
      2. MAX(videos.end_time) among videos we already ingested or deliberately
         skipped                                 (pre-existing / restored data)
      3. service_config.ingest_start_date        (channel with no history)
    (2) is what stops a populated database from re-walking two years of
    playlist: GREATEST() ignores NULLs, so a missing watermark row falls
    through to the data itself instead of to the global floor.
    Returns (baseline, cold_start).
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT GREATEST(
                     (SELECT w.last_seen_end_time
                        FROM channel_watermarks w
                       WHERE w.channel_id = %s),
                     (SELECT MAX(v.end_time)
                        FROM videos v
                        LEFT JOIN ingest_jobs j USING (video_id)
                       WHERE v.channel_id = %s
                         AND (v.has_chat_log OR j.status IN ('done', 'skipped')))
                   )
        """, (channel_id, channel_id))
        baseline = cur.fetchone()[0]
    conn.rollback()
    if baseline is None:
        return start_floor, True
    return baseline, False
