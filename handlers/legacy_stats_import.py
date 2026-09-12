"""One-time, aggregate-only importer for flattened legacy chat archives."""
import gzip
import io
import json
import os
import re
from datetime import timedelta

from common.aws import client
from common.db import get_conn
from common.logging_utils import get_logger
from common.stream_stats import StreamStatsAccumulator, upsert_stream_stats


log = get_logger("legacy_stats_import")
BUCKET = os.environ["RAW_BUCKET"]
PREFIX = "legacy-stream-stats-import/"
VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
MAX_STATUS_IDS = 500
MAX_IMPORT_ITEMS = 32
MIN_REMAINING_MS = 60_000


def handler(event, context):
    event = event or {}
    action = event.get("action")
    if action == "status":
        return status(event.get("video_ids") or [])
    if action == "import":
        return import_batch(event.get("items") or [], context)
    raise ValueError("action must be 'status' or 'import'")


def _video_ids(values, limit):
    if not isinstance(values, list) or len(values) > limit:
        raise ValueError(f"expected a list of at most {limit} video IDs")
    if any(not isinstance(value, str) or not VIDEO_ID_RE.fullmatch(value)
           for value in values):
        raise ValueError("invalid video ID")
    return list(dict.fromkeys(values))


def status(values):
    """Classify IDs before upload so known work is the only data transferred."""
    video_ids = _video_ids(values, MAX_STATUS_IDS)
    if not video_ids:
        return {"ready": [], "eligible": [], "missing": []}
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("""SELECT v.video_id, COALESCE(s.status = 'ready', FALSE)
                       FROM videos v LEFT JOIN video_stream_stats s USING (video_id)
                       WHERE v.video_id = ANY(%s)""", (video_ids,))
        rows = dict(cur.fetchall())
    conn.rollback()
    known = set(rows)
    return {
        "ready": [video_id for video_id in video_ids if rows.get(video_id)],
        "eligible": [video_id for video_id in video_ids
                     if video_id in known and not rows[video_id]],
        "missing": [video_id for video_id in video_ids if video_id not in known],
    }


def import_batch(values, context):
    if not isinstance(values, list) or len(values) > MAX_IMPORT_ITEMS:
        raise ValueError(f"expected at most {MAX_IMPORT_ITEMS} import items")
    items = []
    for value in values:
        if not isinstance(value, dict):
            raise ValueError("each import item must be an object")
        video_id = value.get("video_id")
        key = value.get("key")
        if (not isinstance(video_id, str) or not VIDEO_ID_RE.fullmatch(video_id)
                or not isinstance(key, str)
                or key != f"{PREFIX}{video_id}.jsonl.gz"):
            raise ValueError("invalid import item")
        items.append((video_id, key))
    results, deferred = [], []
    for index, (video_id, key) in enumerate(items):
        remaining = (context.get_remaining_time_in_millis()
                     if context and hasattr(context, "get_remaining_time_in_millis")
                     else 900_000)
        if remaining < MIN_REMAINING_MS:
            deferred.extend({"video_id": item[0], "key": item[1]}
                            for item in items[index:])
            break
        results.append(_process(video_id, key))
    return {"results": results, "deferred": deferred}


def _timing_and_state(video_id):
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("""SELECT v.end_time, v.duration,
                              COALESCE(s.status = 'ready', FALSE)
                       FROM videos v LEFT JOIN video_stream_stats s USING (video_id)
                       WHERE v.video_id=%s""", (video_id,))
        row = cur.fetchone()
    conn.rollback()
    if not row:
        return None
    end_time, duration, ready = row
    seconds = max(0, int(duration.total_seconds())) if duration else 0
    start = ((end_time - timedelta(seconds=seconds)).timestamp()
             if end_time and seconds else None)
    return seconds, start, bool(ready)


def _aggregate(key, duration, start):
    s3 = client("s3")
    response = s3.get_object(Bucket=BUCKET, Key=key)
    accumulator = StreamStatsAccumulator(duration, start)
    nonempty = valid = malformed = 0
    try:
        with gzip.GzipFile(fileobj=response["Body"], mode="rb") as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as lines:
                for line in lines:
                    if not line.strip():
                        continue
                    nonempty += 1
                    try:
                        message = json.loads(line)
                    except json.JSONDecodeError:
                        malformed += 1
                        continue
                    if isinstance(message, dict) and accumulator.add_legacy(message):
                        valid += 1
                    else:
                        malformed += 1
    finally:
        body = response.get("Body")
        if body and hasattr(body, "close"):
            body.close()
    if nonempty and not valid:
        raise ValueError("non-empty archive contained no valid legacy records")
    return accumulator.finish(), {"records": valid, "malformed": malformed,
                                  "empty": nonempty == 0}


def _process(video_id, key):
    result = {"video_id": video_id}
    try:
        timing = _timing_and_state(video_id)
        if timing is None:
            return {**result, "status": "missing-video"}
        duration, start, ready = timing
        if ready:
            return {**result, "status": "skipped-ready"}
        aggregate, counts = _aggregate(key, duration, start)
        conn = get_conn()
        try:
            with conn.cursor() as cur:
                written = upsert_stream_stats(
                    cur, video_id, aggregate, preserve_ready=True)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        status_name = "processed" if written else "skipped-ready"
        log.info("legacy stream statistics import complete",
                 extra={"video_id": video_id, "status": status_name, **counts})
        return {**result, "status": status_name, **counts}
    except Exception as exc:
        log.warning("legacy stream statistics import failed",
                    extra={"video_id": video_id,
                           "error_type": type(exc).__name__})
        return {**result, "status": "failed", "error":
                f"{type(exc).__name__}: {str(exc)[:300]}"}
    finally:
        try:
            client("s3").delete_object(Bucket=BUCKET, Key=key)
        except Exception:
            log.exception("could not delete legacy import object",
                          extra={"video_id": video_id, "key": key})
