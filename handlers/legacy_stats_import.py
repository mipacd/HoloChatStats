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
from common.stream_stats import (
    StreamStatsAccumulator, materially_invalid_timing, timing_outlier_limit,
    upsert_stream_stats,
)
from handlers.ingest import _iter_messages, _missing_raw


log = get_logger("legacy_stats_import")
BUCKET = os.environ["RAW_BUCKET"]
PREFIX = "legacy-stream-stats-import/"
VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
MAX_STATUS_IDS = 500
MAX_IMPORT_ITEMS = 32
MIN_REMAINING_MS = 60_000
ALIGNMENT_TOLERANCE_SECONDS = 60


class RepairBudgetExceeded(Exception):
    """The retained-raw repair should fall back to the local archive."""

_SUSPECT_SQL = """(
    (COALESCE(s.timing_source, 'metadata_derived') = 'metadata_derived'
     AND s.first_message_at < (v.end_time - COALESCE(
        NULLIF(v.duration, INTERVAL '0 seconds'),
        make_interval(secs => j.video_duration_s), INTERVAL '0 seconds'))
        - INTERVAL '60 seconds')
    OR (COALESCE(
          EXTRACT(EPOCH FROM NULLIF(v.duration, INTERVAL '0 seconds')),
          j.video_duration_s, 0) > 0
        AND COALESCE(s.message_count, 0) > 0 AND COALESCE((
        SELECT SUM(value::bigint)
        FROM jsonb_array_elements_text(COALESCE(s.histogram_counts, '[]'::jsonb))
    ), 0) = 0)
    OR COALESCE(s.out_of_range_messages, 0) > GREATEST(
        10, CEIL(COALESCE(s.message_count, 0) * 0.20))
)"""


def handler(event, context):
    event = event or {}
    action = event.get("action")
    if action == "status":
        return status(event.get("video_ids") or [],
                      include_misaligned=bool(event.get("include_misaligned")))
    if action == "import":
        return import_batch(event.get("items") or [], context,
                            replace_misaligned=bool(
                                event.get("replace_misaligned")))
    if action == "timing_audit":
        return timing_audit(event.get("cursor"), event.get("limit", 100))
    if action == "repair_retained":
        return repair_retained(event.get("video_ids") or [], context)
    raise ValueError("unsupported legacy import action")


def _video_ids(values, limit):
    if not isinstance(values, list) or len(values) > limit:
        raise ValueError(f"expected a list of at most {limit} video IDs")
    if any(not isinstance(value, str) or not VIDEO_ID_RE.fullmatch(value)
           for value in values):
        raise ValueError("invalid video ID")
    return list(dict.fromkeys(values))


def status(values, include_misaligned=False):
    """Classify IDs before upload so known work is the only data transferred."""
    video_ids = _video_ids(values, MAX_STATUS_IDS)
    if not video_ids:
        return {"ready": [], "eligible": [], "missing": []}
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(f"""SELECT v.video_id,
                               COALESCE(s.status = 'ready', FALSE),
                               COALESCE(s.status = 'ready' AND {_SUSPECT_SQL}, FALSE)
                        FROM videos v
                        LEFT JOIN ingest_jobs j USING (video_id)
                        LEFT JOIN video_stream_stats s USING (video_id)
                        WHERE v.video_id = ANY(%s)""", (video_ids,))
        rows = {row[0]: (row[1], row[2]) for row in cur.fetchall()}
    conn.rollback()
    known = set(rows)
    misaligned = [video_id for video_id in video_ids
                  if rows.get(video_id, (False, False))[1]]
    result = {
        "ready": [video_id for video_id in video_ids
                  if rows.get(video_id, (False, False))[0]
                  and (not include_misaligned or video_id not in misaligned)],
        "eligible": [video_id for video_id in video_ids
                     if video_id in known and not rows[video_id][0]],
        "missing": [video_id for video_id in video_ids if video_id not in known],
    }
    if include_misaligned:
        result["misaligned"] = misaligned
    return result


def import_batch(values, context, replace_misaligned=False):
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
        results.append(_process(video_id, key, replace_misaligned))
    return {"results": results, "deferred": deferred}


def _timing_and_state(video_id):
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(f"""SELECT v.end_time,
                               COALESCE(NULLIF(v.duration, INTERVAL '0 seconds'),
                                 make_interval(secs => j.video_duration_s)),
                               COALESCE(s.status = 'ready', FALSE),
                               COALESCE(s.status = 'ready' AND {_SUSPECT_SQL}, FALSE)
                        FROM videos v
                        LEFT JOIN ingest_jobs j USING (video_id)
                        LEFT JOIN video_stream_stats s USING (video_id)
                        WHERE v.video_id=%s""", (video_id,))
        row = cur.fetchone()
    conn.rollback()
    if not row:
        return None
    end_time, duration, ready, suspect = row
    seconds = max(0, int(duration.total_seconds())) if duration else 0
    start = ((end_time - timedelta(seconds=seconds)).timestamp()
             if end_time and seconds else None)
    return seconds, start, bool(ready), bool(suspect)


def _aggregate(key, duration, start):
    s3 = client("s3")
    response = s3.get_object(Bucket=BUCKET, Key=key)
    accumulator = StreamStatsAccumulator(duration, start, legacy_rebase=True)
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


def _process(video_id, key, replace_misaligned=False):
    result = {"video_id": video_id}
    try:
        timing = _timing_and_state(video_id)
        if timing is None:
            return {**result, "status": "missing-video"}
        duration, start, ready, suspect = timing
        if ready and not (replace_misaligned and suspect):
            return {**result, "status": "skipped-ready"}
        aggregate, counts = _aggregate(key, duration, start)
        conn = get_conn()
        try:
            with conn.cursor() as cur:
                if replace_misaligned and ready:
                    cur.execute(f"""SELECT COALESCE({_SUSPECT_SQL}, FALSE)
                                     FROM videos v
                                     LEFT JOIN ingest_jobs j USING (video_id)
                                     JOIN video_stream_stats s USING (video_id)
                                     WHERE v.video_id=%s FOR UPDATE OF s""",
                                (video_id,))
                    current = cur.fetchone()
                    if not current or not current[0]:
                        conn.rollback()
                        return {**result, "status": "skipped-ready"}
                if (replace_misaligned and ready
                        and materially_invalid_timing(aggregate, duration)):
                    conn.rollback()
                    return {**result, "status": "validation-failed",
                            "error": "replacement has materially invalid timing",
                            "message_count": aggregate["message_count"],
                            "histogram_message_count": sum(
                                aggregate["histogram_counts"]),
                            "out_of_range_messages": aggregate[
                                "out_of_range_messages"],
                            "outlier_limit": timing_outlier_limit(
                                aggregate["message_count"]),
                            "duration_seconds": duration,
                            "first_offset_seconds": aggregate[
                                "first_offset_seconds"],
                            "last_offset_seconds": aggregate[
                                "last_offset_seconds"]}
                written = upsert_stream_stats(
                    cur, video_id, aggregate,
                    preserve_ready=not (replace_misaligned and ready))
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


def timing_audit(cursor=None, limit=100):
    """Return suspect ready aggregates without reading or mutating raw data."""
    try:
        limit = max(1, min(500, int(limit)))
    except (TypeError, ValueError) as exc:
        raise ValueError("limit must be an integer") from exc
    cursor = cursor if isinstance(cursor, str) else ""
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(f"""SELECT v.video_id, j.channel_id, j.part_count,
                                j.last_offset_s,
                                EXTRACT(EPOCH FROM COALESCE(
                                  NULLIF(v.duration, INTERVAL '0 seconds'),
                                  make_interval(secs => j.video_duration_s),
                                  INTERVAL '0 seconds')),
                                s.first_message_at,
                                EXTRACT(EPOCH FROM (v.end_time - COALESCE(
                                  NULLIF(v.duration, INTERVAL '0 seconds'),
                                  make_interval(secs => j.video_duration_s),
                                  INTERVAL '0 seconds'))),
                                s.timing_source, s.message_count,
                                COALESCE((SELECT SUM(value::bigint)
                                  FROM jsonb_array_elements_text(COALESCE(
                                    s.histogram_counts, '[]'::jsonb))), 0),
                                s.out_of_range_messages,
                                s.first_offset_seconds, s.last_offset_seconds
                         FROM video_stream_stats s
                         JOIN videos v USING (video_id)
                         LEFT JOIN ingest_jobs j USING (video_id)
                         WHERE s.status='ready' AND v.video_id > %s
                           AND {_SUSPECT_SQL}
                         ORDER BY v.video_id LIMIT %s""", (cursor, limit))
        rows = cur.fetchall()
    conn.rollback()
    conn.close()
    items = [{"video_id": r[0], "channel_id": r[1],
              "part_count": int(r[2] or 0),
              "last_offset_seconds": float(r[3] or 0),
              "duration_seconds": int(r[4] or 0),
              "first_message_at": r[5].isoformat() if r[5] else None,
              "calculated_start": float(r[6]) if r[6] is not None else None,
              "timing_source": r[7], "message_count": int(r[8] or 0),
              "histogram_message_count": int(r[9] or 0),
              "out_of_range_messages": int(r[10] or 0),
              "first_offset_seconds": (float(r[11])
                                         if r[11] is not None else None),
              "stored_last_offset_seconds": (float(r[12])
                                               if r[12] is not None else None)}
             for r in rows]
    return {"items": items,
            "next_cursor": rows[-1][0] if len(rows) == limit else None}


def _retained_state(video_id):
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(f"""SELECT j.channel_id, j.part_count, j.last_offset_s,
                               COALESCE(NULLIF(EXTRACT(EPOCH FROM v.duration), 0),
                                        j.video_duration_s, 0),
                               EXTRACT(EPOCH FROM v.end_time) - COALESCE(
                                 NULLIF(EXTRACT(EPOCH FROM v.duration), 0),
                                 j.video_duration_s, 0),
                               COALESCE(s.status='ready' AND {_SUSPECT_SQL}, FALSE)
                        FROM videos v JOIN ingest_jobs j USING (video_id)
                        JOIN video_stream_stats s USING (video_id)
                        WHERE v.video_id=%s""", (video_id,))
        row = cur.fetchone()
    conn.rollback()
    conn.close()
    return row


def _repair_retained_one(video_id, context=None):
    state = _retained_state(video_id)
    if not state or not state[5]:
        return {"video_id": video_id, "status": "already-correct"}
    channel_id, part_count, last_offset, duration, metadata_start, _ = state
    if not channel_id or int(part_count or 0) < 1:
        return {"video_id": video_id, "status": "local-archive-needed"}
    s3 = client("s3")
    accumulator = StreamStatsAccumulator(
        duration, metadata_start, legacy_rebase=True,
        checkpoint_last_offset=last_offset)
    try:
        for index, message in enumerate(
                _iter_messages(s3, channel_id, video_id, int(part_count)), 1):
            if (context and index % 1000 == 0
                    and context.get_remaining_time_in_millis() < 45_000):
                raise RepairBudgetExceeded(
                    "retained raw data exceeded the Lambda time budget")
            accumulator.add(message)
    except Exception as exc:
        if _missing_raw(exc):
            return {"video_id": video_id, "status": "local-archive-needed"}
        raise
    aggregate = accumulator.finish()
    if aggregate["message_count"] and not sum(aggregate["histogram_counts"]):
        return {"video_id": video_id, "status": "validation-failed",
                "error": "recovered histogram is empty"}
    if duration and aggregate["out_of_range_messages"]:
        return {"video_id": video_id, "status": "validation-failed",
                "error": "recovered aggregate still has out-of-range timing"}
    if (aggregate["last_offset_seconds"] is not None and duration
            and aggregate["last_offset_seconds"] > float(duration) + 60):
        return {"video_id": video_id, "status": "validation-failed",
                "error": "recovered offset exceeds video duration"}
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(f"""SELECT COALESCE({_SUSPECT_SQL}, FALSE)
                             FROM videos v LEFT JOIN ingest_jobs j USING (video_id)
                             JOIN video_stream_stats s USING (video_id)
                             WHERE v.video_id=%s FOR UPDATE OF s""", (video_id,))
            current = cur.fetchone()
            if not current or not current[0]:
                conn.rollback()
                return {"video_id": video_id, "status": "already-correct"}
            upsert_stream_stats(cur, video_id, aggregate)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return {"video_id": video_id, "status": "repaired",
            "timing_source": aggregate["timing_source"]}


def repair_retained(values, context):
    video_ids = _video_ids(values, MAX_IMPORT_ITEMS)
    results, deferred = [], []
    for index, video_id in enumerate(video_ids):
        remaining = (context.get_remaining_time_in_millis()
                     if context and hasattr(context, "get_remaining_time_in_millis")
                     else 900_000)
        if remaining < MIN_REMAINING_MS:
            deferred.extend(video_ids[index:])
            break
        try:
            results.append(_repair_retained_one(video_id, context))
        except RepairBudgetExceeded as exc:
            results.append({"video_id": video_id,
                            "status": "time-budget-exceeded",
                            "error": str(exc)})
        except Exception as exc:
            log.warning("retained stream timing repair failed",
                        extra={"video_id": video_id,
                               "error_type": type(exc).__name__})
            results.append({"video_id": video_id, "status": "failed",
                            "error": f"{type(exc).__name__}: {str(exc)[:300]}"})
    return {"results": results, "deferred": deferred}
