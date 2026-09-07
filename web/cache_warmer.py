"""Sequential background warming for bounded, expensive web API caches."""
import logging
import os
import time

import redis
from sqlalchemy import text


log = logging.getLogger("cache_warmer")
REQUESTED_KEY = "cache_warm:requested_month"
COMPLETED_KEY = "cache_warm:completed_month"
COMPLETED_AT_KEY = "cache_warm:completed_at"


def build_tasks(channels, groups, finalized_month):
    """Return bounded API calls ordered by user-visible payoff."""
    month = str(finalized_month)[:7]
    # Exclusive Chat is intentionally first: it is the known timeout-prone UI.
    tasks = [
        ("/api/get_exclusive_chat_users", {"channel": channel})
        for channel in channels
    ]
    tasks.extend([
        ("/api/get_channel_names", {}),
        ("/api/get_date_ranges", {}),
        ("/api/get_number_of_chat_logs", {}),
        ("/api/get_num_messages", {}),
    ])
    tasks.extend(("/api/get_jp_user_percent", {"channel": channel})
                 for channel in channels)
    tasks.extend(
        ("/api/get_message_type_percents",
         {"channel": channel, "language": language})
        for channel in channels for language in ("EN", "JP", "KR", "RU")
    )
    tasks.extend(
        ("/api/get_monthly_streaming_hours",
         {"channel": channel, "include_forecast": "true"})
        for channel in channels
    )
    tasks.extend(
        (path, {"channel_name": channel, "month": month})
        for channel in channels
        for path in ("/api/get_chat_leaderboard", "/api/get_video_highlights")
    )
    tasks.extend(
        ("/api/get_funniest_timestamps", {"channel": channel, "month": month})
        for channel in channels
    )
    # These views have a bounded channel selector. Only their default UTC
    # variants are warmed; arbitrary timezone/year combinations remain lazy.
    tasks.extend(
        (path, {"channel_name": channel})
        for channel in channels
        for path in ("/api/get_stream_frequency", "/api/get_stream_calendar")
    )
    tasks.extend([
        ("/api/channel_clustering",
         {"month": month, "percentile": "95", "type": "2d"}),
        ("/api/content_clustering",
         {"month": month, "percentile": "95", "type": "2d"}),
        ("/api/community_graph",
         {"month": month, "include_edges": "true", "channel_group": "all"}),
    ])
    # The initial page load omits the optional group selector, producing a
    # different cache key from each named group.
    for path in (
        "/api/get_group_chat_makeup",
        "/api/get_chat_engagement",
        "/api/get_group_total_streaming_hours",
        "/api/get_group_avg_streaming_hours",
        "/api/get_group_max_streaming_hours",
        "/api/get_group_streaming_hours_diff",
    ):
        tasks.append((path, {"month": month}))
    for group in groups:
        common = {"group": group, "month": month}
        tasks.extend([
            ("/api/get_group_chat_makeup", common),
            ("/api/get_chat_engagement", common),
            ("/api/get_group_total_streaming_hours", common),
            ("/api/get_group_avg_streaming_hours", common),
            ("/api/get_group_max_streaming_hours", common),
            ("/api/get_group_streaming_hours_diff", common),
            ("/api/get_group_membership_data",
             {"channel_group": group, "month": month}),
            ("/api/get_group_membership_changes",
             {"channel_group": group, "month": month}),
            ("/api/get_user_changes", common),
        ])
    return tasks


def _database_inputs(app):
    """Return the newest published month and active finite selectors."""
    from models import db

    with app.app_context():
        latest = db.session.execute(text(
            "SELECT COALESCE("
            "  (SELECT MAX(observed_month) FROM monthly_merge_state "
            "   WHERE status = 'merged'), "
            "  (SELECT (date_trunc('month', value::timestamptz) "
            "           - INTERVAL '1 month')::date "
            "   FROM service_config WHERE key = 'backlog_floor')"
            ")"
        )).scalar()
        rows = db.session.execute(text(
            "SELECT channel_name, channel_group FROM channels "
            "WHERE active ORDER BY channel_name"
        )).all()
        db.session.remove()
    channels = sorted({row[0] for row in rows if row[0]})
    groups = sorted({row[1] for row in rows if row[1]})
    return latest, channels, groups


def warm_once(app, finalized_month):
    """Warm one month sequentially; cached successes make retries inexpensive."""
    _latest, channels, groups = _database_inputs(app)
    tasks = build_tasks(channels, groups, finalized_month)
    delay = float(os.environ.get("CACHE_WARM_DELAY_SECONDS", "0.25"))
    failures = []
    started = time.monotonic()
    with app.test_client() as client:
        for index, (path, query) in enumerate(tasks, 1):
            try:
                response = client.get(path, query_string=query,
                                      headers={"User-Agent": "HoloChatStats-cache-warmer/1.0"})
                if response.status_code >= 500:
                    failures.append({"path": path, "status": response.status_code})
                    log.warning("cache warm failed %s status=%s params=%s",
                                path, response.status_code, query)
                elif response.status_code >= 400:
                    # A finalized month may legitimately have no rows for an
                    # endpoint. Retrying that deterministic 4xx forever would
                    # prevent the rest of the daily cycle from completing.
                    log.info("cache warm skipped %s status=%s params=%s",
                             path, response.status_code, query)
            except Exception as exc:
                failures.append({"path": path, "error": str(exc)[:200]})
                log.exception("cache warm failed %s params=%s", path, query)
            if index % 25 == 0:
                log.info("cache warm progress month=%s completed=%s total=%s failures=%s",
                         finalized_month, index, len(tasks), len(failures))
            if delay > 0:
                time.sleep(delay)
    result = {"month": str(finalized_month), "tasks": len(tasks),
              "failures": len(failures),
              "seconds": round(time.monotonic() - started, 1)}
    log.info("cache warm complete %s", result)
    return result


def run(app):
    """Poll durable Redis markers and retry incomplete warming passes."""
    host = os.environ.get("REDIS_HOST") or os.environ.get("ELASTICACHE_HOST")
    port = int(os.environ.get("REDIS_PORT")
               or os.environ.get("ELASTICACHE_PORT", "6379"))
    if not host:
        log.error("cache warmer disabled: Redis endpoint is not configured")
        return
    store = redis.Redis(host=host, port=port, decode_responses=True,
                        socket_connect_timeout=5, socket_timeout=10)
    poll_seconds = int(os.environ.get("CACHE_WARM_POLL_SECONDS", "30"))
    retry_seconds = int(os.environ.get("CACHE_WARM_RETRY_SECONDS", "300"))
    interval_seconds = int(os.environ.get("CACHE_WARM_INTERVAL_SECONDS", "86400"))
    while True:
        try:
            requested = store.get(REQUESTED_KEY)
            if not requested:
                latest, _channels, _groups = _database_inputs(app)
                if latest:
                    requested = str(latest)
                    store.setnx(REQUESTED_KEY, requested)
            completed = store.get(COMPLETED_KEY)
            completed_at = float(store.get(COMPLETED_AT_KEY) or 0)
            refresh_due = time.time() - completed_at >= interval_seconds
            if requested and (requested != completed or refresh_due):
                result = warm_once(app, requested)
                if result["failures"] == 0:
                    store.set(COMPLETED_KEY, requested)
                    store.set(COMPLETED_AT_KEY, str(time.time()))
                else:
                    time.sleep(retry_seconds)
                    continue
        except Exception:
            log.exception("cache warmer loop failed; will retry")
            time.sleep(retry_seconds)
            continue
        time.sleep(poll_seconds)
