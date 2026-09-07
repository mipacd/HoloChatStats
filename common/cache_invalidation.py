"""Targeted web-cache invalidation after publishing a completed month."""
import os

import redis


# These API results span months, use a rolling window, or summarize the entire
# live dataset. Month-qualified caches are immutable after publication and are
# deliberately retained. Operational keys (rate limits, metrics, locks, queue
# state) do not match any of these application-specific patterns.
FINALIZED_MONTH_PATTERNS = (
    "channel_recommendations:*",
    "monthly_streaming_hours_*",
    "exclusive_chat_users_*",
    "message_type_percents_*",
    "jp_user_percent_*",
    "stream_frequency_*",
    "stream_calendar_*",
    "channel_names",
    "date_ranges",
    "number_of_chat_logs",
    "num_messages",
)


def invalidate_finalized_month_caches(batch_size=200):
    """Delete aggregate web caches with SCAN, returning the number removed.

    Errors intentionally propagate. The merge handler only advances its
    durable invalidation watermark after this succeeds, so the next scheduled
    merge invocation retries a transient ElastiCache failure.
    """
    host = os.environ.get("REDIS_HOST") or os.environ.get("ELASTICACHE_HOST")
    port = int(os.environ.get("REDIS_PORT")
               or os.environ.get("ELASTICACHE_PORT", "6379"))
    if not host:
        raise RuntimeError("REDIS_HOST/ELASTICACHE_HOST is not configured")
    client = redis.Redis(host=host, port=port, socket_connect_timeout=5,
                         socket_timeout=10)
    removed = 0
    pending = []
    for pattern in FINALIZED_MONTH_PATTERNS:
        for key in client.scan_iter(match=pattern, count=batch_size):
            pending.append(key)
            if len(pending) >= batch_size:
                removed += client.delete(*pending)
                pending.clear()
    if pending:
        removed += client.delete(*pending)
    return removed
