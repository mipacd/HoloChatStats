"""Targeted web-cache invalidation after publishing a completed month."""
import os

import redis


# These API results span months, use a rolling window, or summarize the entire
# live dataset. Month-qualified caches are immutable after publication and are
# deliberately retained. Operational keys (rate limits, metrics, locks, queue
# state) do not match any of these application-specific patterns.
FINALIZED_MONTH_PATTERNS = (
    "channel_recommendations:*",
    "attrition_rates_*",
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
    "published_coverage_v2:*",
)

# A request can reach a month while it is still staging and cache an empty or
# partial result. Delete only that just-published month's variants. Older
# finalized-month keys remain untouched forever.
FINALIZED_MONTH_KEY_TEMPLATES = (
    "channel_clustering_vxd_{month}_*",
    "content_clustering_v2x_{month}_*",
    "community_graph_xcAZSdf_{month}_*",
    "common_users_*_{month}_*",
    "common_users_*_{month}",
    "common_matrix_percent_*_{month}",
    "common_members_*_{month}_*",
    "common_members_*_{month}",
    "group_membership_data_*_{month}",
    "group_membership_summary_*_{month}_*",
    "group_membership_changes_*_{month}",
    "group_streaming_hours_diff_*_{month}",
    "group_total_streaming_hours_*_{month}",
    "group_avg_streaming_hours_*_{month}",
    "group_max_streaming_hours_*_{month}",
    "group_chat_makeup_*_{month}",
    "chat_engagement_{month}_*",
    "chat_leaderboard_*_{month}",
    "user_changes_*_{month}",
    "funniest_timestamps_*_{month}",
    "user_info_*_{month}",
    "video_highlights_*_{month}",
    "recommendation_monthly_data:{month}",
)


def invalidate_finalized_month_caches(batch_size=200, finalized_month=None,
                                      request_warm=True):
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
    patterns = list(FINALIZED_MONTH_PATTERNS)
    if finalized_month is not None:
        month = str(finalized_month)[:7]
        patterns.extend(p.format(month=month)
                        for p in FINALIZED_MONTH_KEY_TEMPLATES)
    for pattern in patterns:
        for key in client.scan_iter(match=pattern, count=batch_size):
            pending.append(key)
            if len(pending) >= batch_size:
                removed += client.delete(*pending)
                pending.clear()
    if pending:
        removed += client.delete(*pending)
    if finalized_month is not None and request_warm:
        client.set("cache_warm:requested_month", str(finalized_month))
    return removed
