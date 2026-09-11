from datetime import datetime, timedelta
from config import settings
import logging
import redis

# Standard Redis connection
log = logging.getLogger(__name__)
r = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT,
                decode_responses=True, socket_connect_timeout=0.5,
                socket_timeout=0.5, retry_on_timeout=False)

def is_rate_limited(user_key: str, admin: bool = False) -> bool:
    """
    Checks if a user has exceeded their daily request limit.
    This function also handles the incrementing to ensure atomicity.

    Returns:
        bool: True if the user is over the limit, False otherwise.
    """
    if admin:
        return False # Admins are not rate-limited

    # Create a key that is unique for the user for the current day (UTC)
    # This acts as a fixed-window counter.
    today = datetime.utcnow().strftime("%Y-%m-%d")
    redis_key = f"llm_usage:{user_key}:{today}"

    # Use a pipeline to execute commands atomically (prevents race conditions)
    pipe = r.pipeline()
    
    # Command 1: Increment the user's count for today
    pipe.incr(redis_key)
    
    # Command 2: Set the key to expire in 24 hours on the first increment
    # This cleans up old keys automatically. TTL is in seconds.
    pipe.expire(redis_key, timedelta(hours=24))
    
    # Execute the pipeline and get the results
    # The result of INCR will be at index 0
    try:
        current_usage, _ = pipe.execute()
    except redis.RedisError as exc:
        # Availability wins over quota enforcement during a cache restart.
        # The proxy still supplies a hashed user key; no prompt content is
        # logged or persisted here.
        log.warning("LLM rate-limit store unavailable; failing open: %s",
                    type(exc).__name__)
        return False

    # Return True if the user is over their limit
    return int(current_usage) > settings.LLM_DAILY_LIMIT

def get_remaining_prompts(user_key: str, exempt: bool = False) -> int:
    """
    Gets the number of remaining prompts for a user today.

    Args:
        user_key: The hashed user identifier.

    Returns:
        int: Number of prompts remaining (minimum 0).
    """
    if exempt:
        return settings.LLM_DAILY_LIMIT
    today = datetime.utcnow().strftime("%Y-%m-%d")
    redis_key = f"llm_usage:{user_key}:{today}"
    
    try:
        current_usage = r.get(redis_key)
    except redis.RedisError as exc:
        log.warning("LLM quota store unavailable; returning full quota: %s",
                    type(exc).__name__)
        return settings.LLM_DAILY_LIMIT
    
    if current_usage is None:
        return settings.LLM_DAILY_LIMIT
    
    remaining = settings.LLM_DAILY_LIMIT - int(current_usage)
    return max(0, remaining)


def record_prompt_usage() -> None:
    """Record one accepted Eri prompt in an aggregate UTC daily counter.

    This deliberately contains no user identifier or prompt content. Retain
    enough daily buckets for the admin dashboard's current-month total.
    """
    today = datetime.utcnow().strftime("%Y-%m-%d")
    key = f"llm_usage_total:{today}"
    try:
        pipe = r.pipeline()
        pipe.incr(key)
        pipe.expire(key, timedelta(days=62))
        pipe.execute()
    except redis.RedisError as exc:
        # Usage telemetry must never make the assistant unavailable.
        log.warning("Eri usage counter unavailable: %s", type(exc).__name__)
