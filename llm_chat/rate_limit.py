from datetime import datetime, timedelta
from config import settings
import redis
import os

# Standard Redis connection
r = redis.Redis(host=os.getenv("REDIS_HOST", "redis"), port=6379, decode_responses=True)

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
    current_usage, _ = pipe.execute()

    # Return True if the user is over their limit
    return int(current_usage) > settings.LLM_DAILY_LIMIT
