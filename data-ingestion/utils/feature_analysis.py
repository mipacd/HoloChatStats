import regex
import re
from collections import defaultdict
from db.connection import get_db_connection, release_db_connection

# Humorous message detection
def has_humor(message):
    """Detects if a message is humorous. This is a simple heuristic check for phrases or characters that are commonly associated with humor.

    Args:
        message (str): The message to check for humor.

    Returns:
        bool: True if the message is humorous, False otherwise.
    """
    humor_list = ["草", "茶葉", "_fbkcha", "_lol", "lmao", "lmfao", "haha", "🤣", "😆", "jaja", "笑",
                  "xd", "wkwk", "ｗ"]
    
    jp_regex = regex.compile(r"[\p{Hiragana}\p{Katakana}\p{Han}]+")
    has_jp = jp_regex.search(message) is not None
    has_w_end = has_jp and message.endswith("w")

    has_lol = re.search(r"\blol\b", message, re.IGNORECASE) is not None

    return (any(substring in message for substring in humor_list) or has_lol or has_w_end)

# Get timestamps of the greatest concentration of features
def get_feature_timestamps(feature_dict):
    """Get timestamps of the greatest concentration of features.

    This function takes a dictionary of features where each feature is associated with a list of tuples of timestamps and values.
    It then groups the timestamps into 30 second buckets and sums the values associated with each bucket.
    The function returns a list of timestamps that are the top bucket for each feature shifted 10 seconds back.

    Args:
        feature_dict (dict): A dictionary of features where each feature is associated with a list of tuples of timestamps and values.

    Returns:
        list: A list of timestamps that are the top bucket for each feature shifted 10 seconds back.
    """
    feature_buckets = defaultdict(int)
    for feature, timestamps in feature_dict.items():
        for timestamp, value in timestamps:
            bucket = int(timestamp // 30) # 30 second buckets
            feature_buckets[bucket] += value
    
    sorted_buckets = sorted(feature_buckets.items(), key=lambda x: x[1], reverse=True)
    top_timestamp = bucket * 30 - 10 if sorted_buckets else None
    # Return top bucket for each feature shifted 10 seconds back
    return {"humor": top_timestamp}

# Update video metadata with feature timestamps
def update_feature_timestamps(video_id, feature_timestamps):
    """
    Updates the funniest_timestamp field in the videos table with the given feature timestamps.

    This function takes a video_id and a dictionary of feature timestamps and updates the
    funniest_timestamp field in the videos table with the 'humor' timestamp from the dictionary.

    Args:
        video_id (str): The ID of the video to update.
        feature_timestamps (dict): A dictionary of feature timestamps where the key is the feature name
            and the value is the timestamp of the feature.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE videos
        SET funniest_timestamp = %s
        WHERE video_id = %s;
        """,
        (feature_timestamps["humor"], video_id)
    )
    conn.commit()
    cursor.close()
    release_db_connection(conn)