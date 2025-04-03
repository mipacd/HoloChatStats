import regex
import re
from collections import defaultdict
from db.connection import get_db_connection, release_db_connection

# Humorous message detection
def has_humor(message):
    """Detects if a message is humorous based on common humor patterns in different languages."""
    humor_list = [
        "草", "茶葉", "_fbkcha", "_lol", "lmao", "lmfao", "haha", "🤣", "😆", 
        "jaja", "笑", "xd", "wkwk", "ｗ", "rofl", "kek", "looool", "xddd"
    ]
    
    jp_regex = regex.compile(r"[\p{Hiragana}\p{Katakana}\p{Han}]+")
    has_jp = jp_regex.search(message) is not None

    # Detect laughter ending in "w" (single or multiple)
    has_w_end = has_jp and re.search(r"ｗ+$", message)

    # Detect "lol" in various cases
    has_lol = re.search(r"\blol+\b", message, re.IGNORECASE) is not None

    # Detect extended laughter like "hahaha" or "looooool"
    repeated_laughter = re.search(r"(ha){2,}|(w{2,})|(o?l{2,}o+l+)", message, re.IGNORECASE) is not None

    return (any(substring in message for substring in humor_list) or has_lol or has_w_end or repeated_laughter)


# Get timestamps of the greatest concentration of features
def get_feature_timestamps(feature_dict):
    """Finds the single funniest moment based on humor concentration in chat."""
    feature_buckets = defaultdict(int)

    # Group timestamps into 30-second buckets
    for timestamp, value in feature_dict["humor"]:
        bucket = int(timestamp // 30)  # 30-second buckets
        feature_buckets[bucket] += value

    # Sort buckets by humor count (highest first)
    sorted_buckets = sorted(feature_buckets.items(), key=lambda x: x[1], reverse=True)

    # Extract the top bucket and shift back 10 seconds
    top_timestamp = (sorted_buckets[0][0] * 30) - 10 if sorted_buckets else None

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