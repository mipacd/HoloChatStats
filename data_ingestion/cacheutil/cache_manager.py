from collections import defaultdict
from psycopg2.extras import execute_values
from datetime import timezone, datetime
import json
import gzip
import os
import psycopg2
import sys
import shutil
from utils.logging_utils import get_logger
from config.settings import get_config
from utils.helpers import is_video_past, get_ignore_list
from utils.feature_analysis import has_humor, get_feature_timestamps, update_feature_timestamps
from db.connection import get_db_connection, release_db_connection
from db.queries import is_metadata_and_chat_log_processed, insert_video_metadata


# Insert chat log from cache into database
def insert_chat_log_from_cache(channel_id, video_id):
    logger = get_logger()
    logger.info(f"Inserting chat log for {video_id} into database.")

    chat_log_path = os.path.join(get_config("Settings", "CacheDir"), "chat_logs", f"{video_id}.jsonl.gz")
    chat_log = []

    with gzip.open(chat_log_path, "rt", encoding="utf-8") as f:
        for line in f:
            chat_log.append(json.loads(line.strip()))

    chat_counts = defaultdict(int)
    message_category_counts = defaultdict(int)
    last_message_at = defaultdict(int)
    username_map = {}
    membership_rank_map = {}
    user_ids = set()
    message_category_by_user = defaultdict(lambda: defaultdict(int))
    last_message_time = 0
    feature_dict = defaultdict(list)

    for message in chat_log:
        user_id = message["user_id"]
        username = message["username"]
        message_time = message["timestamp"]
        membership_rank = message["membership_rank"]

        if membership_rank != -2:
             # Store valid ranks
             membership_rank_map[user_id] = membership_rank
        elif user_id not in membership_rank_map:
             # Default to -2 only if we haven't seen a valid rank yet
             membership_rank_map[user_id] = -2

        message_category = message["message_category"]

        user_ids.add(user_id)
        chat_counts[user_id] += 1
        message_category_counts[message_category] += 1
        last_message_at[user_id] = message_time
        username_map[user_id] = username
        membership_rank_map[user_id] = membership_rank
        message_category_by_user[user_id][message_category] += 1
        last_message_time = max(last_message_time, message_time)

        if has_humor(message["message"]):
            feature_dict["humor"].append((message_time / 1_000_000, 1))

    feature_dict["humor"].append((last_message_time / 1_000_000, 0))

    conn = get_db_connection()
    cursor = conn.cursor()

    # Batch insert user data
    user_data_batch = [
        (user_id, channel_id, datetime.fromtimestamp(last_message_at[user_id] / 1_000_000, timezone.utc), video_id,
         membership_rank_map[user_id], message_category_by_user[user_id]["jp"], message_category_by_user[user_id]["kr"],
         message_category_by_user[user_id]["ru"], message_category_by_user[user_id]["emoji"],
         message_category_by_user[user_id]["es_en_id"], chat_counts[user_id])
        for user_id in user_ids
    ]

    execute_values(cursor, """
        INSERT INTO user_data (
            user_id, channel_id, last_message_at, video_id, membership_rank, 
            jp_count, kr_count, ru_count, emoji_count, es_en_id_count, total_message_count
        )
        VALUES %s
        ON CONFLICT (user_id, channel_id, last_message_at, video_id) DO UPDATE
        SET total_message_count = EXCLUDED.total_message_count,
            membership_rank = EXCLUDED.membership_rank,
            jp_count = EXCLUDED.jp_count,
            kr_count = EXCLUDED.kr_count,
            ru_count = EXCLUDED.ru_count,
            emoji_count = EXCLUDED.emoji_count,
            es_en_id_count = EXCLUDED.es_en_id_count;
    """, user_data_batch)

    # Batch insert users
    user_batch = [(user_id, username_map[user_id]) for user_id in user_ids]
    execute_values(cursor, """
        INSERT INTO users (user_id, username)
        VALUES %s
        ON CONFLICT (user_id) DO UPDATE
        SET username = EXCLUDED.username;
    """, user_batch)

    update_feature_timestamps(video_id, get_feature_timestamps(feature_dict))

    conn.commit()
    cursor.close()
    release_db_connection(conn)

    logger.info(f"✅ Chat log for {video_id} inserted into database.")

    return last_message_time

# Write metadata to cache / append to existing metadata
def write_metadata_to_cache(channel_id, video_id, title, end_time, duration):
    """
    Write video metadata to cache, appending to existing metadata if it exists.

    If the file does not exist, it will be created. If it does exist, it will be overwritten with the new metadata.

    Args:
        channel_id (str): The ID of the channel.
        video_id (str): The ID of the video.
        title (str): The title of the video.
        end_time (str): The end time of the video in ISO 8601 format.
        duration (int): The duration of the video in seconds.

    Returns:
        dict: The written metadata.
    """

    cache_dir = get_config("Settings", "CacheDir")
    metadata_dir = os.path.join(cache_dir, "videos")
    metadata_path = os.path.join(metadata_dir, f"{channel_id}.json")

    if not os.path.exists(metadata_dir):
        os.makedirs(metadata_dir)

    if not os.path.exists(metadata_path):
        metadata = {}
    else:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    # Ensure duration is numeric (seconds)
    numeric_duration = duration
    if hasattr(duration, 'total_seconds'):  # Handle timedelta/interval
        numeric_duration = duration.total_seconds()
    elif not isinstance(duration, (int, float)):  # Handle None/other types
        numeric_duration = 0
        
    metadata[video_id] = {
        "title": title,
        "end_time": end_time,
        "duration": numeric_duration
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    return {
        "channel_id": channel_id,
        "video_id": video_id,
        "title": title, 
        "end_time": end_time,
        "duration": duration        
    }

# Write chat log to cache. Overwrite if it already exists.
def write_chat_log_to_cache(channel_id, video_id, chat_log):
    chat_log_path = os.path.join(get_config("Settings", "CacheDir"), "chat_logs", f"{video_id}.jsonl.gz")
    with gzip.open(chat_log_path, "wt", encoding="utf-8") as f:
        for message in chat_log:
            f.write(json.dumps(message, ensure_ascii=False) + "\n")

def process_cache_dir(download_queue, year, month):
    logger = get_logger()
    """
    Process the cache directory and ensure that both metadata and chat logs are inserted into the database.
    
    This function iterates through all the video metadata files in the cache directory, checks if the corresponding
    chat log exists, and if not, adds it to the download queue. If both the metadata and chat log exist, it ensures
    that they are inserted into the database. If the metadata is missing, it inserts the metadata into the database.
    If the chat log is missing, it inserts the chat log into the database and sets has_chat_log to true. If the chat
    log is missing and the metadata is present, it sets has_chat_log to false.
    
    Args:
        None
    
    Returns:
        None
    """

    logger.info("Processing cache directory...")
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), get_config("Settings", "CacheDir"))
    metadata_dir = os.path.join(cache_dir, "videos")
    chat_log_dir = os.path.join(cache_dir, "chat_logs")
    channels = load_channels()
    # Load and check video metadata
    for filename in os.listdir(metadata_dir):
        if filename.endswith(".json"):
            channel_id = filename.replace(".json", "")
            metadata_path = os.path.join(metadata_dir, filename)
            # Load all video ids from metadata json file keys
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode JSON from {metadata_path}: {e}")
                sys.exit(1)

            # Check each video in the metadata file
            for video_id, video_info in data.items():

                # Skip video if not in month processed
                if not video_info["end_time"].startswith(f"{year}-{month:02d}"):
                    continue
                
                # Skip video if in ignore list
                ignore_list = get_ignore_list()
                if video_id in ignore_list:
                    continue

                chat_log_path = os.path.join(chat_log_dir, f"{video_id}.jsonl.gz")

                chat_log_exists = os.path.exists(chat_log_path) and os.path.getsize(chat_log_path) > 0
                chat_log_in_db, metadata_in_db = is_metadata_and_chat_log_processed(video_id)

                # Add video to download queue if:
                # 1. Chat log is missing from cache AND
                # 2. Not already marked complete in database AND
                # 3. Not already in download queue AND
                # 4. Matches target date range AND
                # 5. Video is past status
                if video_info["end_time"] is None:
                    logger.warning(f"Video {video_id} in channel {channel_id} has no end time. Aborting.")
                    sys.exit(1)
                if (not chat_log_exists and 
                    not chat_log_in_db and
                    (channel_id, video_id) not in download_queue and 
                    video_info["end_time"].startswith(f"{year}-{month:02d}") and 
                    is_video_past(video_id)):
                    download_queue.append((channel_id, video_id))
                    logger.info(f"Chat log missing for {video_id}. Added to download queue.")

                # Insert missing metadata into the database
                if not metadata_in_db:
                    insert_video_metadata(
                        channel_id,
                        video_id,
                        video_info["title"],
                        video_info["end_time"],
                        video_info["duration"]
                    )
                    logger.info(f"Metadata missing for {video_id}. Inserted into database.")

                # Process cached chat log if it's missing from DB
                if chat_log_exists and not chat_log_in_db:
                    last_message_time = insert_chat_log_from_cache(channel_id, video_id)

                    # Ensure `has_chat_log` is correctly updated in the database
                    insert_video_metadata(
                        channel_id,
                        video_id,
                        video_info["title"],
                        video_info["end_time"],
                        video_info["duration"],
                        has_chat_log=(last_message_time is not None)  # Sets True if last message exists
                    )
                    logger.info(f"Chat log processed for {video_id}. Database updated.")

    return download_queue

# Process channels from JSON
def load_channels():
    """
    Load channels from a JSON file and sync them with the database.

    This function reads channel data from a specified JSON file, where channels can
    optionally be grouped. It parses the data into a list of tuples containing channel
    ID, name, and group information, and then inserts or updates this data in the database.

    Returns:
        dict: A dictionary mapping channel names to their respective IDs.
    """
    channels_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    channels_file = os.path.join(channels_dir, get_config("Settings", "ChannelsFile"))
    with open(channels_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    channels = {}
    parsed_channels = []
    
    for group, group_channels in data.items():
        if isinstance(group_channels, dict):  # It's a group
            for name, channel_id in group_channels.items():
                if not name.startswith("_"):
                    parsed_channels.append((channel_id, name, group))
                    channels[name] = channel_id
        else:  # No group assigned
            if not group.startswith("_"):
                parsed_channels.append((group_channels, group, None))
                channels[group] = group_channels

    conn = get_db_connection()
    cursor = conn.cursor()

    psycopg2.extras.execute_values(
        cursor,
        "INSERT INTO channels (channel_id, channel_name, channel_group) VALUES %s "
        "ON CONFLICT (channel_id) DO UPDATE SET channel_name = EXCLUDED.channel_name, channel_group = EXCLUDED.channel_group;",
        parsed_channels
    )

    conn.commit()
    cursor.close()
    release_db_connection(conn)

    return channels

def gzip_uncompressed_chat_logs():
    chat_log_dir = os.path.join(get_config("Settings", "CacheDir"), "chat_logs")
    logger = get_logger()
    for filename in os.listdir(chat_log_dir):
        file_path = os.path.join(chat_log_dir, filename)
        if filename.endswith(".jsonl"):
            gzipped_path = file_path + ".gz"
            with open(file_path, "rb") as f_in, gzip.open(gzipped_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            os.remove(file_path)
            logger.info(f"Gzipped uncompressed chat log: {filename}")
