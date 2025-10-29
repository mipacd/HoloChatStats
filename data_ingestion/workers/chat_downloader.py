from collections import defaultdict
#from chat_downloader import ChatDownloader, errors
from workers.yt_chat_fallback import iter_youtube_chat
from datetime import timezone, datetime
import time
import gc
import os
import sys
import json
from db.connection import get_db_connection, release_db_connection
from cacheutil.cache_manager import write_chat_log_to_cache
from config.settings import get_config
from utils.logging_utils import get_logger
from utils.feature_analysis import has_humor, get_feature_timestamps, update_feature_timestamps
from utils.chat_parser import categorize_message, parse_membership_rank


# Download chat logs from YouTube
def download_chat_log(channel_id, video_id, queue, year, month):
    """
    Downloads chat logs from YouTube for a given video and inserts them into the database.

    This function fetches chat data from a YouTube video using the ChatDownloader library,
    processes the chat messages to categorize them, and updates the database with user
    data, message counts, and language statistics. It caches the chat logs to a file
    and retries the download process on failure, up to a maximum number of retries.

    Args:
        channel_id (str): The ID of the channel the video belongs to.
        video_id (str): The ID of the video to download chat logs for.

    Returns:
        None
    """

    logger = get_logger()

    duration_seconds = None
    try:
        # Construct the path to the channel's metadata cache file
        cache_file = os.path.join(
            os.path.dirname(os.path.abspath(sys.argv[0])), 
            'cache', 'videos', f'{channel_id}.json'
        )
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                # Get the duration for the specific video we are processing
                if video_id in metadata and 'duration' in metadata[video_id]:
                    duration_seconds = metadata[video_id]['duration']
    except Exception as e:
        logger.warning(f"Could not read duration from cache for {video_id}: {e}")


    retry_delay = 5
    retry_count = 0
    while retry_count < int(get_config("Settings", "MaxRetries")):
        try:
            logger.info(f"Downloading chat log for {video_id}. Retry count: {retry_count} Queue size: {str(queue.qsize())}")
            #video_url = f"https://www.youtube.com/watch?v={video_id}"
            #cookie_path = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), get_config("Settings", "CookieFile"))
            #chat = ChatDownloader(cookies=cookie_path).get_chat(video_url, max_attempts=1, inactivity_timeout=30, message_groups=["messages", "superchat"])
            chat = iter_youtube_chat(video_id)

            last_user_message = defaultdict(int)
            message_category_by_user = defaultdict(lambda: defaultdict(int))
            membership_rank_map = {}
            message_category_counts = defaultdict(int)
            feature_dict = defaultdict(list)
            chat_counts = defaultdict(int)
            username_map = {}
            last_message_at = defaultdict(int)
            user_ids = set()
            last_message_time = 0
            chat_log_entries = []

            for message in chat:
                author = message["author"]
                user_id = author.get("id")
                username = author.get("name")
                badges = author.get("badges")
                chat_message = message["message"]
                timestamp = message["timestamp"]

                if not chat_message or not isinstance(chat_message, str):
                    continue
                elif chat_message.strip() == "":
                    continue

                username_map[user_id] = username
                user_ids.add(user_id)

                # Get membership rank
                # membership_rank_text = badges[0].get("title", "").lower() if badges else ""
                membership_rank_text = badges[0].lower() if badges else ""
                membership_rank = parse_membership_rank(membership_rank_text)
                membership_rank_map[user_id] = membership_rank

                # Get message category
                message_category = categorize_message(chat_message)
                if not message_category:
                    continue
                message_category_by_user[user_id][message_category] += 1
                message_category_counts[message_category] += 1

                # Store last message time for each user
                last_user_message[user_id] = timestamp

                # Store humor timestamps
                if has_humor(chat_message):
                    #feature_dict["humor"].append((timestamp / 1_000_000, 1))
                    feature_dict["humor"].append((timestamp, 1))

                # Store user chat counts
                chat_counts[user_id] += 1

                # Store last message time per user per video
                #last_message_at[user_id] = max(last_message_at[user_id], timestamp / 1_000_000)
                last_message_at[user_id] = max(last_message_at[user_id], timestamp)

                # Store last message time overall
                #last_message_time = max(last_message_time, timestamp / 1_000_000)
                last_message_time = max(last_message_time, timestamp)

                # Store chat log entries
                chat_log_entries.append({
                    "user_id": user_id,
                    "username": username,
                    "timestamp": timestamp,
                    "membership_rank": membership_rank,
                    "message_category": message_category,
                    "message": chat_message
                })

            if duration_seconds is not None:
                # If the last message is earlier than the video's end minus a 60s tolerance,
                # it means the download likely stalled and timed out prematurely.
                if last_message_time < (duration_seconds - 60):
                    raise TimeoutError(
                        f"Chat download for {video_id} timed out prematurely. "
                        f"Last message at {int(last_message_time)}s, video duration is {int(duration_seconds)}s."
                    )

            feature_dict["humor"].append((last_message_time, 0))
            if last_message_time > 0:
                observed_month = datetime.fromtimestamp(last_message_time).replace(day=1).date()
            else:
                observed_month = datetime(year, month, 1).date()

            # Write chat log to cache
            write_chat_log_to_cache(channel_id, video_id, chat_log_entries)

            # Write chat log to database
            conn = get_db_connection()
            cursor = conn.cursor()
            for user_id in user_ids:
                if queue.full():
                    logger.info(f"Queue full. Waiting for space...")
                    time.sleep(1)
                    continue
                # Add to database worker queue
                queue.put((
                    user_id,
                    username_map[user_id],
                    channel_id,
                    datetime.fromtimestamp(last_message_at[user_id], timezone.utc),
                    video_id,
                    membership_rank_map[user_id],
                    message_category_by_user[user_id]["jp"],
                    message_category_by_user[user_id]["kr"],
                    message_category_by_user[user_id]["ru"],
                    message_category_by_user[user_id]["emoji"],
                    message_category_by_user[user_id]["es_en_id"],
                    chat_counts[user_id],
                    observed_month
                ))

            cursor.execute("""
            UPDATE videos SET has_chat_log = TRUE WHERE video_id = %s;
            """, (video_id,))


            conn.commit()
            cursor.close()
            release_db_connection(conn)
            update_feature_timestamps(video_id, get_feature_timestamps(feature_dict))

            # Ensure chat is no longer in memory
            del chat
            gc.collect()

            break
        #except errors.NoChatReplay:
        except RuntimeError as e:
            logger.warning(f"No chat replay found for {video_id}.")
            return
        except Exception as e:
            error_message = str(e)
            critical_errors = ["members", "not available", "removed", "private"]
            if any(keyword in error_message.lower() for keyword in critical_errors):
                logger.warning(f"Unable to access video {video_id}. Error: {error_message}")
                return
            else:
                retry_count += 1
                wait_time = min(retry_delay * (2 ** retry_count), 60)
                if retry_count < int(get_config("Settings", "MaxRetries")):
                    logger.warning(f"Error downloading chat log for {video_id}. Retrying in {wait_time} seconds. Error: {error_message}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Max retries exceeded for {video_id}. Error: {error_message}")
                    return

    logger.info(f"Chat log for {video_id} downloaded and processed.")