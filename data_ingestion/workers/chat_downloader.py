from collections import defaultdict
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


def download_chat_log(channel_id, video_id, queue, year, month):
    logger = get_logger()

    duration_seconds = None
    try:
        cache_file = os.path.join(
            os.path.dirname(os.path.abspath(sys.argv[0])), 
            'cache', 'videos', f'{channel_id}.json'
        )
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                if video_id in metadata and 'duration' in metadata[video_id]:
                    duration_seconds = metadata[video_id]['duration']
    except Exception as e:
        logger.warning(f"Could not read duration from cache for {video_id}: {e}")

    retry_delay = 5
    retry_count = 0
    while retry_count < int(get_config("Settings", "MaxRetries")):
        try:
            logger.info(f"Downloading chat log for {video_id}. Retry count: {retry_count} Queue size: {str(queue.qsize())}")
            chat = iter_youtube_chat(video_id)

            last_user_message = defaultdict(int)
            message_category_by_user = defaultdict(lambda: defaultdict(int))
            membership_rank_map = {}  # Only stores known ranks (from badges)
            message_category_counts = defaultdict(int)
            feature_dict = defaultdict(list)
            chat_counts = defaultdict(int)
            username_map = {}
            last_message_at = defaultdict(int)
            user_ids = set()
            last_message_time = 0
            chat_log_entries = []
            
            # NEW: Track users who only have gift membership events (no regular messages with badges)
            gift_only_users = set()
            users_with_known_rank = set()

            for message in chat:
                author = message["author"]
                user_id = author.get("id")
                username = author.get("name")
                badges = author.get("badges")
                chat_message = message["message"]
                timestamp = message["timestamp"]
                message_type = message.get("message_type", "chat")

                if message_type in ("new_member", "gift_member"):
                    username_map[user_id] = username
                    user_ids.add(user_id)
                    
                    if message_type == "gift_member" and not badges:
                        # Gift membership with no badge info - rank is unknown
                        membership_rank = None
                        gift_only_users.add(user_id)
                    else:
                        # new_member or gift_member with badges - we have rank info
                        membership_rank_text = badges[0].lower() if badges else ""
                        membership_rank = parse_membership_rank(membership_rank_text)
                        # Only update rank map if we have valid rank info
                        membership_rank_map[user_id] = membership_rank
                        users_with_known_rank.add(user_id)
                    
                    last_message_at[user_id] = timestamp
                    last_message_time = max(last_message_time, timestamp)
                    
                    chat_log_entries.append({
                        "user_id": user_id,
                        "username": username,
                        "timestamp": timestamp,
                        "membership_rank": membership_rank,
                        "message_category": None,
                        "message": "",
                        "message_type": message_type,
                        "gifter": message.get("gifter") if message_type == "gift_member" else None
                    })
                    continue

                if not chat_message or not isinstance(chat_message, str):
                    continue
                elif chat_message.strip() == "":
                    continue

                username_map[user_id] = username
                user_ids.add(user_id)

                membership_rank_text = badges[0].lower() if badges else ""
                membership_rank = parse_membership_rank(membership_rank_text)
                membership_rank_map[user_id] = membership_rank
                users_with_known_rank.add(user_id)

                message_category = categorize_message(chat_message)
                if not message_category:
                    continue
                message_category_by_user[user_id][message_category] += 1
                message_category_counts[message_category] += 1

                last_user_message[user_id] = timestamp

                if has_humor(chat_message):
                    feature_dict["humor"].append((timestamp, 1))

                chat_counts[user_id] += 1
                last_message_at[user_id] = max(last_message_at[user_id], timestamp)
                last_message_time = max(last_message_time, timestamp)

                chat_log_entries.append({
                    "user_id": user_id,
                    "username": username,
                    "timestamp": timestamp,
                    "membership_rank": membership_rank,
                    "message_category": message_category,
                    "message": chat_message
                })

            if duration_seconds is not None:
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

            write_chat_log_to_cache(channel_id, video_id, chat_log_entries)

            conn = get_db_connection()
            cursor = conn.cursor()
            
            for user_id in user_ids:
                while queue.full():
                    logger.info(f"Queue full. Waiting for space...")
                    time.sleep(1)
                
                # Determine if this user only has gift membership data (no known rank)
                is_gift = user_id in gift_only_users and user_id not in users_with_known_rank
                
                # Get membership rank: use known rank if available, else None for gift-only users
                final_membership_rank = membership_rank_map.get(user_id)  # Returns None if not present
                
                queue.put((
                    user_id,
                    username_map[user_id],
                    channel_id,
                    datetime.fromtimestamp(last_message_at[user_id], timezone.utc),
                    video_id,
                    final_membership_rank,
                    message_category_by_user[user_id]["jp"],
                    message_category_by_user[user_id]["kr"],
                    message_category_by_user[user_id]["ru"],
                    message_category_by_user[user_id]["emoji"],
                    message_category_by_user[user_id]["es_en_id"],
                    chat_counts[user_id],
                    observed_month,
                    is_gift  # NEW: Pass is_gift flag
                ))

            cursor.execute("""
            UPDATE videos SET has_chat_log = TRUE WHERE video_id = %s;
            """, (video_id,))

            conn.commit()
            cursor.close()
            release_db_connection(conn)
            update_feature_timestamps(video_id, get_feature_timestamps(feature_dict))

            del chat
            gc.collect()

            break
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