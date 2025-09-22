# Debugging script
import argparse
import json
import gzip
import sys
from pathlib import Path
from datetime import datetime

def find_user_messages(channel_id: str, username: str, month_str: str):
    """
    Scans the JSON cache to find and print all messages from a specific user
    on a given channel for a particular month.

    Args:
        channel_id (str): The ID of the channel to search within.
        username (str): The username to search for (case-insensitive).
        month_str (str): The month to filter by, in "YYYY-MM" format.
    """
    project_root = Path(__file__).parent
    videos_cache_path = project_root / 'cache' / 'videos'
    chat_logs_path = project_root / 'cache' / 'chat_logs'
    
    found_message_count = 0

    print(f"🔍 Searching for messages by '{username}' on channel '{channel_id}' for month '{month_str}'...")

    # --- Step 1: Find all videos for the given channel and month ---
    videos_to_scan = []
    channel_metadata_file = videos_cache_path / f"{channel_id}.json"

    if not channel_metadata_file.exists():
        print(f"❌ ERROR: Metadata cache file not found at: {channel_metadata_file}")
        sys.exit(1)

    try:
        with open(channel_metadata_file, 'r', encoding='utf-8') as f:
            video_data = json.load(f)
        
        for video_id, metadata in video_data.items():
            # Ensure metadata is a dictionary with 'end_time'
            if isinstance(metadata, dict) and 'end_time' in metadata:
                end_time = datetime.fromisoformat(metadata['end_time'])
                if end_time.strftime('%Y-%m') == month_str:
                    videos_to_scan.append(video_id)
    except json.JSONDecodeError:
        print(f"❌ ERROR: Could not parse JSON from {channel_metadata_file}")
        sys.exit(1)

    if not videos_to_scan:
        print("\n--- No videos found for that channel and month in the cache. ---")
        return

    print(f"Found {len(videos_to_scan)} videos to scan for chat logs...")

    # --- Step 2: Scan the chat log for each relevant video ---
    for video_id in videos_to_scan:
        log_file = chat_logs_path / f"{video_id}.jsonl.gz"
        if not log_file.exists():
            continue

        messages_in_this_video = []
        try:
            with gzip.open(log_file, 'rt', encoding='utf-8') as f:
                for line in f:
                    message_data = json.loads(line)
                    msg_username = message_data.get('username')
                    
                    if msg_username and msg_username == username:
                        # Convert UTC microsecond timestamp to a readable string
                        dt_object = datetime.fromtimestamp(message_data['timestamp'] / 1_000_000)
                        human_time = dt_object.strftime('%Y-%m-%d %H:%M:%S')
                        
                        messages_in_this_video.append(
                            f"  [{human_time}] {message_data.get('message')}"
                        )
        except Exception as e:
            print(f"\n⚠️ Warning: Could not process log file for {video_id}. Error: {e}")
            continue

        if messages_in_this_video:
            print(f"\n--- Messages found in video {video_id} ---")
            for line in messages_in_this_video:
                print(line)
            found_message_count += len(messages_in_this_video)

    print(f"\n--- ✅ Search complete. Found a total of {found_message_count} messages. ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find all chat messages for a user on a channel in a given month from the JSON cache."
    )
    parser.add_argument("channel_id", help="The channel ID to search within (e.g., UC-hM6YJuNYVAmUWxeIr9FeA).")
    parser.add_argument("username", help="The username to search for (e.g., 'eternal').")
    parser.add_argument("month", help="The month to search within, in YYYY-MM format (e.g., '2025-08').")
    
    args = parser.parse_args()

    find_user_messages(args.channel_id, args.username, args.month)