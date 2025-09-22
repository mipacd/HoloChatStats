import json
import gzip
import time
from pathlib import Path

def format_seconds(seconds: float) -> str:
    """Converts seconds into a human-readable HH:MM:SS format."""
    if seconds is None:
        return "N/A"
    delta = time.gmtime(seconds)
    return time.strftime("%H:%M:%S", delta)

def verify_chat_durations_efficiently():
    """
    More efficient check that iterates through existing chat logs and looks up
    the corresponding video metadata.
    """
    project_root = Path(__file__).parent
    videos_cache_path = project_root / 'cache' / 'videos'
    chat_logs_path = project_root / 'cache' / 'chat_logs'
    
    if not videos_cache_path.exists() or not chat_logs_path.exists():
        print("❌ ERROR: 'cache/videos' or 'cache/chat_logs' directory not found.")
        return

    # --- Step 1: Load ALL video metadata into a single dictionary for fast lookups ---
    print("Pre-loading all video metadata from cache...")
    all_video_metadata = {}
    for channel_file in videos_cache_path.glob('*.json'):
        try:
            with open(channel_file, 'r', encoding='utf-8') as f:
                videos_data = json.load(f)
                all_video_metadata.update(videos_data)
        except json.JSONDecodeError:
            print(f"⚠️ Warning: Could not parse JSON for {channel_file.name}. Skipping.")
            continue
    print(f"--> Metadata for {len(all_video_metadata)} videos loaded.")

    print("\n🔍 Starting verification of existing chat logs...")
    discrepancy_count = 0

    # --- Step 2: Iterate directly through the chat log files that exist ---
    for log_file in chat_logs_path.glob('*.jsonl.gz'):
        video_id = log_file.stem.split('.')[0]

        # Look up metadata for this video
        metadata = all_video_metadata.get(video_id)
        if not metadata or 'duration' not in metadata:
            continue # Skip if no metadata or duration is found

        video_duration = metadata['duration']
        if video_duration is None:
            continue

        # --- The rest of the logic is the same as before ---
        first_timestamp = None
        last_timestamp = None
        try:
            with gzip.open(log_file, 'rt', encoding='utf-8') as cf:
                for line in cf:
                    if line.strip().startswith('{'):
                        message_data = json.loads(line)
                        ts = message_data.get('timestamp')
                        if ts:
                            if first_timestamp is None:
                                first_timestamp = ts
                            last_timestamp = ts
        except Exception as e:
            print(f"⚠️ Warning: Could not read or parse {log_file.name}. Error: {e}")
            continue

        if first_timestamp is None or last_timestamp is None:
            continue

        chat_duration = (last_timestamp - first_timestamp) / 1_000_000
        discrepancy = video_duration - chat_duration

        if discrepancy >= 30:
            discrepancy_count += 1
            print(f"\nDiscrepancy found in: {log_file.name}")
            print(f"  - Video Duration:  {format_seconds(video_duration)} ({video_duration:.2f}s)")
            print(f"  - Chat Duration:   {format_seconds(chat_duration)} ({chat_duration:.2f}s)")
            print(f"  - Difference:      Chat log is {discrepancy:.2f}s shorter")

    print(f"\n--- ✅ Verification complete. Found {discrepancy_count} logs with significant discrepancies. ---")


if __name__ == "__main__":
    verify_chat_durations_efficiently()