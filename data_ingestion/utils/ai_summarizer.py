import os
import sys
import re
import json
import gzip
import datetime
import math
import pandas as pd
import requests
import psycopg2
import psycopg2.extras
from pathlib import Path

# 1. Get the absolute path to the project's root directory (one level up from 'utils')
PROJECT_ROOT = Path(__file__).parent.parent
# 2. Add the project root to Python's path
sys.path.append(str(PROJECT_ROOT))
# 3. Now we can reliably import from the config module
from config.settings import get_config

# --- Configuration ---
try:
    AI_SERVER_URL = get_config("Settings", "AIServerURL")
    DB_CONFIG = {
        "dbname": get_config("Database", "DBName"),
        "user": get_config("Database", "DBUser"),
        "password": get_config("Database", "DBPass"),
        "host": get_config("Database", "DBHost"),
        "port": get_config("Database", "DBPort")
    }
except (FileNotFoundError, KeyError) as e:
    print(f"FATAL: Could not load settings from config.ini. Error: {e}")
    sys.exit(1) # Exit if configuration is missing

CACHE_DIR = PROJECT_ROOT / "cache"
VIDEO_CACHE_DIR = CACHE_DIR / "videos"
CHAT_LOG_DIR = CACHE_DIR / "chat_logs"
STATE_FILE = PROJECT_ROOT / "resume_state.json"

# --- Helper Functions ---

def clean_chat_for_ai(text: str) -> str:
    """
    A simplified cleaner that removes custom emotes and URLs, preserving native text.
    """
    # Removes custom YouTube emotes like :pekorapain:
    text = re.sub(r':[^:\s]+:', ' ', text)
    # Removes URLs (low cost, high benefit for noise reduction)
    text = re.sub(r'https?://\S+', ' ', text)
    # Normalizes whitespace to a single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def generate_ai_results_batch(payload: list[dict]) -> list[dict] | None:
    """Sends a batch of contexts (text + channel_name) to the AI server."""
    if not payload: return []
    try:
        response = requests.post(AI_SERVER_URL, json={"contexts": payload})
        response.raise_for_status()
        result = response.json()
        if result.get("status") == "success":
            return result.get("data")
        return None
    except requests.exceptions.RequestException as e:
        print(f"FATAL: Could not connect to the AI server. Details: {e}")
        return None

def get_videos_for_month(year: int, month: int) -> dict:
    """Scans the video cache and includes the channel_id in the returned data."""
    target_videos = {}
    for channel_file in VIDEO_CACHE_DIR.glob("*.json"):
        channel_id = channel_file.stem
        with open(channel_file, 'r', encoding='utf-8') as f:
            videos_data = json.load(f)
            for video_id, metadata in videos_data.items():
                try:
                    end_time = datetime.datetime.fromisoformat(metadata['end_time'])
                    if end_time.year == year and end_time.month == month and 'duration' in metadata:
                        target_videos[video_id] = {'metadata': metadata, 'channel_id': channel_id}
                except (TypeError, ValueError):
                    continue
    return target_videos

def process_video_chat_log(video_id: str, duration_seconds: float, channel_name: str, channel_id: str, video_title: str, video_end_time: datetime.datetime):
    """
    The definitive, corrected function. It combines all successful logic:
    1.  Calculates the true video start time from metadata.
    2.  Uses a robust integer-based bucketing method (not resample).
    3.  Filters for the "safe zone" to avoid intros/outros.
    4.  Applies the requested 10-second lead-up to the final timestamp.
    """
    # --- Setup and Pre-checks ---
    duration_minutes = duration_seconds / 60.0
    if duration_minutes < 10:
        return []
    
    num_highlights = min(5, math.floor(duration_minutes / 30))
    if num_highlights == 0 and duration_minutes >= 10:
        num_highlights = 1

    log_file = CHAT_LOG_DIR / f"{video_id}.jsonl.gz"
    if not log_file.exists(): return []

    try:
        with gzip.open(log_file, 'rt', encoding='utf-8') as f:
            chat_data = [json.loads(line) for line in f.read().strip().split('\n') if line]
    except (json.JSONDecodeError, EOFError):
        print(f"  -> Corrupt chat log for {video_id}. Skipping.")
        return []
        
    if not chat_data: return []

    df = pd.DataFrame(chat_data)

    # --- Definitive Timestamp and Bucketing Logic ---

    # 1. Convert the microsecond UTC timestamp into a proper datetime index.
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='us', utc=True)
    df = df.set_index('datetime').sort_index()

    # 2. Calculate the video's true start time from its metadata (our reliable anchor).
    video_duration_timedelta = datetime.timedelta(seconds=duration_seconds)
    true_video_start_time = video_end_time - video_duration_timedelta

    # 3. Calculate the TRUE elapsed seconds for each message using the anchor.
    df['elapsed_seconds'] = (df.index - true_video_start_time).total_seconds()
    
    # 4. Filter to the "safe zone" using the correct elapsed_seconds.
    BUFFER_PERCENTAGE = 0.05
    start_buffer_seconds = duration_seconds * BUFFER_PERCENTAGE
    end_buffer_seconds = duration_seconds * (1 - BUFFER_PERCENTAGE)
    df_safezone = df[(df['elapsed_seconds'] >= start_buffer_seconds) & (df['elapsed_seconds'] <= end_buffer_seconds)].copy()

    if df_safezone.empty:
        print(f"  -> No chat data available within the safe zone for {video_id}.")
        return []

    # 5. Use the proven integer-based bucketing method (replaces resample).
    df_safezone['bucket'] = (df_safezone['elapsed_seconds'] // 15).astype(int)
    activity = df_safezone.groupby('bucket').size()
    
    top_bursts = activity.nlargest(num_highlights)
    top_bursts = top_bursts[top_bursts > 9]
    if top_bursts.empty: return []
    
    # --- AI Processing and Final Timestamp Calculation ---
    highlights_to_insert = []
    for bucket_index, _ in top_bursts.items():
        # Calculate the absolute time of the spike's start from the bucket index.
        spike_start_seconds = bucket_index * 15
        spike_datetime = true_video_start_time + datetime.timedelta(seconds=spike_start_seconds)

        # Get context for the AI from the full dataframe.
        window_df = df[(df.index >= spike_datetime) & (df.index < spike_datetime + pd.Timedelta(seconds=15))]
        full_context = " ".join(window_df['message'].tolist())
        
        # (Substance check could go here if needed)

        # Generate AI results for this specific highlight
        ai_result = generate_ai_results_batch([{"text": full_context, "channel_name": channel_name, "video_title": video_title}])
        if not ai_result or not ai_result[0]:
            continue

        # 6. Apply the requested 10-second lead-up time.
        adjusted_spike_datetime = spike_datetime - datetime.timedelta(seconds=10)
        
        # 7. Convert to a Unix timestamp to be saved.
        final_unix_timestamp = int(adjusted_spike_datetime.timestamp())

        ai_data = ai_result[0]
        if ai_data.get("topic") and ai_data.get("summary"):
            highlights_to_insert.append((
                video_id,
                final_unix_timestamp,
                ai_data["topic"],
                ai_data["summary"],
                ai_data["embedding"],
                channel_id
            ))
            
    return highlights_to_insert

def populate_video_highlights(year: int, month: int):
    """Main function with pause and resume capability."""
    try:
        print("Creating channel name lookup map...")
        conn = psycopg2.connect(**DB_CONFIG)
        with conn.cursor() as cur:
            cur.execute("SELECT channel_id, channel_name FROM channels")
            channel_map = {row[0]: row[1] for row in cur.fetchall()}
        conn.close()
        print(f"--> Channel map created with {len(channel_map)} entries.")
    except Exception as e:
        print(f"FATAL: Could not connect to database to create channel map: {e}")
        return

    videos_to_process = {}
    if STATE_FILE.exists():
        print("--> Found a resume file. Loading previous session...")
        with open(STATE_FILE, 'r') as f:
            videos_to_process = json.load(f)
        if not videos_to_process:
            print("    Resume file is empty. Starting a new session.")
            os.remove(STATE_FILE)
            videos_to_process = get_videos_for_month(year, month)
        else:
            print(f"--> Resuming with {len(videos_to_process)} videos remaining.")
    else:
        print("--> Starting a new session.")
        videos_to_process = get_videos_for_month(year, month)

    if not videos_to_process:
        print("No videos found to process.")
        return

    videos_list = list(videos_to_process.items())
    total_videos = len(videos_list)
    total_highlights_generated = 0

    try:
        for i, (video_id, data) in enumerate(videos_list):
            print(f"\n--- Processing video {i+1}/{total_videos}: {video_id} ---")
            
            try:
                conn_check = psycopg2.connect(**DB_CONFIG)
                with conn_check.cursor() as cur:
                    cur.execute("SELECT 1 FROM video_highlights WHERE video_id = %s LIMIT 1", (video_id,))
                    if cur.fetchone():
                        print("-> SKIPPING: Highlights for this video already exist in the database.")
                        videos_to_process.pop(video_id)
                        continue
                conn_check.close()
            except Exception as e:
                print(f"-> WARNING: Could not check for existing video highlights: {e}")

            channel_id = data['channel_id']
            channel_name = channel_map.get(channel_id, "Unknown Streamer")
            video_title = data['metadata'].get('title', "Untitled Video")
            video_duration = data['metadata']['duration']
            video_end_time = datetime.datetime.fromisoformat(data['metadata']['end_time'])

            # Pass the new parameters to the processing function
            video_highlights = process_video_chat_log(video_id, video_duration, channel_name, channel_id, video_title, video_end_time)
            if video_highlights:
                total_highlights_generated += len(video_highlights)
                print(f"-> ✅ Generated {len(video_highlights)} highlights. Inserting now...")
                
                for highlight in video_highlights:
                    seconds, topic, summary = highlight[1], highlight[2], highlight[3]
                    timestamp_str = f"{seconds // 60:02d}:{seconds % 60:02d}"
                    print(f"  - [{timestamp_str}] Topic: '{topic}' | Summary: {summary}")

                try:
                    conn_insert = psycopg2.connect(**DB_CONFIG)
                    with conn_insert.cursor() as cur:
                        psycopg2.extras.execute_values(
                            cur,
                            """
                            INSERT INTO video_highlights (
                                video_id, start_seconds, topic_tag, generated_summary, summary_embedding, channel_id
                            ) VALUES %s
                            ON CONFLICT (video_id, start_seconds) DO NOTHING;
                            """,
                            video_highlights
                        )
                        conn_insert.commit()
                    conn_insert.close()
                except Exception as e:
                    print(f"-> ❌ DATABASE ERROR: Failed to insert highlights for {video_id}: {e}")
            
            videos_to_process.pop(video_id)

    except KeyboardInterrupt:
        print("\n\n! PAUSE request received. Saving remaining videos to resume later...")
        with open(STATE_FILE, 'w') as f:
            json.dump(videos_to_process, f, indent=2)
        print(f"--> Successfully saved {len(videos_to_process)} remaining videos to {STATE_FILE.name}.")
        print("--> To resume, just run the script again.")
        sys.exit(0)

    if STATE_FILE.exists():
        print("--> Processing complete. Removing resume file.")
        os.remove(STATE_FILE)

    print(f"\n--- Processing Complete ---")
    print(f"Processed {total_videos} videos and generated a total of {total_highlights_generated} new highlights.")
    print("---------------------------\n")


if __name__ == '__main__':
    target_year = 2025
    target_month = 9
    populate_video_highlights(target_year, target_month)