from pyyoutube import Api
from datetime import datetime, timezone
from multiprocessing import get_context
from chat_downloader import sites
import isodate
import urllib3
import requests
import time
import pyyoutube
from config.settings import get_config
from utils.logging_utils import get_logger
from utils.helpers import get_ignore_list, is_video_past
from db.queries import is_metadata_and_chat_log_processed
from cacheutil.cache_manager import write_metadata_to_cache, load_channels
from db.queries import insert_video_metadata
from db.connection import init_db_pool

def get_metadata_for_channel(channel_name, channel_id, year, month, download_queue):
    """
    Fetches video metadata from YouTube for a given channel and date range.
    """
    # 1. Initialization
    yt_api = Api(api_key=get_config("API", "YOUTUBE_API_KEY"))
    chat_downloader = sites.YouTubeChatDownloader()
    ignore_list = get_ignore_list()
    logger = get_logger()
    
    max_retries = int(get_config("Settings", "MaxRetries"))
    start_month = datetime(year, month, 1, tzinfo=timezone.utc)
    end_month = start_month.replace(month=month % 12 + 1, year=year + month // 12)

    playlist_id = "UU" + channel_id[2:]
    page_token = None
    stop_pagination = False

    while not stop_pagination:
        playlist_items = None
        
        # 2. Fetch Playlist Page with Retry Logic
        for attempt in range(max_retries):
            try:
                playlist_items = yt_api.get_playlist_items(
                    playlist_id=playlist_id, 
                    page_token=page_token, 
                    count=50
                )
                break # Success
            except (requests.exceptions.ConnectionError, urllib3.exceptions.ProtocolError, pyyoutube.error.PyYouTubeException) as e:
                if "quota" in str(e).lower():
                    logger.error(f"❌ Quota exceeded for {channel_name}. Stopping.")
                    return
                
                wait = 2 ** attempt
                logger.warning(f"Connection error fetching page for {channel_name} (Attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
        
        if not playlist_items or not playlist_items.items:
            break

        # 3. Process Individual Videos
        for item in playlist_items.items:
            video_id = item.contentDetails.videoId
            
            if video_id in ignore_list:
                continue

            # Fetch video data (ChatDownloader call)
            video_data = None
            for attempt in range(max_retries):
                try:
                    video_data = chat_downloader.get_video_data(video_id=video_id)
                    break
                except Exception as e:
                    logger.warning(f"Failed to get video data for {video_id}: {e}. Retry {attempt+1}...")
                    time.sleep(1)

            if not video_data:
                continue

            # Date Logic
            end_date = video_data["end_time"]
            if not end_date:
                end_date = datetime.fromisoformat(item.contentDetails.videoPublishedAt.replace('Z', '+00:00')).replace(tzinfo=timezone.utc)
            else:
                end_date = datetime.fromtimestamp(end_date / 1_000_000, timezone.utc)

            # Check if we should stop paginating (went past our target month)
            if end_date < start_month:
                stop_pagination = True
                break 

            # Check if video is within target range
            if start_month <= end_date < end_month:
                if not is_video_past(video_id):
                    logger.info(f"Skipping {video_id}; not a concluded stream.")
                    continue

                duration = video_data["duration"]
                if not duration:
                    # Fallback to YT API for duration if needed
                    try:
                        v_details = yt_api.get_video_by_id(video_id=video_id)
                        duration = isodate.parse_duration(v_details.items[0].contentDetails.duration).total_seconds()
                    except:
                        duration = 0

                # Check DB status
                is_chat_processed, is_meta_processed = is_metadata_and_chat_log_processed(video_id)
                if is_meta_processed and is_chat_processed:
                    continue

                # Add to Shared Queue
                if (channel_id, video_id) not in download_queue and video_data["continuation_info"]:
                    download_queue.append((channel_id, video_id))
                    logger.info(f"Added {video_id} to queue for {channel_name}")

                # Cache & DB persistence
                write_metadata_to_cache(
                    channel_id=channel_id, video_id=video_id, title=item.snippet.title,
                    end_time=end_date.isoformat(), duration=duration
                )
                insert_video_metadata(
                    channel_id=channel_id, video_id=video_id, title=item.snippet.title,
                    end_time=end_date.isoformat(), duration=duration
                )

        # 4. Advance Pagination
        page_token = playlist_items.nextPageToken
        if not page_token:
            stop_pagination = True
        
        # Rate limit once per PAGE (50 videos)
        time.sleep(1)
        
def get_metadata_for_date_range(year, month, download_queue):
    """
    Fetch and process video metadata for all channels within a specified date range.

    This function loads all channels from the configuration and processes each channel
    in parallel using multiple threads to retrieve video metadata from YouTube for the
    specified year and month. For each channel, it fetches metadata for all videos within
    that date range and updates the download queue and database accordingly.

    Args:
        year (int): The year for which to fetch video metadata.
        month (int): The month for which to fetch video metadata.

    Returns:
        None
    """

    channels = load_channels()
    channel_items = list(channels.items())
    num_threads = int(get_config("Settings", "NumThreads"))

    with get_context("spawn").Pool(processes=num_threads, initializer=init_db_pool) as pool:
        pool.starmap(get_metadata_for_channel, [(name, channel_id, year, month, download_queue) for name, channel_id in channel_items])