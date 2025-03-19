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
from db.queries import is_chat_log_processed, is_metadata_processed
from cacheutil.cache_manager import write_metadata_to_cache, load_channels
from db.queries import insert_video_metadata
from db.connection import init_db_pool

# Get metadata for videos within date range from YouTube and add to download queue
def get_metadata_for_channel(channel_name, channel_id, year, month, download_queue):
    """
    Fetches video metadata from YouTube for a given channel and date range and adds it to the download queue.

    This function fetches video metadata from YouTube for a given channel and date range, and adds it to the download queue if it has not been processed previously.

    Args:
        channel_name (str): The name of the channel.
        channel_id (str): The ID of the channel.
        year (int): The year to fetch metadata for.
        month (int): The month to fetch metadata for.

    Returns:
        None
    """

    yt_api = Api(api_key=get_config("API", "YOUTUBE_API_KEY"))
    # Compute start and end of month
    logger = get_logger()
    start_month = datetime(year, month, 1, tzinfo=timezone.utc)
    end_month = start_month.replace(month=month % 12 + 1, year=year + month // 12)

    retry_count = 0
    retry_delay = 5

    playlist_id = "UU" + channel_id[2:]
    stop_pagination = False
    page_token = None

    try:
        logger.info(f"Getting metadata for {channel_name} ({channel_id})")
        while not stop_pagination:
            playlist_items = yt_api.get_playlist_items(playlist_id=playlist_id, page_token=page_token, count=50)

            if not playlist_items or not playlist_items.items:
                logger.warning(f"Failed to get playlist items for {channel_name} ({channel_id})")
                break

            if not playlist_items.items:
                break

            for item in playlist_items.items:
                video_id = item.contentDetails.videoId
                video_data = sites.YouTubeChatDownloader().get_video_data(video_id=video_id)

                # If video status is not past, skip it
                if video_data["status"] != "past":
                    logger.info(f"Skipping {video_id} as it is not in the past. Status: {video_data['status']}")
                    continue

                end_date = video_data["end_time"]
                duration = video_data["duration"] 
                if not duration:
                    duration = isodate.parse_duration(yt_api.get_video_by_id(video_id=video_id).items[0].contentDetails.duration).total_seconds()

                # Prefer end date from chat log if available, otherwise get it from YT API
                if not end_date:
                    end_date = datetime.fromisoformat(item.contentDetails.videoPublishedAt.replace('Z', '+00:00')).replace(tzinfo=timezone.utc)
                else:
                    end_date = datetime.fromtimestamp(end_date / 1_000_000, timezone.utc)

                # Stop pagination if video is too old
                if end_date < start_month:
                    stop_pagination = True
                    break

                # If video is in database, skip it
                if is_chat_log_processed(video_id) and is_metadata_processed(video_id):
                    logger.info(f"Skipping {video_id} as it is already processed.")
                    continue

                # Add video to download queue if it's within date range and has chat log
                if start_month <= end_date < end_month:
                    if (channel_id, video_id) not in download_queue and video_data["continuation_info"]:
                        download_queue.append((channel_id, video_id))
                        logger.info(f"Added {video_id} to download queue for {channel_name} ({channel_id})")

                # Write metadata to cache
                write_metadata_to_cache(
                    channel_id=channel_id,
                    video_id=video_id,
                    title=item.snippet.title,
                    end_time=end_date.isoformat(),
                    duration=duration
                )

                # Write metadata to database
                insert_video_metadata(
                    channel_id=channel_id,
                    video_id=video_id,
                    title=item.snippet.title,
                    end_time=end_date.isoformat(),
                    duration=duration
                )

            page_token = playlist_items.nextPageToken
            if not page_token:
                stop_pagination = True

    except (urllib3.exceptions.SSLError, requests.exceptions.SSLError) as e:
        retry_count += 1
        if retry_count < int(get_config("Settings", "MaxRetries")):
            wait_time = min(retry_delay * (2 ** retry_count), 60)
            print(e)
            logger.info(f"⚠️ SSL Error encountered, retrying in {wait_time} seconds... ({retry_count}/{get_config('Settings', 'MaxRetries')})")
            time.sleep(wait_time)  
        else:
            logger.error(f"❌ Max retries exceeded. Error: {e}")
            return
    except pyyoutube.error.PyYouTubeException as e:
        if "quota" in str(e).lower():
            logger.error(f"❌ Quota exceeded. Error: {e}")
            return
        else:
            retry_count += 1
            if retry_count < int(get_config("Settings", "MaxRetries")):
                logger.info(f"⚠️ YouTube API error encountered, retrying in {retry_delay} seconds... ({retry_count}/{get_config('Settings', 'MaxRetries')})")
                wait_time = min(retry_delay * (2 ** retry_count), 60)
                time.sleep(wait_time)  
            else:
                logger.error(f"❌ Max retries exceeded. Error: {e}")
                return
        
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