import hashlib
import os
import time
import re
import socket
import sqlite3
import logging
import redis
import psycopg2
import pytz
import json
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from flask import g, request, session
from functools import wraps
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from googleapiclient.discovery import build


load_dotenv()

# Configuration
DB_CONFIG = {
    "dbname": os.getenv("POSTGRES_DB"),
    "user": os.getenv("POSTGRES_USER"),
    "password": os.getenv("POSTGRES_PASSWORD"),
    "host": os.getenv("POSTGRES_HOST"),
    "port": os.getenv("POSTGRES_PORT"),
    "client_encoding": "UTF8"
}

REDIS_CONFIG = {
    "host": os.getenv("REDIS_HOST"),
    "port": os.getenv("REDIS_PORT"),
}

LANGUAGES = {
    'en': 'English',
    'ja': '日本語',
    'ko': '한국어'
}

SUSPICIOUS_PATHS = [
    r"/wp-.*", r"/xmlrpc\.php", r"/admin", r"/phpmyadmin", r"/shell", r"/\.env", 
    r"/cgi-bin", r"/config", r"/etc/passwd", r"/api/.*?/debug", r"/\.git"
]

# Global cache
_hostname_cache = {}

# Load sentence transformer model
print("Loading sentence transformer model for vector search...")
EMBEDDER = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("Model loaded successfully.")


def setup_logging(app):
    """Setup logging configuration for the Flask app."""
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    app.logger.handlers.clear()
    app.logger.addHandler(handler)
    app.logger.setLevel(logging.INFO)
    logging.getLogger('werkzeug').setLevel(logging.WARNING)


def resolve_hostname(ip):
    """Resolve hostname from IP address with caching."""
    if ip in _hostname_cache:
        return _hostname_cache[ip]

    try:
        hostname = socket.gethostbyaddr(ip)[0]
    except Exception:
        hostname = ip

    _hostname_cache[ip] = hostname
    return hostname


def get_db_connection():
    """Get a PostgreSQL database connection."""
    return psycopg2.connect(**DB_CONFIG)


def get_sqlite_connection():
    """Creates a new SQLite connection stored in Flask's g object."""
    if not hasattr(g, 'sqlite_conn'):
        g.sqlite_conn = sqlite3.connect('usage.db')
        g.sqlite_conn.execute("""
            CREATE TABLE IF NOT EXISTS usage (
                ip TEXT,
                date TEXT, 
                count INTEGER,
                PRIMARY KEY (ip, date)
            )
        """)
    return g.sqlite_conn


def get_redis_connection():
    """Get or create Redis connection stored in Flask's g object."""
    if not hasattr(g, 'redis_conn'):
        g.redis_conn = redis.StrictRedis(
            host=REDIS_CONFIG["host"],
            port=REDIS_CONFIG["port"],
            decode_responses=True
        )
    return g.redis_conn


def get_locale():
    """Get the current locale from session."""
    return session.get('language', 'en')


def check_rate_limit(ip, daily_limit):
    """Check SQLite rate limit for IP, returns remaining requests."""
    today = datetime.utcnow().strftime('%Y-%m-%d')
    conn = get_sqlite_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR IGNORE INTO usage (ip, date, count) VALUES (?, ?, 0)
        """, (ip, today))
        cursor.execute("""
            SELECT count FROM usage WHERE ip = ? AND date = ?
        """, (ip, today))
        count = cursor.fetchone()[0]
        conn.commit()
        return max(0, int(daily_limit) - int(count))
    except Exception as e:
        logging.error(f"Rate limit check failed: {str(e)}")
        return int(daily_limit)


def get_current_month():
    """Return current month in YYYY-MM format."""
    return datetime.utcnow().strftime('%Y-%m')


def get_previous_two_months():
    """Get the previous two months in YYYY-MM format."""
    today = datetime.today().replace(day=1)
    prev_month = today - timedelta(days=1)
    prev2_month = (prev_month.replace(day=1)) - timedelta(days=1)
    return [prev2_month.strftime('%Y-%m'), prev_month.strftime('%Y-%m')]


def inc_cache_hit_count():
    """Increment cache hit count in Redis."""
    redis_conn = g.redis_conn
    today = datetime.now(pytz.utc).strftime("%Y-%m-%d")
    redis_conn.incr(f"cache_hits:{today}")


def inc_cache_miss_count():
    """Increment cache miss count in Redis."""
    redis_conn = g.redis_conn
    today = datetime.now(pytz.utc).strftime("%Y-%m-%d")
    redis_conn.incr(f"cache_misses:{today}")


def track_metrics(response):
    """Tracks unique visitors per country and aggregates page views over 30 days."""
    if response.status_code != 200:
        return response
    
    country = request.headers.get("CF-IPCountry", "Unknown")
    page = request.path
    
    if any(path in page for path in ["/api/", "/static/", "/favicon.ico", "/set_language/"]):
        return response

    today = datetime.now(pytz.utc).strftime("%Y-%m-%d")
    visitor_ip = hashlib.sha256(
        request.headers.get("CF-Connecting-IP", request.remote_addr).encode()
    ).hexdigest()

    redis_conn = get_redis_connection()
    
    # Track unique visitors per country
    redis_conn.sadd(f"unique_visitors_country:{country}:{today}", visitor_ip)
    redis_conn.sadd(f"unique_visitors:{today}", visitor_ip)

    # Aggregate page views across 30 days
    redis_conn.hincrby("page_views_30d", page, 1)

    # Set expiry for cleanup
    expiry_time = 2592000  # 30 days in seconds
    redis_conn.expire(f"unique_visitors_country:{country}:{today}", expiry_time)
    redis_conn.expire(f"unique_visitors:{today}", expiry_time)
    redis_conn.expire("page_views_30d", expiry_time)
    redis_conn.expire(f"cache_hits:{today}", expiry_time)
    redis_conn.expire(f"cache_misses:{today}", expiry_time)

    return response


def get_metrics():
    """Retrieve latest metrics data."""
    redis_conn = redis.StrictRedis(
        host=REDIS_CONFIG["host"],
        port=REDIS_CONFIG["port"],
        decode_responses=True
    )
    today = datetime.now(pytz.utc)
    metrics = {}

    metrics["page_views"] = redis_conn.hgetall("page_views_30d")

    # Aggregate country visits over 30 days
    country_counts = {}
    for i in range(30):
        date_str = (today - relativedelta(days=i)).strftime("%Y-%m-%d")
        for country in redis_conn.keys(f"unique_visitors_country:*:{date_str}"):
            country_code = country.split(":")[1]
            unique_visitors = redis_conn.scard(country)
            country_counts[country_code] = country_counts.get(country_code, 0) + unique_visitors

    metrics["country_visits"] = dict(sorted(country_counts.items(), key=lambda x: x[1], reverse=True))

    # Unique visitors per day
    metrics["unique_visitors"] = { 
        (today - relativedelta(days=i)).strftime("%Y-%m-%d"): redis_conn.scard(
            f"unique_visitors:{(today - relativedelta(days=i)).strftime('%Y-%m-%d')}"
        )
        for i in range(30)
    }

    metrics["cache_data"] = {
        (today - relativedelta(days=i)).strftime("%Y-%m-%d"): {
            "cache_hits": int(redis_conn.get(
                f"cache_hits:{(today - relativedelta(days=i)).strftime('%Y-%m-%d')}"
            ) or 0),
            "cache_misses": int(redis_conn.get(
                f"cache_misses:{(today - relativedelta(days=i)).strftime('%Y-%m-%d')}"
            ) or 0)
        } for i in range(30)
    }

    return metrics


def timeout(seconds=5):
    """Decorator to add timeout functionality."""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = f(*args, **kwargs)
            duration = time.time() - start
            if duration > seconds:
                raise TimeoutError("Request timed out")
            return result
        return wrapper
    return decorator


def streaming_hours_query(aggregation_function, group=None):
    """Build query for streaming hours with specified aggregation function."""
    group = request.args.get('group', None)
    month = request.args.get('month', datetime.utcnow().strftime('%Y-%m'))
    month_start = f"{month}-01"

    base_query = f"""
        SELECT
            c.channel_name,
            DATE_TRUNC('month', v.end_time AT TIME ZONE 'UTC') AS month,
            {aggregation_function}(EXTRACT(EPOCH FROM v.duration)) / 3600 AS hours
        FROM videos v
        JOIN channels c ON v.channel_id = c.channel_id
        WHERE DATE_TRUNC('month', v.end_time AT TIME ZONE 'UTC') = %s::DATE
    """

    params = [month_start]

    if group and group != "All":
        base_query += " AND c.channel_group = %s"
        params.append(group)

    base_query += " GROUP BY c.channel_name, month ORDER BY hours DESC"

    return base_query, params


def parse_search_query(raw_query: str):
    """
    Parses a raw search query to separate search operators from the main text.
    
    Returns:
        A tuple containing:
        - clean_query (str): The query text with operators removed.
        - filters (dict): A dictionary of extracted filter values.
        - error (str|None): An error message if validation fails.
    """
    filters = {
        "channel_name": None,
        "from_date": None,
        "to_date": None,
    }
    
    channel_pattern = r'channel:"([^"]+)"|channel:(\S+)'
    from_pattern = r'from:(\d{4}-\d{2}-\d{2})'
    to_pattern = r'to:(\d{4}-\d{2}-\d{2})'
    
    # Extract Channel
    channel_match = re.search(channel_pattern, raw_query)
    if channel_match:
        filters["channel_name"] = channel_match.group(1) or channel_match.group(2)
        raw_query = raw_query[:channel_match.start()] + raw_query[channel_match.end():]

    # Extract From Date
    from_match = re.search(from_pattern, raw_query)
    if from_match:
        try:
            datetime.strptime(from_match.group(1), '%Y-%m-%d')
            filters["from_date"] = from_match.group(1)
            raw_query = raw_query[:from_match.start()] + raw_query[from_match.end():]
        except ValueError:
            return None, None, f"Invalid 'from' date format: {from_match.group(1)}. Use YYYY-MM-DD."

    # Extract To Date
    to_match = re.search(to_pattern, raw_query)
    if to_match:
        try:
            datetime.strptime(to_match.group(1), '%Y-%m-%d')
            filters["to_date"] = to_match.group(1)
            raw_query = raw_query[:to_match.start()] + raw_query[to_match.end():]
        except ValueError:
            return None, None, f"Invalid 'to' date format: {to_match.group(1)}. Use YYYY-MM-DD."

    clean_query = raw_query.strip()
    return clean_query, filters, None


def validate_month_format(month_str):
    """Validate and parse month string in YYYY-MM format."""
    try:
        return datetime.strptime(month_str, "%Y-%m")
    except ValueError:
        return None


def format_month_for_sql(month_str):
    """Convert YYYY-MM to YYYY-MM-01 for SQL queries."""
    return f"{month_str}-01"

def load_channel_mapping():
    """
    Load and flatten channel name to ID mapping from channel.json.
    
    Returns:
        dict: Mapping of lowercase channel names to their info
    """
    channel_file = os.path.join(
        os.path.dirname(__file__), 
        'channel.json'
    )
    
    with open(channel_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Flatten the nested structure for easy lookup
    mapping = {}
    for organization, channels in data.items():
        for name, channel_id in channels.items():
            mapping[name.lower()] = {
                'id': channel_id,
                'name': name,
                'organization': organization
            }
    
    return mapping


def get_youtube_service(current_app):
    """
    Build and return YouTube API service client.
    
    Returns:
        googleapiclient.discovery.Resource: YouTube API service
    """
    api_key = current_app.config.get('YOUTUBE_API_KEY')
    if not api_key:
        raise ValueError("YouTube API key not configured")
    
    return build('youtube', 'v3', developerKey=api_key)


def determine_content_type(video_info):
    """
    Determine if a video is a stream, premiere, or regular upload.
    
    Args:
        video_info: Video details from YouTube API
        
    Returns:
        str: 'stream', 'premiere', or 'video'
    """
    live_details = video_info.get('liveStreamingDetails', {})
    snippet = video_info.get('snippet', {})
    
    # Check if it has live streaming details
    if live_details:
        # Check liveBroadcastContent for current status hints
        broadcast_content = snippet.get('liveBroadcastContent', 'none')
        
        # If it has concurrent viewers or actual start time, it was live content
        if live_details.get('concurrentViewers') or live_details.get('actualStartTime'):
            # Premieres typically have shorter durations and no concurrent viewers history
            # But this is hard to distinguish after the fact
            # Check if it was originally a premiere by looking at the duration
            # Premieres are usually pre-recorded so they have exact durations
            content_details = video_info.get('contentDetails', {})
            duration = content_details.get('duration', '')
            
            # If there's an actual end time close to scheduled + duration, likely premiere
            # For now, we'll mark anything with liveStreamingDetails as stream
            # unless we can find better indicators
            return 'stream'
        
        # Has scheduled time but no actual start = upcoming
        if live_details.get('scheduledStartTime'):
            return 'stream'
    
    return 'video'


def fetch_live_and_upcoming_streams(youtube, channel_id, event_type, limit):
    """
    Fetch live or upcoming streams/premieres for a channel.
    
    Args:
        youtube: YouTube API service
        channel_id: YouTube channel ID
        event_type: 'live' or 'upcoming'
        limit: Maximum results to return
        
    Returns:
        list: List of stream data dictionaries
    """
    streams = []
    
    # Search for live/upcoming content
    search_response = youtube.search().list(
        part='snippet',
        channelId=channel_id,
        type='video',
        eventType=event_type,
        maxResults=limit,
        order='date'
    ).execute()
    
    items = search_response.get('items', [])
    if not items:
        return streams
    
    # Get video IDs for batch details request
    video_ids = [item['id']['videoId'] for item in items]
    
    # Fetch detailed video information
    videos_response = youtube.videos().list(
        part='liveStreamingDetails,snippet,contentDetails,statistics',
        id=','.join(video_ids)
    ).execute()
    
    # Create a lookup for video details
    video_details = {
        v['id']: v for v in videos_response.get('items', [])
    }
    
    for item in items:
        video_id = item['id']['videoId']
        snippet = item['snippet']
        
        video_info = video_details.get(video_id, {})
        live_details = video_info.get('liveStreamingDetails', {})
        statistics = video_info.get('statistics', {})
        
        content_type = determine_content_type(video_info)
        
        stream_data = {
            'title': snippet['title'],
            'video_id': video_id,
            'url': f'https://www.youtube.com/watch?v={video_id}',
            'thumbnail': snippet.get('thumbnails', {}).get('medium', {}).get('url', ''),
            'status': event_type,
            'content_type': content_type,
            'published_at': snippet.get('publishedAt', '')
        }
        
        # Add timing information based on stream status
        if event_type == 'upcoming':
            scheduled_time = live_details.get('scheduledStartTime')
            if scheduled_time:
                stream_data['scheduled_start'] = scheduled_time
                
        elif event_type == 'live':
            actual_start = live_details.get('actualStartTime')
            concurrent_viewers = live_details.get('concurrentViewers')
            if actual_start:
                stream_data['started_at'] = actual_start
            if concurrent_viewers:
                stream_data['concurrent_viewers'] = int(concurrent_viewers)
        
        streams.append(stream_data)
    
    return streams


def fetch_past_videos(youtube, channel_id, limit):
    """
    Fetch past videos including completed streams, premieres, and uploads.
    
    Args:
        youtube: YouTube API service
        channel_id: YouTube channel ID
        limit: Maximum results to return
        
    Returns:
        list: List of video data dictionaries
    """
    videos = []
    
    # First, get the channel's uploads playlist ID
    channel_response = youtube.channels().list(
        part='contentDetails',
        id=channel_id
    ).execute()
    
    if not channel_response.get('items'):
        return videos
    
    uploads_playlist_id = (
        channel_response['items'][0]
        .get('contentDetails', {})
        .get('relatedPlaylists', {})
        .get('uploads')
    )
    
    if not uploads_playlist_id:
        return videos
    
    # Get videos from the uploads playlist
    playlist_response = youtube.playlistItems().list(
        part='snippet,contentDetails',
        playlistId=uploads_playlist_id,
        maxResults=limit
    ).execute()
    
    items = playlist_response.get('items', [])
    if not items:
        return videos
    
    # Get video IDs for detailed information
    video_ids = [
        item['contentDetails']['videoId'] 
        for item in items
    ]
    
    # Fetch detailed video information including statistics
    videos_response = youtube.videos().list(
        part='liveStreamingDetails,snippet,contentDetails,statistics',
        id=','.join(video_ids)
    ).execute()
    
    for video_info in videos_response.get('items', []):
        video_id = video_info['id']
        snippet = video_info.get('snippet', {})
        live_details = video_info.get('liveStreamingDetails', {})
        content_details = video_info.get('contentDetails', {})
        statistics = video_info.get('statistics', {})
        
        # Determine content type
        if live_details:
            # Has live streaming details - was a stream or premiere
            if live_details.get('actualEndTime'):
                content_type = 'stream'
            elif live_details.get('scheduledStartTime'):
                # Had a scheduled start, likely was a premiere or stream
                content_type = 'stream'
            else:
                content_type = 'stream'
        else:
            content_type = 'video'
        
        video_data = {
            'title': snippet.get('title', ''),
            'video_id': video_id,
            'url': f'https://www.youtube.com/watch?v={video_id}',
            'thumbnail': snippet.get('thumbnails', {}).get('medium', {}).get('url', ''),
            'status': 'completed',
            'content_type': content_type,
            'published_at': snippet.get('publishedAt', ''),
            'duration': content_details.get('duration', '')
        }
        
        # Add view count for completed videos
        view_count = statistics.get('viewCount')
        if view_count:
            video_data['view_count'] = int(view_count)
        
        # Add like count if available
        like_count = statistics.get('likeCount')
        if like_count:
            video_data['like_count'] = int(like_count)
        
        # Add timing information from live streaming details
        if live_details:
            actual_start = live_details.get('actualStartTime')
            actual_end = live_details.get('actualEndTime')
            scheduled_start = live_details.get('scheduledStartTime')
            
            if scheduled_start:
                video_data['scheduled_start'] = scheduled_start
            if actual_start:
                video_data['started_at'] = actual_start
            if actual_end:
                video_data['ended_at'] = actual_end
        
        videos.append(video_data)
    
    return videos
