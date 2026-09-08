import hashlib
import os
import time
import re
import socket
import sqlite3
import logging
import redis
import pytz
import json
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from flask import g, request, session, jsonify
from functools import wraps
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from googleapiclient.discovery import build
from sqlalchemy import func, extract
# When running inside Lambda the package is "web.models"; when running
# standalone via server.py it's just "models".  Try both.
try:
    from web.models import db, Channel, Video, User
except ImportError:
    from models import db, Channel, Video, User
load_dotenv()
def _redis_config():
    """Return (host, port) for Redis / ElastiCache.
    Priority:
      1. ELASTICACHE_HOST / ELASTICACHE_PORT  (set by deploy.py for Lambda)
      2. REDIS_HOST / REDIS_PORT              (legacy .env for local dev)
      3. localhost:6379                        (fallback)
    """
    host = (
        os.getenv("ELASTICACHE_HOST")
        or os.getenv("REDIS_HOST")
        or "localhost"
    )
    port = int(
        os.getenv("ELASTICACHE_PORT")
        or os.getenv("REDIS_PORT")
        or 6379
    )
    return host, port
REDIS_HOST, REDIS_PORT = _redis_config()
REDIS_CONFIG = {
    "host": REDIS_HOST,
    "port": REDIS_PORT,
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
SQLALCHEMY_ENGINE_OPTIONS = {
    "connect_args": {
        # shortest link in the chain: nginx 120s > gunicorn 120s > pg 20s
        "options": "-c default_transaction_read_only=on -c statement_timeout=20000",
        "application_name": "chat-ingest-web",
    },
    "pool_pre_ping": True,
    "pool_recycle": 300,
}
_hostname_cache = {}
print("Loading sentence transformer model for vector search...")
EMBEDDER = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("Model loaded successfully.")
def setup_logging(app):
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    app.logger.handlers.clear()
    app.logger.addHandler(handler)
    app.logger.setLevel(logging.INFO)
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
def resolve_hostname(ip):
    if ip in _hostname_cache:
        return _hostname_cache[ip]
    try:
        hostname = socket.gethostbyaddr(ip)[0]
    except Exception:
        hostname = ip
    _hostname_cache[ip] = hostname
    return hostname
def get_database_uri():
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    host = os.getenv("POSTGRES_HOST")
    port = os.getenv("POSTGRES_PORT")
    dbname = os.getenv("POSTGRES_DB")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}?client_encoding=utf8"
def get_sqlite_connection():
    """Creates a new SQLite connection stored in Flask's g object.
    In Lambda the filesystem is /tmp; we just skip if it fails."""
    if not hasattr(g, 'sqlite_conn'):
        try:
            db_path = os.getenv("SQLITE_PATH", "usage.db")
            g.sqlite_conn = sqlite3.connect(db_path)
            g.sqlite_conn.execute("""
                CREATE TABLE IF NOT EXISTS usage (
                    ip TEXT, date TEXT, count INTEGER,
                    PRIMARY KEY (ip, date)
                )
            """)
        except Exception:
            g.sqlite_conn = None
    return g.sqlite_conn
def get_redis_connection():
    """Get or create Redis connection stored in Flask's g object.
    Works with both standalone Redis and ElastiCache."""
    if not hasattr(g, 'redis_conn'):
        g.redis_conn = redis.StrictRedis(
            host=REDIS_CONFIG["host"],
            port=REDIS_CONFIG["port"],
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
        )
    return g.redis_conn
def get_locale():
    return session.get('language', 'en')
def check_rate_limit(ip, daily_limit):
    today = datetime.utcnow().strftime('%Y-%m-%d')
    conn = get_sqlite_connection()
    if conn is None:
        return int(daily_limit)
    try:
        cursor = conn.cursor()
        cursor.execute("INSERT OR IGNORE INTO usage (ip, date, count) VALUES (?, ?, 0)", (ip, today))
        cursor.execute("SELECT count FROM usage WHERE ip = ? AND date = ?", (ip, today))
        count = cursor.fetchone()[0]
        conn.commit()
        return max(0, int(daily_limit) - int(count))
    except Exception as e:
        logging.error(f"Rate limit check failed: {str(e)}")
        return int(daily_limit)
def get_current_month():
    return datetime.utcnow().strftime('%Y-%m')
def get_previous_two_months():
    today = datetime.today().replace(day=1)
    prev_month = today - timedelta(days=1)
    prev2_month = (prev_month.replace(day=1)) - timedelta(days=1)
    return [prev2_month.strftime('%Y-%m'), prev_month.strftime('%Y-%m')]
def inc_cache_hit_count():
    try:
        redis_conn = g.redis_conn
        today = datetime.now(pytz.utc).strftime("%Y-%m-%d")
        redis_conn.incr(f"cache_hits:{today}")
    except Exception:
        pass
def inc_cache_miss_count():
    try:
        redis_conn = g.redis_conn
        today = datetime.now(pytz.utc).strftime("%Y-%m-%d")
        redis_conn.incr(f"cache_misses:{today}")
    except Exception:
        pass
METRICS_NAMESPACE = "v2"
_INTERNAL_PATH_PREFIXES = (
    "/api/", "/static/", "/socket.io/", "/admin/", "/set_language/", "/_",
)
_INTERNAL_PATHS = {"/health", "/favicon.ico", "/set_language"}
def is_public_page(path):
    """Only browser-visible SPA routes belong in site analytics."""
    if not isinstance(path, str):
        return False
    path = path.split("?", 1)[0].split("#", 1)[0].strip()
    if not path.startswith("/") or len(path) > 200:
        return False
    if path in _INTERNAL_PATHS or path.startswith(_INTERNAL_PATH_PREFIXES):
        return False
    return "." not in path.rsplit("/", 1)[-1]
def record_page_view(page):
    """Record one validated public page view and its visitor dimensions."""
    if not is_public_page(page):
        return False
    page = page.split("?", 1)[0].split("#", 1)[0].strip()
    country = request.headers.get("CF-IPCountry", "Unknown")
    source_ip = (request.headers.get("CF-Connecting-IP")
                 or request.headers.get("X-Real-IP")
                 or request.remote_addr
                 or "unknown")
    today = datetime.now(pytz.utc).strftime("%Y-%m-%d")
    visitor_ip = hashlib.sha256(source_ip.encode()).hexdigest()
    try:
        redis_conn = get_redis_connection()
        expiry_time = 2592000
        prefix = METRICS_NAMESPACE
        pipe = redis_conn.pipeline()
        pipe.sadd(f"{prefix}:unique_visitors_country:{country}:{today}", visitor_ip)
        pipe.expire(f"{prefix}:unique_visitors_country:{country}:{today}", expiry_time)
        pipe.sadd(f"{prefix}:unique_visitors:{today}", visitor_ip)
        pipe.expire(f"{prefix}:unique_visitors:{today}", expiry_time)
        pipe.hincrby(f"{prefix}:page_views:{today}", page, 1)
        pipe.expire(f"{prefix}:page_views:{today}", expiry_time)
        pipe.execute()
        return True
    except Exception:
        logging.exception("Unable to record page-view metric")
        return False
def track_metrics(response):
    if response.status_code != 200:
        return response
    record_page_view(request.path)
    return response
def get_metrics():
    redis_conn = redis.StrictRedis(
        host=REDIS_CONFIG["host"],
        port=REDIS_CONFIG["port"],
        decode_responses=True,
    )
    today = datetime.now(pytz.utc)
    dates = [(today - relativedelta(days=i)).strftime("%Y-%m-%d") for i in range(30)]
    metrics = {}
    page_totals = {}
    for d in dates:
        for page, count in redis_conn.hgetall(
                f"{METRICS_NAMESPACE}:page_views:{d}").items():
            if is_public_page(page):
                page_totals[page] = page_totals.get(page, 0) + int(count)
    metrics["page_views"] = page_totals
    country_counts = {}
    for d in dates:
        pattern = f"{METRICS_NAMESPACE}:unique_visitors_country:*:{d}"
        for key in redis_conn.scan_iter(match=pattern, count=100):
            cc = key.split(":")[2]
            country_counts[cc] = country_counts.get(cc, 0) + redis_conn.scard(key)
    metrics["country_visits"] = dict(sorted(country_counts.items(), key=lambda x: x[1], reverse=True))
    metrics["unique_visitors"] = {
        d: redis_conn.scard(f"{METRICS_NAMESPACE}:unique_visitors:{d}")
        for d in dates
    }
    metrics["cache_data"] = {
        d: {
            "cache_hits": int(redis_conn.get(f"cache_hits:{d}") or 0),
            "cache_misses": int(redis_conn.get(f"cache_misses:{d}") or 0),
        }
        for d in dates
    }
    return metrics
def timeout(seconds=5):
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
def streaming_hours_query(agg_func):
    group = request.args.get('group', None)
    month = request.args.get('month', datetime.utcnow().strftime('%Y-%m'))
    month_start = f"{month}-01"
    month_col = func.date_trunc('month', func.timezone('UTC', Video.end_time)).label('month')
    hours_col = (agg_func(extract('epoch', Video.duration)) / 3600).label('hours')
    query = (
        db.session.query(Channel.channel_name, month_col, hours_col)
        .join(Channel, Video.channel_id == Channel.channel_id)
        .filter(func.date_trunc('month', func.timezone('UTC', Video.end_time)) == month_start)
    )
    if group and group != "All":
        query = query.filter(Channel.channel_group == group)
    query = query.group_by(Channel.channel_name, month_col).order_by(hours_col.desc())
    return query
def parse_search_query(raw_query: str):
    filters = {"channel_name": None, "from_date": None, "to_date": None}
    channel_pattern = r'channel:"([^"]+)"|channel:(\S+)'
    from_pattern = r'from:(\d{4}-\d{2}-\d{2})'
    to_pattern = r'to:(\d{4}-\d{2}-\d{2})'
    channel_match = re.search(channel_pattern, raw_query)
    if channel_match:
        filters["channel_name"] = channel_match.group(1) or channel_match.group(2)
        raw_query = raw_query[:channel_match.start()] + raw_query[channel_match.end():]
    from_match = re.search(from_pattern, raw_query)
    if from_match:
        try:
            datetime.strptime(from_match.group(1), '%Y-%m-%d')
            filters["from_date"] = from_match.group(1)
            raw_query = raw_query[:from_match.start()] + raw_query[from_match.end():]
        except ValueError:
            return None, None, f"Invalid 'from' date format: {from_match.group(1)}. Use YYYY-MM-DD."
    to_match = re.search(to_pattern, raw_query)
    if to_match:
        try:
            datetime.strptime(to_match.group(1), '%Y-%m-%d')
            filters["to_date"] = to_match.group(1)
            raw_query = raw_query[:to_match.start()] + raw_query[to_match.end():]
        except ValueError:
            return None, None, f"Invalid 'to' date format: {to_match.group(1)}. Use YYYY-MM-DD."
    return raw_query.strip(), filters, None
def validate_month_format(month_str):
    try:
        return datetime.strptime(month_str, "%Y-%m")
    except ValueError:
        return None
def format_month_for_sql(month_str):
    return f"{month_str}-01"
def load_channel_mapping():
    channel_file = os.path.join(os.path.dirname(__file__), 'channel.json')
    with open(channel_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    mapping = {}
    for organization, channels in data.items():
        for name, channel_id in channels.items():
            mapping[name.lower()] = {'id': channel_id, 'name': name, 'organization': organization}
    return mapping
def get_youtube_service(current_app):
    api_key = current_app.config.get('YOUTUBE_API_KEY')
    if not api_key:
        raise ValueError("YouTube API key not configured")
    return build('youtube', 'v3', developerKey=api_key)
def determine_content_type(video_info):
    live_details = video_info.get('liveStreamingDetails', {})
    snippet = video_info.get('snippet', {})
    if live_details:
        if live_details.get('concurrentViewers') or live_details.get('actualStartTime'):
            return 'stream'
        if live_details.get('scheduledStartTime'):
            return 'stream'
    return 'video'
def fetch_live_and_upcoming_streams(youtube, channel_id, event_type, limit):
    streams = []
    search_response = youtube.search().list(
        part='snippet', channelId=channel_id, type='video',
        eventType=event_type, maxResults=limit, order='date'
    ).execute()
    items = search_response.get('items', [])
    if not items:
        return streams
    video_ids = [item['id']['videoId'] for item in items]
    videos_response = youtube.videos().list(
        part='liveStreamingDetails,snippet,contentDetails,statistics',
        id=','.join(video_ids)
    ).execute()
    video_details = {v['id']: v for v in videos_response.get('items', [])}
    for item in items:
        video_id = item['id']['videoId']
        snippet = item['snippet']
        video_info = video_details.get(video_id, {})
        live_details = video_info.get('liveStreamingDetails', {})
        content_type = determine_content_type(video_info)
        stream_data = {
            'title': snippet['title'], 'video_id': video_id,
            'url': f'https://www.youtube.com/watch?v={video_id}',
            'thumbnail': snippet.get('thumbnails', {}).get('medium', {}).get('url', ''),
            'status': event_type, 'content_type': content_type,
            'published_at': snippet.get('publishedAt', '')
        }
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
    videos = []
    channel_response = youtube.channels().list(part='contentDetails', id=channel_id).execute()
    if not channel_response.get('items'):
        return videos
    uploads_playlist_id = (
        channel_response['items'][0].get('contentDetails', {})
        .get('relatedPlaylists', {}).get('uploads')
    )
    if not uploads_playlist_id:
        return videos
    playlist_response = youtube.playlistItems().list(
        part='snippet,contentDetails', playlistId=uploads_playlist_id, maxResults=limit
    ).execute()
    items = playlist_response.get('items', [])
    if not items:
        return videos
    video_ids = [item['contentDetails']['videoId'] for item in items]
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
        content_type = 'stream' if live_details else 'video'
        video_data = {
            'title': snippet.get('title', ''), 'video_id': video_id,
            'url': f'https://www.youtube.com/watch?v={video_id}',
            'thumbnail': snippet.get('thumbnails', {}).get('medium', {}).get('url', ''),
            'status': 'completed', 'content_type': content_type,
            'published_at': snippet.get('publishedAt', ''),
            'duration': content_details.get('duration', '')
        }
        view_count = statistics.get('viewCount')
        if view_count:
            video_data['view_count'] = int(view_count)
        like_count = statistics.get('likeCount')
        if like_count:
            video_data['like_count'] = int(like_count)
        if live_details:
            for k, v in [('scheduledStartTime', 'scheduled_start'),
                         ('actualStartTime', 'started_at'),
                         ('actualEndTime', 'ended_at')]:
                if live_details.get(k):
                    video_data[v] = live_details[k]
        videos.append(video_data)
    return videos
def get_or_compute_cached(redis_key, compute_fn):
    try:
        cached_data = g.redis_conn.get(redis_key)
    except Exception:
        cached_data = None
    if cached_data:
        inc_cache_hit_count()
        try:
            # Convert entries created by older releases with a TTL to the new
            # durable analytics-cache policy.
            g.redis_conn.persist(redis_key)
        except Exception:
            pass
        return jsonify(json.loads(cached_data))
    inc_cache_miss_count()
    result = compute_fn()
    if isinstance(result, tuple):
        return result
    try:
        # Analytics caches are durable. Finalized-month keys are immutable;
        # rolling/aggregate keys are explicitly invalidated after publication.
        g.redis_conn.set(redis_key, json.dumps(result))
    except Exception:
        pass
    return jsonify(result)
def cached_json(key_func):
    def decorator(view_func):
        @wraps(view_func)
        def wrapper(*args, **kwargs):
            return get_or_compute_cached(key_func(), lambda: view_func(*args, **kwargs))
        return wrapper
    return decorator
def parse_month(month_str):
    return datetime.strptime(month_str, "%Y-%m")
def month_range(month_str, months=1):
    start = parse_month(month_str)
    end = start + relativedelta(months=months)
    return start, end
def resolve_user_id(identifier):
    if not identifier.startswith('@'):
        return identifier
    user = db.session.query(User).filter(User.username == identifier).first()
    return user.user_id if user else None
