import os
import re
import time
import json
from flask import Flask, request, session, g, Response
from flask_babel import Babel
from flask_socketio import SocketIO
from werkzeug.middleware.proxy_fix import ProxyFix
from dotenv import load_dotenv

from utils import (
    setup_logging, resolve_hostname, get_sqlite_connection,
    get_redis_connection, get_locale, track_metrics, get_metrics,
    record_page_view, is_public_page,
    SUSPICIOUS_PATHS, get_database_uri, SQLALCHEMY_ENGINE_OPTIONS
)
from api import api_bp
from routes import routes_bp
from models import db
from flask_cors import CORS
from flask_compress import Compress
import redis
import threading
from cache_warmer import run as run_cache_warmer


load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": ["http://localhost:5173", "https://holochatstats.info"]}})
# Threading + simple-websocket is compatible with current Python/Gunicorn and
# avoids Eventlet, which is no longer actively maintained. Engine.IO's default
# same-origin checks remain enabled.
socketio = SocketIO(app, async_mode='threading')

# Setup session key, babel and OpenRouter configuration
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY")
app.config["SESSION_TYPE"] = "filesystem"
app.config["BABEL_DEFAULT_LOCALE"] = "en"
app.config["BABEL_TRANSLATION_DIRECTORIES"] = "translations"
app.config["JSON_AS_ASCII"] = False
app.config["OPENROUTER_URL"] = os.getenv("OPENROUTER_URL")
app.config["OPENROUTER_MODEL"] = os.getenv("OPENROUTER_MODEL")
app.config["DAILY_LIMIT"] = os.getenv("LLM_DAILY_LIMIT")
app.config["YOUTUBE_API_KEY"] = os.getenv("YOUTUBE_API_KEY")
app.config["SQLALCHEMY_DATABASE_URI"] = get_database_uri()
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = SQLALCHEMY_ENGINE_OPTIONS
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)

# Setup logging
setup_logging(app)

# Register blueprints
app.register_blueprint(api_bp)
app.register_blueprint(routes_bp)

db.init_app(app)


@app.post('/api/metrics/page-view')
def page_view_metric():
    """Receive same-origin SPA navigation events; retain no raw client IP."""
    body = request.get_json(silent=True) or {}
    path = body.get("path")
    if not is_public_page(path):
        return {"ok": False, "error": "invalid page path"}, 400
    # A Redis outage can drop a metric, but must not turn a valid navigation
    # into a misleading validation error in the browser console.
    return {"ok": True, "stored": record_page_view(path)}, 202

# Initialize Babel
babel = Babel(app)
babel.init_app(app, locale_selector=get_locale)


@app.before_request
def before_request():
    _ensure_cache_warmer()
    # ── Health endpoint — always allow ──
    if request.path == "/health":
        return
    # ── Skip filtering for internal/API-Gateway requests ──
    # floci's proxy and internal checks don't send browser UAs.
    forwarded = request.headers.get("X-Forwarded-For") or ""
    is_internal = (
        request.remote_addr.startswith("172.")
        or request.remote_addr.startswith("10.")
        or request.remote_addr == "127.0.0.1"
        or "amazonaws.com" in request.headers.get("User-Agent", "")
    )
    is_cache_warmer = (
        request.remote_addr == "127.0.0.1"
        and request.headers.get("User-Agent")
        == "HoloChatStats-cache-warmer/1.0"
    )
    if not is_cache_warmer:
        # The low-priority warmer waits until foreground traffic has been
        # quiet before starting another expensive calculation.
        app.extensions["cache_warmer_last_foreground"] = time.monotonic()
    real_ip = request.headers.get("CF-Connecting-IP", request.remote_addr)
    hostname = resolve_hostname(real_ip)
    query = request.query_string.decode()
    query_str = f"?{query}" if query else ""
    # User-Agent filtering — skip for internal requests
    if not is_internal:
        ua = request.headers.get("User-Agent", "").lower()
        allowed_bots = ["googlebot", "bingbot", "duckduckbot", "applebot",
                        "facebookexternalhit", "holochatstats-llm/1.0"]
        is_browser_like = "mozilla" in ua
        is_known_bot = any(bot in ua for bot in allowed_bots)
        if not ua or (not is_browser_like and not is_known_bot):
            app.logger.warning(
                f"Blocked non-browser request from {real_ip} ({hostname}) "
                f"to {request.path}{query_str}")
            return Response("Access denied", status=403)
    # Suspicious URL patterns
    for pattern in SUSPICIOUS_PATHS:
        if re.search(pattern, request.path, re.IGNORECASE):
            app.logger.warning(
                f"Blocked suspicious path {request.path} from {real_ip} "
                f"({hostname})")
            return Response("Access denied", status=403)
    # Rate limiting (using Redis)
    if not is_cache_warmer:
        try:
            redis_conn = get_redis_connection()
            key = f"rate:{real_ip}"
            now = int(time.time())
            rate_limit_window = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
            max_requests = int(os.getenv("MAX_REQUESTS_PER_WINDOW", "120"))
            with redis_conn.pipeline() as pipe:
                pipe.zremrangebyscore(key, 0, now - rate_limit_window)
                pipe.zadd(key, {str(now): now})
                pipe.zcard(key)
                pipe.expire(key, rate_limit_window)
                _, _, req_count, _ = pipe.execute()
            if req_count > max_requests:
                app.logger.warning(
                    f"Rate limit exceeded for {real_ip} ({hostname}) - "
                    f"{req_count} reqs/{rate_limit_window}s")
                return Response("Too Many Requests", status=429)
        except redis.exceptions.ConnectionError as e:
            app.logger.warning(f"Redis unavailable for rate limiting: {e}")
        except Exception as e:
            app.logger.warning(f"Rate limiting error: {e}")
    app.logger.info(
        f"Request from {real_ip} ({hostname}) to {request.path}{query_str}")
    if 'language' not in session:
        user_lang = request.headers.get(
            'Accept-Language', 'en').split(',')[0][:2]
        session['language'] = (user_lang if user_lang in ['en', 'ja', 'ko']
                               else 'en')
    try:
        get_sqlite_connection()
    except Exception:
        pass


_cache_warmer_thread = None
_cache_warmer_lock = threading.Lock()


def _ensure_cache_warmer():
    """Start exactly one daemon warmer in this single Gunicorn worker."""
    global _cache_warmer_thread
    if _cache_warmer_thread and _cache_warmer_thread.is_alive():
        return
    with _cache_warmer_lock:
        if _cache_warmer_thread and _cache_warmer_thread.is_alive():
            return
        _cache_warmer_thread = threading.Thread(
            target=run_cache_warmer, args=(app,), daemon=True,
            name="web-cache-warmer")
        _cache_warmer_thread.start()


@app.after_request
def after_request(response):
    track_metrics(response)
    return response


@app.teardown_request
def teardown_request(exception):
    if hasattr(g, 'db_conn'):
        g.db_conn.close()
    if hasattr(g, 'sqlite_conn'):
        g.sqlite_conn.close()


_metrics_task = None


def metrics_updates():
    """Run one broadcaster per web process, independent of reconnects."""
    while True:
        try:
            with app.app_context():
                socketio.emit("metrics_update", json.dumps(get_metrics()))
        except Exception:
            app.logger.exception("Unable to publish site metrics update")
        socketio.sleep(float(os.getenv("METRICS_REFRESH_SECONDS", "10")))


@socketio.on('connect')
def start_metrics_updates():
    global _metrics_task
    if _metrics_task is None:
        _metrics_task = socketio.start_background_task(metrics_updates)


@socketio.on('request_update')
def send_update():
    """Send an immediate snapshot only to the requesting socket."""
    with app.app_context():
        socketio.emit("metrics_update", json.dumps(get_metrics()), to=request.sid)


if __name__ == '__main__':
    Compress(app)
    socketio.run(app, debug=True)
