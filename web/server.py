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
    setup_logging, resolve_hostname, get_db_connection, get_sqlite_connection,
    get_redis_connection, get_locale, track_metrics, get_metrics,
    SUSPICIOUS_PATHS
)
from api import api_bp
from routes import routes_bp

load_dotenv()

app = Flask(__name__)
socketio = SocketIO(app, async_mode='eventlet')

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

app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)

# Setup logging
setup_logging(app)

# Register blueprints
app.register_blueprint(api_bp)
app.register_blueprint(routes_bp)

# Initialize Babel
babel = Babel(app)
babel.init_app(app, locale_selector=get_locale)


@app.before_request
def before_request():
    real_ip = request.headers.get("CF-Connecting-IP", request.remote_addr)
    hostname = resolve_hostname(real_ip)
    query = request.query_string.decode()
    query_str = f"?{query}" if query else ""

    # User-Agent filtering
    ua = request.headers.get("User-Agent", "").lower()
    allowed_bots = ["googlebot", "bingbot", "duckduckbot",  "applebot", "facebookexternalhit", "holochatstats-llm/1.0"]

    is_browser_like = "mozilla" in ua
    is_known_bot = any(bot in ua for bot in allowed_bots)

    if not ua or (not is_browser_like and not is_known_bot):
        app.logger.warning(f"Blocked non-browser request from {real_ip} ({hostname}) to {request.path}{query_str}")
        return Response("Access denied", status=403)

    # Suspicious URL patterns
    for pattern in SUSPICIOUS_PATHS:
        if re.search(pattern, request.path, re.IGNORECASE):
            app.logger.warning(f"Blocked suspicious path {request.path} from {real_ip} ({hostname})")
            return Response("Access denied", status=403)

    # Rate limiting (using Redis)
    redis_conn = get_redis_connection()

    key = f"rate:{real_ip}"
    now = int(time.time())
    rate_limit_window = int(os.getenv("RATE_LIMIT_WINDOW"))
    max_requests = int(os.getenv("MAX_REQUESTS_PER_WINDOW"))
    
    with redis_conn.pipeline() as pipe:
        pipe.zremrangebyscore(key, 0, now - rate_limit_window)
        pipe.zadd(key, {str(now): now})
        pipe.zcard(key)
        pipe.expire(key, rate_limit_window)
        _, _, req_count, _ = pipe.execute()

    if req_count > max_requests:
        app.logger.warning(f"Rate limit exceeded for {real_ip} ({hostname}) - {req_count} reqs/{rate_limit_window}s")
        return Response("Too Many Requests", status=429)

    app.logger.info(f"Request from {real_ip} ({hostname}) to {request.path}{query_str}")

    if 'language' not in session:
        user_lang = request.headers.get('Accept-Language', 'en').split(',')[0][:2]
        session['language'] = user_lang if user_lang in ['en', 'ja', 'ko'] else 'en'

    if not hasattr(g, 'db_conn'):
        g.db_conn = get_db_connection()
        g.db_conn.cursor().execute("SET statement_timeout = '300s'")

    get_sqlite_connection()


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


@socketio.on('request_update')
def send_update():
    with app.app_context():
        socketio.emit("metrics_update", json.dumps(get_metrics()))
    while True:
        with app.app_context():
            metrics = get_metrics()
            socketio.emit("metrics_update", json.dumps(metrics))
        socketio.sleep(5)


if __name__ == '__main__':
    app.run(debug=True)