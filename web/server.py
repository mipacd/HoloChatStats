import hashlib
from flask import Flask, request, jsonify, render_template, session, redirect, g, Response
from flask_babel import Babel, _
from flask_socketio import SocketIO
import pytz
from werkzeug.middleware.proxy_fix import ProxyFix
import sqlite3
from datetime import datetime, timedelta, timezone
import requests
from dateutil.relativedelta import relativedelta
import configparser
import os
import psycopg2
import logging
import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import leidenalg as la
import igraph as ig
import plotly.graph_objects as go
import numpy as np
from plotly.utils import PlotlyJSONEncoder
import json
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import RunReportRequest, DateRange, Metric
from sentence_transformers import SentenceTransformer
from functools import wraps
import time
import redis
import re

def get_config(key1, key2):
    """
    Reads a value from config.ini.

    Args:
        key1 (str): Section name in config.ini.
        key2 (str): Option name in config.ini.

    Returns:
        str: Value associated with key1 and key2 in config.ini.

    Raises:
        FileNotFoundError: If config.ini is not found.
        KeyError: If key1 or key2 are not found in config.ini.
    """
    config = configparser.ConfigParser()
    caller_script_dir = os.path.dirname(os.path.abspath(__name__))
    ini_file = os.path.join(caller_script_dir, 'config.ini')
    if not config.read(ini_file):
        raise FileNotFoundError("config.ini not found.")
    try:
        return config[key1][key2]
    except KeyError as e:
        raise KeyError(f"Key not found in config.ini: {e}")

app = Flask(__name__)
socketio = SocketIO(app, async_model='eventlet')
# Setup session key and babel configuration
app.config["SECRET_KEY"] = get_config("Settings", "SecretKey")
app.config["SESSION_TYPE"] = "filesystem"
app.config["BABEL_DEFAULT_LOCALE"] = "en"
app.config["BABEL_TRANSLATION_DIRECTORIES"] = "translations"
app.config["JSON_AS_ASCII"] = False
app.config["GA_ID"] = get_config("Settings", "GoogleAnalyticsID")
app.config["OPENROUTER_URL"] = "https://openrouter.ai/api/v1"  # OpenRouter URL
app.config["OPENROUTER_MODEL"] = "deepseek/deepseek-chat-v3-0324:free"  # Default model  
app.config["DAILY_LIMIT"] = 3  # 3 queries per user per day

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = get_config("API", "GAAPIKeyFile")

print("Loading sentence transformer model for vector search...")
EMBEDDER = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("Model loaded successfully.")

app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)


DB_CONFIG = {
    "dbname": get_config("Database", "DBName"),
    "user": get_config("Database", "DBUser"),
    "password": get_config("Database", "DBPass"),
    "host": get_config("Database", "DBHost"),
    "port": get_config("Database", "DBPort"),
    "client_encoding": "UTF8"
}
LANGUAGES = {
    'en': 'English',
    'ja': '日本語',
    'ko': '한국어'
}
REDIS_CONFIG = {
    "host": get_config("Redis", "Host"),
    "port": get_config("Redis", "Port"),
}

# Setup logging
def setup_logging():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    app.logger.handlers.clear()
    app.logger.addHandler(handler)
    app.logger.setLevel(logging.INFO)

    # Optional: silence werkzeug logs if you want full control
    logging.getLogger('werkzeug').setLevel(logging.WARNING)

setup_logging()

def get_google_analytics_visitors():
    """Fetch visitor count from Google Analytics"""
    count = 0
    client = BetaAnalyticsDataClient()
    request = RunReportRequest(
        property=f"properties/{ get_config('API', 'GAOldHCSPropertyID') }",
        date_ranges=[DateRange(start_date="365daysAgo", end_date="today")],
        metrics=[Metric(name="activeUsers")],
    )
    response = client.run_report(request)

    count += int(response.rows[0].metric_values[0].value)

    request = RunReportRequest(
        property=f"properties/ { get_config('API', 'GANewHCSPropertyID') }",
        date_ranges=[DateRange(start_date="365daysAgo", end_date="today")],
        metrics=[Metric(name="activeUsers")],
    )
    response = client.run_report(request)

    count += int(response.rows[0].metric_values[0].value)
    
    return count

# Access logging and detect language
@app.before_request
def before_request():
    real_ip = request.headers.get("CF-Connecting-IP", request.remote_addr)
    query = request.query_string.decode()
    query_str = f"?{query}" if query else ""

    # Non-browser and bot traffic blocking
    ua = request.headers.get("User-Agent", "").lower()
    allowed_bots = ["googlebot", "bingbot", "duckduckbot", "applebot", "facebookexternalhit"]

    is_browser_like = "mozilla" in ua
    is_known_bot = any(bot in ua for bot in allowed_bots)

    #if not ua or (not is_browser_like and not is_known_bot):
    #    app.logger.warning(f"Blocked non-browser request from {real_ip} to {request.path}{query_str}")
        #return Response("Access denied", status=403)
   # else:
    #    app.logger.info(f"Request from {real_ip} to {request.path}{query_str}")


    if 'language' not in session:
        user_lang = request.headers.get('Accept-Language', 'en').split(',')[0][:2]
        session['language'] = user_lang if user_lang in ['en', 'ja', 'ko'] else 'en'
    
    # Initialize database connection
    if not hasattr(g, 'db_conn'):
        g.db_conn = get_db_connection()
        g.db_conn.cursor().execute("SET statement_timeout = '300s'")
    
    # Initialize Redis connection
    if not hasattr(g, 'redis_conn'):
        g.redis_conn = redis.StrictRedis(
            host=REDIS_CONFIG["host"],
            port=REDIS_CONFIG["port"],
            decode_responses=True
        )

@app.teardown_request
def teardown_request(exception):
    if hasattr(g, 'db_conn'):
        g.db_conn.close()

def get_locale():
    return session.get('language', 'en')

babel = Babel(app)
babel.init_app(app, locale_selector=get_locale)

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

@app.context_processor
def inject_ga_id():
    return { "GA_ID" : app.config.get("GA_ID", "") }


@app.route('/')
def index():
    return render_template("index.html", _=_, get_locale=get_locale)

# Language selection
@app.route('/set_language/<language>')
def set_language(language):
    if language in ['en', 'ja', 'ko']:
        session['language'] = language
        print("Language stored in session:", session['language'])
        return redirect(request.referrer or "/")
    return "Invalid language", 400

def get_sqlite_connection():
    """Creates a new SQLite connection stored in Flask's g object"""
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

# Track site metrics in Redis
def track_metrics(response):
    """Tracks unique visitors per country and aggregates page views over 30 days."""
    if response.status_code != 200:
        return response
    country = request.headers.get("CF-IPCountry", "Unknown")
    page = request.path
    if any(path in page for path in ["/api/", "/static/", "/favicon.ico", "/set_language/"]):
        return response

    # Get current UTC date in YYYY-MM-DD format
    today = datetime.now(pytz.utc).strftime("%Y-%m-%d")
    
    visitor_ip = hashlib.sha256(request.headers.get("CF-Connecting-IP", request.remote_addr).encode()).hexdigest()

    # Track unique visitors per country
    g.redis_conn.sadd(f"unique_visitors_country:{country}:{today}", visitor_ip)
    g.redis_conn.sadd(f"unique_visitors:{today}", visitor_ip)

    # Aggregate page views across 30 days instead of daily counts
    g.redis_conn.hincrby("page_views_30d", page, 1)

    # Ensure expiry for cleanup
    expiry_time = 2592000  # 30 days in seconds
    g.redis_conn.expire(f"unique_visitors_country:{country}:{today}", expiry_time)
    g.redis_conn.expire(f"unique_visitors:{today}", expiry_time)
    g.redis_conn.expire("page_views_30d", expiry_time)
    g.redis_conn.expire(f"cache_hits:{today}", expiry_time)
    g.redis_conn.expire(f"cache_misses:{today}", expiry_time)

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
        (today - relativedelta(days=i)).strftime("%Y-%m-%d"): redis_conn.scard(f"unique_visitors:{(today - relativedelta(days=i)).strftime('%Y-%m-%d')}")
        for i in range(30)
    }

    metrics["cache_data"] = {
        (today - relativedelta(days=i)).strftime("%Y-%m-%d"): {
            "cache_hits": int(redis_conn.get(f"cache_hits:{(today - relativedelta(days=i)).strftime('%Y-%m-%d')}") or 0),
            "cache_misses": int(redis_conn.get(f"cache_misses:{(today - relativedelta(days=i)).strftime('%Y-%m-%d')}") or 0)
        } for i in range(30)
    }

    return metrics

@socketio.on('request_update')
def send_update():
    with app.app_context():
        socketio.emit("metrics_update", json.dumps(get_metrics()))
    while True:
        with app.app_context():  # Ensures Flask context is active
            metrics = get_metrics()
            socketio.emit("metrics_update", json.dumps(metrics))
        socketio.sleep(5)  # Adjust update interval as needed


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

@app.before_request
def before_request():
    real_ip = request.headers.get("CF-Connecting-IP", request.remote_addr)
    app.logger.info(f"Request from {real_ip} to {request.path}")
    if 'language' not in session:
        user_lang = request.headers.get('Accept-Language', 'en').split(',')[0][:2]
        session['language'] = user_lang if user_lang in ['en', 'ja', 'ko'] else 'en'
    
    g.db_conn = get_db_connection()
    g.db_conn.cursor().execute("SET statement_timeout = '300s'")
    # Initialize SQLite connection for this request
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


def timeout(seconds=5):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = f(*args, *kwargs)
            duration = time.time() - start
            if duration > seconds:
                raise TimeoutError("Request timed out")
            return result
        return wrapper
    return decorator

def check_rate_limit(ip):
    """Check SQLite rate limit for IP, returns remaining requests"""
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
        return max(0, app.config["DAILY_LIMIT"] - count)
    except Exception as e:
        app.logger.error(f"Rate limit check failed: {str(e)}")
        return app.config["DAILY_LIMIT"]  # Fail open by returning full limit
    
# Return current month in YYYY-MM format
def get_current_month():
    return datetime.utcnow().strftime('%Y-%m')

@app.route('/api/llm/query', methods=['POST'])
@timeout(300)
def llm_query():
    try:
        data = request.get_json()
        # Get Cloudflare IP if available, fallback to remote address
        real_ip = request.headers.get("CF-Connecting-IP", request.remote_addr)
        
        # Check SQLite daily limit
        remaining = check_rate_limit(real_ip)
        if remaining <= 0:
            return jsonify({
                "error": "Daily query limit reached",
                "limit": app.config["DAILY_LIMIT"]
            }), 429
        if not data or 'question' not in data or 'chart_context' not in data:
            return jsonify({"error": "Invalid request format"}), 400
        
        question_hash = hash(data['question'])
        redis_key = f"llm_query_{question_hash}"
        cached_response = g.redis_conn.get(redis_key)
        if cached_response:
            inc_cache_hit_count()
            return jsonify(json.loads(cached_response))
        inc_cache_miss_count()

        # Format prompt based on chart type
        context = data['chart_context']
        chart_type = context.get('chart_type', 'membership_counts')
        
        if chart_type == 'membership_counts':
            prompt = f"""You are an expert data analyst answering questions about VTuber membership counts.

Context:
- Chart Type: Membership counts by duration  
- Group: {context.get('selected_group', 'Unknown')}
- Month: {context.get('selected_month', 'Unknown')}

Data Format Guidelines:
1. Numbers represent unique members (not percentages)
2. Durations: New (0), 1 Month, 2 Months, etc.
3. Values are exact counts - don't estimate

Current Data (CSV):
{context.get('current_data', [])}

Rules for Responses:
- Only use the provided data
- Sum numbers under a channel to get the monthly total unless a duration is specified
- If unsure, say "Data not available"
- Keep answers under 3 sentences  
- Do not provide calculations or reasoning

Question: {data['question']}

Answer:"""
        
        elif chart_type == 'user_changes':
            prompt = f"""You are an expert data analyst answering questions about VTuber user changes.

Context:
- Chart Type: User changes
- Group: {context.get('selected_group', 'Unknown')}  
- Month: {context.get('selected_month', 'Unknown')}
- Data includes: channel, users gained, users lost, net change
- Threshold: Users met 5 message threshold in one month but not the other

Current Data (CSV):
{context.get('current_data', [])}

Rules for Responses:
- Only use the provided data
- Focus on notable gains/losses and net changes
- If unsure, say "Data not available"
- Keep answers under 3 sentences
- Do not provide calculations unless explicitly asked

Question: {data['question']}

Answer:"""
        
        elif chart_type == 'membership_changes':  
            prompt = f"""You are an expert data analyst answering questions about VTuber membership changes.

Context:  
- Chart Type: Membership gains/losses
- Group: {context.get('selected_group', 'Unknown')}
- Month: {context.get('selected_month', 'Unknown')}
- Data includes: channel, gains, losses, differential
- Based on last recorded message from each user between months

Current Data (CSV):  
{context.get('current_data', [])}

Rules for Responses:
- Only use the provided data
- Focus on notable gains/losses and net differentials
- If unsure, say "Data not available"  
- Keep answers under 3 sentences
- Do not provide calculations unless explicitly asked

Question: {data['question']}

Answer:"""
        
        elif chart_type == 'chat_makeup':
            prompt = f"""You are an expert data analyst answering questions about VTuber chat composition.

Context:
- Chart Type: Chat makeup average rates per minute by language and emote usage
- Group: {context.get('selected_group', 'Unknown')}
- Month: {context.get('selected_month', 'Unknown')}  
- Data includes: channel, EN/ES/ID, JP, KR, RU, Emote message classifications

Current Data (CSV):
{context.get('current_data', [])}

Rules for Responses:
- Only use the provided data
- Focus on notable language distributions and emote usage
- If unsure, say "Data not available"
- Keep answers under 3 sentences
- Do not provide calculations unless explicitly asked

Question: {data['question']}

Answer:"""
        

        # Block Japan and South Korea via Cloudflare headers
        country = request.headers.get("CF-IPCountry", "").upper()
        if country in ["JP", "KR"]:
            return jsonify({
                "error": "Service not available in your country",
                "country": country
            }), 403

        # Call OpenRouter
        response = requests.post(
            f"{app.config['OPENROUTER_URL']}/chat/completions",
            json={
                "model": app.config["OPENROUTER_MODEL"],
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            },
            headers={
                "Authorization": f"Bearer {get_config('API', 'OpenRouterAPIKey')}",
                "Content-Type": "application/json"
            },
            timeout=60
        )
        response.raise_for_status()

        # Process OpenRouter response
        result = response.json()
        print(result)
        answer = result["choices"][0]["message"]["content"].strip()
        
        # Update usage count
        conn = get_sqlite_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE usage SET count = count + 1 
                WHERE ip = ? AND date = ?
            """, (real_ip, datetime.utcnow().strftime('%Y-%m-%d')))
            conn.commit()
        except Exception as e:
            app.logger.error(f"Failed to update usage count: {str(e)}")

        # Cache the response in Redis
        output = {
            "answer": answer,
            "supporting_data": None,
            "error": None,
            "remaining_queries": remaining - 1
        }
        g.redis_conn.set(redis_key, json.dumps(output))

        return jsonify(output)


    except requests.exceptions.RequestException as e:
        app.logger.error(f"OpenRouter connection error: {str(e)}")
        return jsonify({"error": "LLM service unavailable"}), 503
    except TimeoutError:
        return jsonify({"error": "Request timed out"}), 504
    except Exception as e:
        app.logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

def streaming_hours_query(aggregation_function, group=None):
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

@app.route('/api/channel_clustering', methods=['GET'])
def channel_clustering():
    try:
        filter_month = request.args.get("month")
        percentile = request.args.get("percentile", "95")
        graph_type = request.args.get("type", "2d")  # Default to 2d if not specified
        if not filter_month:
            return jsonify({"error": "Month filter (e.g., '2025-03') is required"}), 400
        redis_key = f"channel_clustering_{filter_month}_{percentile}_{graph_type}"

        cursor = g.db_conn.cursor()

        # Check if data is cached in Redis
        cached_data = g.redis_conn.get(redis_key)
        if cached_data:
            inc_cache_hit_count()
            # If cached data is found, return it
            return jsonify(json.loads(cached_data))
        inc_cache_miss_count()

        cursor.execute("""
            SELECT ud.user_id, ch.channel_name, SUM(ud.total_message_count) AS message_weight
            FROM user_data ud
            JOIN channels ch ON ud.channel_id = ch.channel_id
            WHERE DATE_TRUNC('month', ud.last_message_at) = %s::DATE
            GROUP BY ud.user_id, ch.channel_name;
        """, (filter_month + "-01",))
        rows = cursor.fetchall()

        cursor.close()

        data = pd.DataFrame(rows, columns=['user_id', 'channel_name', 'message_weight'])
        if data.empty:
            return jsonify({"error": "No data found for the specified month"}), 404

        user_channel_matrix = data.pivot(index='user_id', columns='channel_name', values='message_weight').fillna(0)
        similarity_matrix = cosine_similarity(user_channel_matrix.T)
        channel_names = user_channel_matrix.columns

        G = nx.Graph()
        threshold = np.percentile(similarity_matrix, float(percentile))

        for i, channel_a in enumerate(channel_names):
            for j, channel_b in enumerate(channel_names):
                if i != j and similarity_matrix[i, j] > threshold:
                    G.add_edge(channel_a, channel_b, weight=similarity_matrix[i, j])

        # Convert networkx Graph to igraph Graph
        g_igraph = ig.Graph.from_networkx(G)
        
        # Find partition using Leiden algorithm
        partition = la.find_partition(
            g_igraph, 
            la.RBConfigurationVertexPartition,
            weights='weight',
            resolution_parameter=1.0
        )
        
        # Convert partition
        partition_dict = {g_igraph.vs[node]['_nx_name']: partition.membership[node] 
                         for node in range(g_igraph.vcount())}
        community_colors = [partition_dict[node] for node in G.nodes]

        fig = go.Figure()
        
        if graph_type == "3d":
            pos = nx.forceatlas2_layout(G, dim=3, strong_gravity=True)
            node_x, node_y, node_z = zip(*[pos[node] for node in G.nodes()])
            
            edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
            min_weight, max_weight = min(edge_weights), max(edge_weights)
            normalized_weights = [(w - min_weight) / (max_weight - min_weight) for w in edge_weights]
            edge_traces = []
            hover_traces = []  # Separate layer for tooltips
            num_hover_points = 10  # Increase for better line coverage

            edge_x, edge_y, edge_z = [], [], []  # Initialize lists for 3D edges

            for u, v in G.edges():
                x0, y0, z0 = pos[u]  # Unpack 3D position for first node
                x1, y1, z1 = pos[v]  # Unpack 3D position for second node
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edge_z.extend([z0, z1, None])  # Include Z coordinate


            for (u, v), norm_weight in zip(G.edges(), normalized_weights):
                x0, y0, z0 = pos[u]
                x1, y1, z1 = pos[v]
                adjusted_opacity = 0.1 + (norm_weight ** 1.1) * 0.9

                # Line Trace (3D)
                fig.add_trace(go.Scatter3d(
                    x=edge_x,
                    y=edge_y,
                    z=edge_z,
                    mode='lines',
                    line=dict(width=1.5, color=f'rgba(0,0,0,{adjusted_opacity})'),  # Darkness scaling
                    hoverinfo='none'  # Disable hover for lines themselves
                ))

                line_x = np.linspace(x0, x1, num_hover_points)
                line_y = np.linspace(y0, y1, num_hover_points)
                line_z = np.linspace(z0, z1, num_hover_points)  # Include Z dimension

                hover_traces.append(go.Scatter3d(
                    x=line_x.tolist(),
                    y=line_y.tolist(),
                    z=line_z.tolist(),
                    mode='markers',
                    marker=dict(size=6, color='rgba(255,255,255,0)'),  # Fully transparent markers
                    hoverinfo='text',
                    hovertext=[f"{u} ↔ {v}<br>Score: {G[u][v]['weight'] * 100:.2f}"] * num_hover_points
                ))

            # Add both traces to the figure
            for trace in edge_traces:
                fig.add_trace(trace)
            for trace in hover_traces:
                fig.add_trace(trace)

            
            # Add nodes as 3D scatter points
            fig.add_trace(go.Scatter3d(
                x=node_x,
                y=node_y,
                z=node_z,
                mode='markers+text',
                text=list(G.nodes),
                textposition="top center",
                hoverinfo='text',
                hovertext=[f"{node}<br>Connected to: {', '.join(G.neighbors(node))}" for node in G.nodes],
                marker=dict(
                    size=12,
                    color=community_colors,
                    colorscale='Viridis',
                    line=dict(color='black', width=1)
                )
            ))
            
            formatted_month = datetime.strptime(filter_month, "%Y-%m").strftime("%B %Y")
            fig.update_layout(
                title=f"Channel User Similarity Graph (3D) for {formatted_month}",
                title_x=0.5,
                showlegend=False,
                margin=dict(l=10, r=10, t=50, b=10),
                scene=dict(
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    zaxis=dict(visible=False)
                )
            )
        else:
            # Default 2D layout
            pos = nx.forceatlas2_layout(G, strong_gravity=True)  # Better spacing
            node_x, node_y = zip(*pos.values())

            edge_x, edge_y = [], []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

            edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
            min_weight, max_weight = min(edge_weights), max(edge_weights)
            normalized_weights = [(w - min_weight) / (max_weight - min_weight) for w in edge_weights]
            edge_traces = []
            hover_traces = []  # Separate layer for tooltips
            num_hover_points = 10  # Increase for better line coverage

            for (u, v), norm_weight in zip(G.edges(), normalized_weights):
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                adjusted_opacity = 0.1 + (norm_weight ** 1.1) * 0.9

                # Line Trace
                edge_traces.append(go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=1.5, color=f'rgba(0,0,0,{adjusted_opacity})'),  # Darkness scaling
                    hoverinfo='none'  # Disable hover for lines themselves
                ))

                line_x = np.linspace(x0, x1, num_hover_points)
                line_y = np.linspace(y0, y1, num_hover_points)

                # Midpoint Marker for Hover Tooltip
                hover_traces.append(go.Scatter(
                    x=line_x.tolist(),
                    y=line_y.tolist(),
                    mode='markers',
                    marker=dict(size=6, color='rgba(255,255,255,0)'),  # Fully transparent markers
                    hoverinfo='text',
                    hovertext=[f"{u} ↔ {v}<br>Score: {G[u][v]['weight'] * 100:.2f}"] * num_hover_points
                ))

            # Add both traces to the figure
            for trace in edge_traces:
                fig.add_trace(trace)
            for trace in hover_traces:
                fig.add_trace(trace)

            fig.add_trace(go.Scatter(
                x=node_x,
                y=node_y,
                mode='markers+text',
                text=list(G.nodes),
                textposition="top center",
                hoverinfo='text',  # Only show hover text
                hovertext=[f"{node}<br>Connected to: {', '.join(G.neighbors(node))}" for node in G.nodes],  # Show connections
                marker=dict(
                    size=12,  # Larger nodes
                    color=community_colors,
                    colorscale='Viridis',  # Better contrast
                    line=dict(color='black', width=1)  # Node border for clarity
                )
            ))

            formatted_month = datetime.strptime(filter_month, "%Y-%m").strftime("%B %Y")
            fig.update_layout(
                title=f"Channel User Similarity Graph for {formatted_month}",
                title_x=0.5,
                showlegend=False,
                margin=dict(l=10, r=10, t=50, b=10),
                xaxis=dict(visible=False),  # Hide X-axis
                yaxis=dict(visible=False),  # Hide Y-axis
                plot_bgcolor='white'
            )


        graph_json = json.dumps(fig, cls=PlotlyJSONEncoder)

        output = { "graph_json": graph_json }

        g.redis_conn.set(redis_key, json.dumps(output))

        return jsonify(output)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
def get_previous_two_months():
    today = datetime.today().replace(day=1)
    prev_month = today - timedelta(days=1)
    prev2_month = (prev_month.replace(day=1)) - timedelta(days=1)
    return [prev2_month.strftime('%Y-%m'), prev_month.strftime('%Y-%m')]

@app.route("/api/recommend", methods=["GET"])
def recommend_channels():
    """
    API to recommend channels for a user based on their recent activity.
    Accepts a unique user_id or a user handle (@username) as input.
    """
    try:
        identifier = request.args.get("identifier", "")
        if not identifier:
            return jsonify({"error": "Missing required parameter: 'identifier' is required."}), 400

        conn = get_db_connection()
        cursor = conn.cursor()
        user_id = None

        if identifier.startswith('@'):
            cursor.execute("SELECT user_id FROM users WHERE username = %s", (identifier,))
            result = cursor.fetchone()
            if not result:
                return jsonify({"error": f"User handle '{identifier}' not found"}), 404
            user_id = result[0]
        else:
            # Assume it's a user_id directly
            user_id = identifier

        redis_key = f"channel_recommendations_{user_id}"
        cached_data = g.redis_conn.get(redis_key)
        if cached_data:
            inc_cache_hit_count()
            return jsonify(json.loads(cached_data))
        inc_cache_miss_count()

        # Get the most recent full month of data for recommendations
        # (Assuming we want recommendations based on the last complete month)
        last_month = datetime.utcnow().date().replace(day=1) - relativedelta(days=1)
        month_to_query = last_month.strftime("%Y-%m-01")

        # --- Intermediate Caching for the month's full activity data ---
        redis_intermediate_key = f"channel_recommendation_data_{month_to_query}"
        cached_rows = g.redis_conn.get(redis_intermediate_key)
        
        if cached_rows:
            rows = json.loads(cached_rows)
        else:
            # Query user activity for the last complete month
            cursor.execute("""
                SELECT ud.user_id, ch.channel_name, SUM(ud.total_message_count) AS message_weight
                FROM user_data ud
                JOIN channels ch ON ud.channel_id = ch.channel_id
                WHERE DATE_TRUNC('month', ud.last_message_at) = %s::DATE
                GROUP BY ud.user_id, ch.channel_name;
            """, (month_to_query,))
            rows = cursor.fetchall()
            g.redis_conn.set(redis_intermediate_key, json.dumps(rows))

        if not rows:
            return jsonify({"error": "Not enough recent data available to generate recommendations."}), 404

        data = pd.DataFrame(rows, columns=['user_id', 'channel_name', 'message_weight'])
        
        # Create user-channel interaction matrix
        user_channel_matrix = data.pivot(index='user_id', columns='channel_name', values='message_weight').fillna(0)

        # Compute cosine similarity between channels
        similarity_matrix = cosine_similarity(user_channel_matrix.T)
        channel_names = user_channel_matrix.columns
        similarity_df = pd.DataFrame(similarity_matrix, index=channel_names, columns=channel_names)

        # Get channels the user has interacted with
        if user_id not in user_channel_matrix.index:
            return jsonify({"error": "No activity found for this user in the last month."}), 404

        user_vector = user_channel_matrix.loc[user_id]
        user_channels = user_vector[user_vector > 0].index

        # Calculate recommendation scores
        recommendation_scores = similarity_df[user_channels].sum(axis=1)
        recommendation_scores = recommendation_scores.drop(labels=user_channels, errors='ignore')

        top_recommendations = recommendation_scores.sort_values(ascending=False).head(5)
        
        # Normalize and scale scores
        ideal_max = len(user_channels)
        raw_normalized = (top_recommendations / ideal_max) * 100
        normalized_scores = np.log1p(raw_normalized) / np.log1p(100) * 100

        response = [
            {"channel_name": channel, "score": round(score, 2)}
            for channel, score in normalized_scores.items()
        ]

        output = { "recommended_channels": response }
        g.redis_conn.set(redis_key, json.dumps(output)) # Cache final result for 24 hours

        return jsonify(output)

    except Exception as e:
        # Log the full error for debugging
        print(f"An error occurred in recommend_channels: {e}")
        return jsonify({"error": "An internal server error occurred."}), 500
    finally:
        if 'conn' in locals() and conn:
            cursor.close()
            conn.close()




@app.route('/api/get_monthly_streaming_hours', methods=['GET'])
def get_monthly_streaming_hours():
    """API to fetch total streaming hours per month for a given channel."""

    channel_name = request.args.get('channel')
    redis_key = f"monthly_streaming_hours_{channel_name}"
    # Check if data is cached in Redis
    cached_data = g.redis_conn.get(redis_key)
    if cached_data:
        inc_cache_hit_count()
        # If cached data is found, return it
        return jsonify(json.loads(cached_data))
    inc_cache_miss_count()

    if not channel_name:
        return jsonify({"error": _("Missing required parameters")}), 400

    query = """
        SELECT 
            DATE_TRUNC('month', v.end_time)::DATE AS month, 
            ROUND(CAST(SUM(EXTRACT(EPOCH FROM v.duration)) / 3600 AS NUMERIC), 2) AS total_streaming_hours
        FROM videos v
        JOIN channels c ON v.channel_id = c.channel_id
        WHERE c.channel_name = %s
        GROUP BY month
        ORDER BY month;
    """

    cursor = g.db_conn.cursor()
    cursor.execute(query, (channel_name,))
    results = cursor.fetchall()
    cursor.close()

    # Convert results into JSON format
    output = jsonify([{"month": row[0].strftime('%Y-%m'), "total_streaming_hours": row[1]} for row in results])
    g.redis_conn.set(redis_key, output.get_data(as_text=True))
    return output


@app.route('/api/get_group_total_streaming_hours', methods=['GET'])
def get_group_total_streaming_hours():
    month = request.args.get('month', datetime.utcnow().strftime('%Y-%m'))
    group = request.args.get('group', None)

    redis_key = f"group_total_streaming_hours_{group}_{month}"
    cached_data = g.redis_conn.get(redis_key)
    if cached_data:
        inc_cache_hit_count()
        return jsonify(json.loads(cached_data))
    inc_cache_miss_count()
    
    query, params = streaming_hours_query('SUM')
    try:
        with g.db_conn.cursor() as cur:
            cur.execute(query, params)
            results = cur.fetchall()
        data = [
            {"channel": row[0], "month": row[1].strftime('%Y-%m'), "hours": round(row[2], 2)}
            for row in results if row[2] is not None
        ]
        output = jsonify({"success": True, "data": data})
        g.redis_conn.set(redis_key, output.get_data(as_text=True))
        return output
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/get_group_avg_streaming_hours', methods=['GET'])
def get_group_avg_streaming_hours():
    month = request.args.get('month', datetime.utcnow().strftime('%Y-%m'))
    group = request.args.get('group', None)

    redis_key = f"group_avg_streaming_hours_{group}_{month}"
    cached_data = g.redis_conn.get(redis_key)
    if cached_data:
        inc_cache_hit_count()
        return jsonify(json.loads(cached_data))
    inc_cache_miss_count()
    
    query, params = streaming_hours_query('AVG')
    try:
        with g.db_conn.cursor() as cur:
            cur.execute(query, params)
            results = cur.fetchall()
        data = [
            {"channel": row[0], "month": row[1].strftime('%Y-%m'), "hours": round(row[2], 2)}
            for row in results if row[2] is not None
        ]
        output = jsonify({"success": True, "data": data})
        g.redis_conn.set(redis_key, output.get_data(as_text=True))
        return output
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/get_group_max_streaming_hours', methods=['GET'])
def get_group_max_streaming_hours():
    month = request.args.get('month', datetime.utcnow().strftime('%Y-%m'))
    group = request.args.get('group', None)

    redis_key = f"group_max_streaming_hours_{group}_{month}"
    cached_data = g.redis_conn.get(redis_key)
    if cached_data:
        inc_cache_hit_count()
        return jsonify(json.loads(cached_data))
    inc_cache_miss_count()
    
    query, params = streaming_hours_query('MAX')
    try:
        with g.db_conn.cursor() as cur:
            cur.execute(query, params)
            results = cur.fetchall()
        data = [
            {"channel": row[0], "month": row[1].strftime('%Y-%m'), "hours": round(row[2], 2)}
            for row in results if row[2] is not None
        ]
        output = jsonify({"success": True, "data": data})
        g.redis_conn.set(redis_key, output.get_data(as_text=True))
        return output
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    
@app.route('/api/get_group_chat_makeup', methods=['GET'])
def get_group_chat_makeup():
    """API to fetch chat makeup statistics per channel, with optional filtering."""
    group = request.args.get('group', None)  # "Hololive", "Indie", or None
    month = request.args.get('month', datetime.utcnow().strftime('%Y-%m'))  # Default: current month

    redis_key = f"group_chat_makeup_{group}_{month}"
    cached_data = g.redis_conn.get(redis_key)
    if cached_data:
        inc_cache_hit_count()
        return jsonify(json.loads(cached_data))
    inc_cache_miss_count()

    # Convert 'YYYY-MM' to 'YYYY-MM-01' and the end of that month ('YYYY-MM-01')
    start_month = datetime.strptime(month + "-01", '%Y-%m-%d').strftime('%Y-%m-01')


    # SQL query to calculate rates
    query = """
        WITH streaming_time AS (
            SELECT 
                v.channel_id,
                DATE_TRUNC('month', v.end_time AT TIME ZONE 'UTC') AS observed_month,
                SUM(EXTRACT(EPOCH FROM v.duration)) / 60 AS total_streaming_minutes
            FROM videos v
            WHERE duration IS NOT NULL 
            AND duration > INTERVAL '0 seconds'
            AND has_chat_log = 't'
            AND DATE_TRUNC('month', v.end_time AT TIME ZONE 'UTC') = %s::DATE
            GROUP BY v.channel_id, observed_month
        )
        SELECT 
            c.channel_name,
            st.observed_month,
            SUM(cls.es_en_id_count)::DECIMAL / NULLIF(SUM(st.total_streaming_minutes), 0) AS es_en_id_rate_per_minute,
            SUM(cls.jp_count)::DECIMAL / NULLIF(SUM(st.total_streaming_minutes), 0) AS jp_rate_per_minute,
            SUM(cls.kr_count)::DECIMAL / NULLIF(SUM(st.total_streaming_minutes), 0) AS kr_rate_per_minute,
            SUM(cls.ru_count)::DECIMAL / NULLIF(SUM(st.total_streaming_minutes), 0) AS ru_rate_per_minute,
            SUM(cls.emoji_count)::DECIMAL / NULLIF(SUM(st.total_streaming_minutes), 0) AS emoji_rate_per_minute
        FROM chat_language_stats_mv cls
        JOIN channels c ON cls.channel_id = c.channel_id
        JOIN streaming_time st 
            ON st.channel_id = c.channel_id 
            AND CAST(st.observed_month AS DATE) = CAST(cls.observed_month AS DATE)
    """

    params = [start_month]

    # Group filtering
    if group:
        query += " WHERE c.channel_group = %s"
        params.append(group)

    query += " GROUP BY c.channel_name, st.observed_month ORDER BY SUM(st.total_streaming_minutes) DESC"

    try:
        cur = g.db_conn.cursor()
        cur.execute(query, params)
        results = cur.fetchall()
        cur.close()

        # Convert to JSON
        data = [
            {
                "channel_name": row[0],
                "observed_month": row[1].strftime('%Y-%m'),  # Format date for consistency
                "es_en_id_rate_per_minute": round(float(row[2]) if row[2] is not None else 0, 2),
                "jp_rate_per_minute": round(float(row[3]) if row[3] is not None else 0, 2),
                "kr_rate_per_minute": round(float(row[4]) if row[4] is not None else 0, 2),
                "ru_rate_per_minute": round(float(row[5]) if row[5] is not None else 0, 2),
                "emoji_rate_per_minute": round(float(row[6]) if row[6] is not None else 0, 2)
            }
            for row in results
        ]

        output =  jsonify({"success": True, "data": data})
        g.redis_conn.set(redis_key, output.get_data(as_text=True))
        return output

    except Exception as e:
        print(f"Error: {e}")  # Print error message for debugging
        return jsonify({"success": False, "error": str(e)}), 500
    
@app.route('/api/get_common_users', methods=['GET'])
def get_common_users():
    """API to get common users between two (channel, month) pairs using set arithmetic."""
    channel_a_name = request.args.get('channel_a')
    month_a_str = request.args.get('month_a')  # YYYY-MM format
    channel_b_name = request.args.get('channel_b')
    month_b_str = request.args.get('month_b')  # YYYY-MM format

    if not (channel_a_name and month_a_str and channel_b_name and month_b_str):
        return jsonify({"error": _("Missing required parameters")}), 400
    
    redis_key = f"common_users_{channel_a_name}_{month_a_str}_{channel_b_name}_{month_b_str}"
    cached_data = g.redis_conn.get(redis_key)
    if cached_data:
        inc_cache_hit_count()
        return jsonify(json.loads(cached_data))
    inc_cache_miss_count()
    
    cursor = g.db_conn.cursor()
    
    # A simple, reusable query to get all user IDs for a given channel and month
    query = """
        SELECT DISTINCT mua.user_id
        FROM mv_user_monthly_activity mua
        JOIN channels c ON mua.channel_id = c.channel_id
        WHERE c.channel_name = %s
          AND mua.observed_month = %s::DATE;
    """
    
    try:
        # Fetch the list of users for the first channel/month
        month_a_date = f"{month_a_str}-01"
        cursor.execute(query, (channel_a_name, month_a_date))
        users_a = {row[0] for row in cursor.fetchall()}

        # Fetch the list of users for the second channel/month
        month_b_date = f"{month_b_str}-01"
        cursor.execute(query, (channel_b_name, month_b_date))
        users_b = {row[0] for row in cursor.fetchall()}
        
    except Exception as e:
        cursor.close()
        return jsonify({"error": f"Database query failed: {e}"}), 500
    finally:
        cursor.close()

    if not users_a or not users_b:
        return jsonify({}) # Return empty if one or both have no users

    # Perform the set arithmetic in Python
    common_users = users_a.intersection(users_b)
    
    num_common_users = len(common_users)
    total_users_a = len(users_a)
    total_users_b = len(users_b)

    # Calculate percentages
    percent_a_to_b = (100.0 * num_common_users / total_users_a) if total_users_a > 0 else 0
    percent_b_to_a = (100.0 * num_common_users / total_users_b) if total_users_b > 0 else 0

    data = {
        "channel_a": channel_a_name,
        "channel_b": channel_b_name,
        "month_a": month_a_str,
        "month_b": month_b_str,
        "num_common_users": num_common_users,
        "percent_A_to_B_users": round(percent_a_to_b, 2),
        "percent_B_to_A_users": round(percent_b_to_a, 2)
    }

    g.redis_conn.set(redis_key, json.dumps(data), ex=86400) # Cache for 24 hours
    
    return jsonify(data)

@app.route('/api/get_common_users_matrix', methods=['GET'])
def get_common_users_matrix():
    """
    API to generate a matrix of common user or member percentages for a
    given list of channels and a specific month.
    """
    month_str = request.args.get('month')
    channels_str = request.args.get('channels')

    members_only_str = request.args.get('members_only', 'false')
    members_only = members_only_str.lower() in ['true', '1', 'yes']

    if not (month_str and channels_str):
        return jsonify({"error": "Missing 'month' or 'channels' parameter"}), 400
        
    channel_names = [name.strip() for name in channels_str.split(',')]
    if len(channel_names) < 2:
        return jsonify({"error": "Please provide at least two channel names."}), 400

    sorted_channels_key = ",".join(sorted(channel_names))
    matrix_type = "members" if members_only else "users"
    redis_key = f"common_matrix_percent_{matrix_type}_{sorted_channels_key}_{month_str}"
    cached_data = g.redis_conn.get(redis_key)
    if cached_data:
        return jsonify(json.loads(cached_data))

    if members_only:
        # This query filters for members only by checking membership_rank
        query = """
            SELECT DISTINCT ud.user_id
            FROM user_data ud
            JOIN channels c ON ud.channel_id = c.channel_id
            WHERE c.channel_name = %s
              AND DATE_TRUNC('month', ud.last_message_at) = %s::DATE
              AND ud.membership_rank >= 0;
        """
    else:
        # This is the query for all users
        query = """
            SELECT DISTINCT mua.user_id
            FROM mv_user_monthly_activity mua
            JOIN channels c ON mua.channel_id = c.channel_id
            WHERE c.channel_name = %s
              AND mua.observed_month = %s::DATE;
        """

    user_sets = {}
    month_date = f"{month_str}-01"
    cursor = g.db_conn.cursor()
    
    try:
        for name in channel_names:
            cursor.execute(query, (name, month_date))
            user_sets[name] = {row[0] for row in cursor.fetchall()}
    except Exception as e:
        return jsonify({"error": f"Database query failed: {e}"}), 500
    finally:
        cursor.close()

    n = len(channel_names)
    matrix = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            set_i = user_sets[channel_names[i]]
            total_users_i = len(set_i)

            if i == j:
                matrix[i][j] = 100.0 if total_users_i > 0 else 0.0
                continue

            set_j = user_sets[channel_names[j]]
            
            if total_users_i > 0:
                common_users = len(set_i.intersection(set_j))
                percentage = (100.0 * common_users) / total_users_i
                matrix[i][j] = round(percentage, 2)
            else:
                matrix[i][j] = 0.0

    response_data = {
        "labels": channel_names,
        "matrix": matrix
    }
    
    g.redis_conn.set(redis_key, json.dumps(response_data), ex=86400)
    
    return jsonify(response_data)

@app.route('/api/get_common_members', methods=['GET'])
def get_common_members():
    """API to get common members between two (channel, month) pairs using set arithmetic."""
    channel_a_name = request.args.get('channel_a')
    month_a_str = request.args.get('month_a')  # YYYY-MM format
    channel_b_name = request.args.get('channel_b')
    month_b_str = request.args.get('month_b')  # YYYY-MM format

    if not (channel_a_name and month_a_str and channel_b_name and month_b_str):
        return jsonify({"error": _("Missing required parameters")}), 400
    
    redis_key = f"common_members_{channel_a_name}_{month_a_str}_{channel_b_name}_{month_b_str}"
    cached_data = g.redis_conn.get(redis_key)
    if cached_data:
        inc_cache_hit_count()
        return jsonify(json.loads(cached_data))
    inc_cache_miss_count()
    
    cursor = g.db_conn.cursor()
    
    # A simple, reusable query to get all member IDs for a given channel and month.
    # The key is the 'ud.membership_rank >= 0' filter.
    query = """
        SELECT DISTINCT ud.user_id
        FROM user_data ud
        JOIN channels c ON ud.channel_id = c.channel_id
        WHERE c.channel_name = %s
          AND DATE_TRUNC('month', ud.last_message_at) = %s::DATE
          AND ud.membership_rank >= 0;
    """
    
    try:
        # Fetch the list of members for the first channel/month
        month_a_date = f"{month_a_str}-01"
        cursor.execute(query, (channel_a_name, month_a_date))
        members_a = {row[0] for row in cursor.fetchall()}

        # Fetch the list of members for the second channel/month
        month_b_date = f"{month_b_str}-01"
        cursor.execute(query, (channel_b_name, month_b_date))
        members_b = {row[0] for row in cursor.fetchall()}
        
    except Exception as e:
        cursor.close()
        return jsonify({"error": f"Database query failed: {e}"}), 500
    finally:
        cursor.close()

    if not members_a or not members_b:
        # Return an empty object if one of the channels has no members for that month
        return jsonify({})

    # Perform the set arithmetic in Python, which is highly efficient
    common_members = members_a.intersection(members_b)
    
    num_common_members = len(common_members)
    total_members_a = len(members_a)
    total_members_b = len(members_b)

    # Calculate percentages
    percent_a_to_b = (100.0 * num_common_members / total_members_a) if total_members_a > 0 else 0
    percent_b_to_a = (100.0 * num_common_members / total_members_b) if total_members_b > 0 else 0

    data = {
        "channel_a": channel_a_name,
        "channel_b": channel_b_name,
        "month_a": month_a_str,
        "month_b": month_b_str,
        "num_common_members": num_common_members,
        "percent_A_to_B_members": round(percent_a_to_b, 2),
        "percent_B_to_A_members": round(percent_b_to_a, 2)
    }

    g.redis_conn.set(redis_key, json.dumps(data), ex=86400) # Cache for 24 hours
    
    return jsonify(data)



@app.route('/api/get_group_membership_data', methods=['GET'])
def get_group_membership_counts():
    cursor = g.db_conn.cursor()
    channel_group = request.args.get('channel_group')
    month = request.args.get('month')  # Expected format: YYYY-MM

    redis_key = f"group_membership_data_{channel_group}_{month}"
    cached_data = g.redis_conn.get(redis_key)
    if cached_data:
        inc_cache_hit_count()
        return jsonify(json.loads(cached_data))
    inc_cache_miss_count()

    if not channel_group or not month:
        return jsonify({"error": _("Missing required parameters")}), 400

    query = """SELECT channel_name, membership_rank, membership_count, percentage_total FROM mv_membership_data WHERE channel_group = %s AND observed_month = %s::DATE;"""

    cursor.execute(query, (channel_group, f"{month}-01"))
    results = cursor.fetchall()
    cursor.close()

    output = [
        [
            row[0],
            int(row[1]),
            float(row[2]),
            float(row[3])
        ] for row in results
    ]

    g.redis_conn.set(redis_key, json.dumps(output))

    return jsonify(output)

@app.route('/api/get_group_membership_changes', methods=['GET'])
def get_group_membership_changes():
    """API to fetch membership gains, losses, and differential by group and month."""
    cursor = g.db_conn.cursor()

    channel_group = request.args.get('channel_group')
    month = request.args.get('month')  # Expected format: YYYY-MM

    redis_key = f"group_membership_changes_{channel_group}_{month}"
    cached_data = g.redis_conn.get(redis_key)
    if cached_data:
        inc_cache_hit_count()
        return jsonify(json.loads(cached_data))
    inc_cache_miss_count()
    
    # Validate required parameters
    if not channel_group or not month:
        return jsonify({"error": _("Missing required parameters")}), 400

    # SQL query
    query = """
        WITH membership_changes AS (
            SELECT
                m.user_id,
                m.channel_id,
                DATE_TRUNC('month', m.last_message_at)::DATE AS observed_month,
                m.membership_rank,
                LAG(m.membership_rank) OVER (
                    PARTITION BY m.user_id, m.channel_id
                    ORDER BY m.last_message_at
                ) AS previous_membership_rank
            FROM user_data m
            JOIN channels c ON m.channel_id = c.channel_id
            WHERE c.channel_group = %s
              AND DATE_TRUNC('month', m.last_message_at) = %s::DATE
        ),
        gains AS (
            SELECT
                mc.user_id,
                mc.channel_id,
                mc.observed_month
            FROM membership_changes mc
            WHERE mc.previous_membership_rank = -1
              AND mc.membership_rank > -1
        ),
        expirations AS (
            SELECT
                mc.user_id,
                mc.channel_id,
                mc.observed_month
            FROM membership_changes mc
            WHERE mc.previous_membership_rank > -1
              AND mc.membership_rank = -1
        )
        SELECT
            c.channel_name,
            g.observed_month,
            COUNT(DISTINCT g.user_id) AS gains_count,
            COUNT(DISTINCT e.user_id) AS losses_count,
            (COUNT(DISTINCT g.user_id) - COUNT(DISTINCT e.user_id)) AS differential
        FROM channels c
        LEFT JOIN gains g ON g.channel_id = c.channel_id
        LEFT JOIN expirations e ON e.channel_id = c.channel_id
        WHERE g.observed_month = %s::DATE OR e.observed_month = %s::DATE
        GROUP BY c.channel_name, g.observed_month
        ORDER BY differential DESC;
    """
    
    try:
        # Execute the query with the provided parameters
        cursor.execute(query, (channel_group, f"{month}-01", f"{month}-01", f"{month}-01"))
        results = cursor.fetchall()
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        cursor.close()

    # Format the results as JSON
    response_data = [
        {
            "channel_name": row[0],
            "observed_month": row[1].strftime('%Y-%m') if row[1] else None,
            "gains_count": row[2],
            "losses_count": row[3],
            "differential": row[4]
        }
        for row in results if row[1] is not None
    ]

    g.redis_conn.set(redis_key, json.dumps(response_data))

    return jsonify(response_data)


@app.route('/api/get_group_streaming_hours_diff', methods=['GET'])
def get_group_streaming_hours_diff():
    """API to fetch the change in streaming hours since the previous month for a given group."""
    month = request.args.get('month')
    channel_group = request.args.get('group', None)

    redis_key = f"group_streaming_hours_diff_{channel_group or 'all'}_{month}"
    cached_data = g.redis_conn.get(redis_key)
    if cached_data:
        inc_cache_hit_count()
        return Response(cached_data, mimetype='application/json')
    inc_cache_miss_count()

    if not month:
        return jsonify({"success": False, "error": "Missing required parameter: month"}), 400

    try:
        month_date = datetime.strptime(month, "%Y-%m")
    except ValueError:
        return jsonify({"success": False, "error": "Invalid month format. Use YYYY-MM."}), 400

    params = []
    group_filter_sql = ""
    if channel_group:
        group_filter_sql = "WHERE c.channel_group = %s"
        params.append(channel_group)
    
    # Add the date parameter for the main WHERE clause
    params.append(f"{month_date.strftime('%Y-%m-01')}")

    query = f"""
        WITH monthly_streaming AS (
            SELECT
                c.channel_name,
                DATE_TRUNC('month', v.end_time AT TIME ZONE 'UTC') AS observed_month,
                SUM(EXTRACT(EPOCH FROM v.duration)) / 3600 AS total_streaming_hours
            FROM videos v
            JOIN channels c ON v.channel_id = c.channel_id
            {group_filter_sql}
            GROUP BY c.channel_name, observed_month
        )
        SELECT
            m1.channel_name,
            m1.observed_month,
            m1.total_streaming_hours,
            COALESCE((m1.total_streaming_hours - m2.total_streaming_hours), m1.total_streaming_hours) AS change_from_previous_month
        FROM monthly_streaming m1
        LEFT JOIN monthly_streaming m2
            ON m1.channel_name = m2.channel_name AND m1.observed_month = (m2.observed_month + INTERVAL '1 month')
        WHERE m1.observed_month = %s
        ORDER BY change_from_previous_month DESC;
    """

    cursor = g.db_conn.cursor()
    cursor.execute(query, tuple(params))
    results = cursor.fetchall()
    cursor.close()

    data = [
        {
            "channel": row[0],
            "month": row[1].strftime('%Y-%m'),
            "hours": float(round(row[2], 2)) if row[2] is not None else 0.0,
            "change": float(round(row[3], 2)) if row[3] is not None else 0.0
        }
        for row in results
    ]

    output = {"success": True, "data": data}
    
    # Cache the final JSON string that will be sent to the user
    g.redis_conn.set(redis_key, json.dumps(output))

    return jsonify(output)

@app.route('/api/get_chat_leaderboard', methods=['GET'])
def get_chat_leaderboard():
    """API to fetch the top 10 chatters for a given channel and month."""

    channel_name = request.args.get('channel_name')
    month = request.args.get('month')  # Expected format: YYYY-MM

    redis_key = f"chat_leaderboard_{channel_name}_{month}"
    cached_data = g.redis_conn.get(redis_key)
    if cached_data:
        inc_cache_hit_count()
        return jsonify(json.loads(cached_data))
    inc_cache_miss_count()

    if not channel_name or not month:
        return jsonify({"error": _("Missing required parameters")}), 400

    # Convert to first day of the month for DATE comparison
    month_start = f"{month}-01"

    query = """
        WITH chat_counts AS (
            SELECT 
                ud.user_id, 
                u.username AS user_name, 
                SUM(ud.total_message_count) AS message_count
            FROM user_data ud
            JOIN channels c ON ud.channel_id = c.channel_id
            JOIN users u ON ud.user_id = u.user_id
            WHERE c.channel_name = %s
              AND ud.last_message_at >= %s::DATE
              AND ud.last_message_at < (%s::DATE + INTERVAL '1 month')
            GROUP BY ud.user_id, u.username
        )
        SELECT user_name, message_count
        FROM chat_counts
        ORDER BY message_count DESC
        LIMIT 10;
    """

    cursor = g.db_conn.cursor()
    cursor.execute(query, (channel_name, month_start, month_start))
    
    results = cursor.fetchall()
    cursor.close()

    if not results:
        return jsonify({"error": _("No data found")}), 404
    
    output = [
        {
            "user_name": row[0],
            "message_count": row[1]
        }
        for row in results
    ]

    g.redis_conn.set(redis_key, json.dumps(output))

    return jsonify(output)

@app.route('/api/get_user_changes', methods=['GET'])
def get_user_changes():
    """API to fetch user gains and losses per channel in a given group and month."""

    channel_group = request.args.get('group')
    month = request.args.get('month')  # YYYY-MM format

    if not (channel_group and month):
        return jsonify({"error": _("Missing required parameters")}), 400
    
    redis_key = f"user_changes_{channel_group}_{month}"
    cached_data = g.redis_conn.get(redis_key)
    if cached_data:
        inc_cache_hit_count()
        return jsonify(json.loads(cached_data))
    inc_cache_miss_count()

    # Compute start dates
    current_month_start = f"{month}-01"
    previous_month_start = (datetime.strptime(month, "%Y-%m") - relativedelta(months=1)).strftime("%Y-%m-01")

    query = """
        WITH current_month_users AS (
        SELECT user_id, channel_id
        FROM mv_user_monthly_activity
        WHERE observed_month = %s::DATE AND monthly_message_count >= 5
        ),
        previous_month_users AS (
            SELECT user_id, channel_id
            FROM mv_user_monthly_activity
            WHERE observed_month = %s::DATE AND monthly_message_count >= 5
        ),
        channel_groups AS (
            SELECT channel_id, channel_group
            FROM channels
        )
        SELECT
            c.channel_name,
            SUM(CASE WHEN uma1.user_id IS NOT NULL THEN 1 ELSE 0 END) AS users_gained,
            SUM(CASE WHEN uma2.user_id IS NOT NULL THEN 1 ELSE 0 END) AS users_lost,
            SUM(CASE WHEN uma1.user_id IS NOT NULL THEN 1 ELSE 0 END) - 
            SUM(CASE WHEN uma2.user_id IS NOT NULL THEN 1 ELSE 0 END) AS net_change
        FROM channels c
        JOIN channel_groups cg ON c.channel_id = cg.channel_id
        LEFT JOIN current_month_users uma1 ON c.channel_id = uma1.channel_id
        LEFT JOIN previous_month_users uma2 ON c.channel_id = uma2.channel_id AND uma1.user_id = uma2.user_id
        WHERE cg.channel_group = %s
        GROUP BY c.channel_name
    """

    cursor = g.db_conn.cursor()
    cursor.execute(query, (current_month_start, previous_month_start, channel_group))
    results = cursor.fetchall()
    cursor.close()

    # Filter out channels where there is no data for one or both months
    filtered_results = [
        {
            "channel": row[0],
            "users_gained": row[1],
            "users_lost": row[2],
            "net_change": row[3]
        }
        for row in results
        if row[1] > 0 and row[2] > 0  # Exclude channels with no data for either month
    ]

    g.redis_conn.set(redis_key, json.dumps(filtered_results))

    return jsonify(filtered_results)

@app.route('/api/get_exclusive_chat_users', methods=['GET'])
def get_exclusive_chat_users():
    """API to fetch the percentage of exclusive chat users for a given channel per month."""

    channel_name = request.args.get('channel')

    if not channel_name:
        return jsonify({"error": _("Missing required parameters")}), 400
    
    redis_key = f"exclusive_chat_users_{channel_name}"
    cached_data = g.redis_conn.get(redis_key)
    if cached_data:
        inc_cache_hit_count()
        return jsonify(json.loads(cached_data))
    inc_cache_miss_count()
    
    cursor = g.db_conn.cursor()
    cursor.execute(
        "SELECT channel_id, channel_group FROM channels WHERE channel_name = %s", (channel_name,)
    )
    channel_info = cursor.fetchone()
    if not channel_info:
        return jsonify({"error": "Invalid channel"}), 400

    channel_id, channel_group = channel_info

    query = """
        WITH channel_specific_users AS (
            SELECT
                user_id,
                activity_month,
                channel_id
            FROM mv_user_activity
            WHERE channel_id = %s
        ),
        exclusive_users AS (
            SELECT
                csu.activity_month,
                COUNT(DISTINCT csu.user_id) AS exclusive_users_count
            FROM channel_specific_users csu
            LEFT JOIN mv_user_activity gu
                ON csu.user_id = gu.user_id
                AND gu.channel_id <> csu.channel_id
                AND gu.channel_group = %s
            WHERE NOT EXISTS (
                SELECT 1
                FROM mv_user_activity gu
                WHERE gu.user_id = csu.user_id
                AND gu.channel_group = %s
                AND gu.channel_id <> csu.channel_id
            )
            GROUP BY csu.activity_month
        ),
        total_users_per_month AS (
            SELECT
                activity_month,
                COUNT(DISTINCT user_id) AS total_users_count
            FROM channel_specific_users
            GROUP BY activity_month
        )
        SELECT
            tu.activity_month,
            ROUND((eu.exclusive_users_count::NUMERIC / tu.total_users_count) * 100, 2) AS exclusive_percent
        FROM total_users_per_month tu
        JOIN exclusive_users eu
            ON tu.activity_month = eu.activity_month
        ORDER BY tu.activity_month;
    """

    cursor.execute(query, (channel_id, channel_group, channel_group))
    results = cursor.fetchall()
    cursor.close()

    output = [
        {"month": row[0].strftime('%Y-%m'), "percent": float(row[1])}
        for row in results
    ]

    g.redis_conn.set(redis_key, json.dumps(output))

    return jsonify(output)

@app.route('/api/get_message_type_percents', methods=['GET'])
def get_message_type_percents():
    """API to fetch the percentage of a channel's monthly messages for a specified language
       and calculate message rate (language-specific messages/minute) based on video durations with chat logs."""

    # Get query parameters
    channel_name = request.args.get('channel')
    language = request.args.get('language').upper()

    # Validate parameters
    if not channel_name or not language:
        return jsonify({"error": _("Missing required parameters")}), 400

    if language not in ["EN", "JP", "KR", "RU"]:
        return jsonify({"error": _("Invalid language parameter. Must be one of: EN, JP, KR, RU.")}), 400
    
    redis_key = f"message_type_percents_{channel_name}_{language}"
    cached_data = g.redis_conn.get(redis_key)
    if cached_data:
        inc_cache_hit_count()
        return jsonify(json.loads(cached_data))
    inc_cache_miss_count()

    # Map language codes to database columns
    language_column_map = {
        "EN": "es_en_id_count",
        "JP": "jp_count",
        "KR": "kr_count",
        "RU": "ru_count"
    }

    # Get the appropriate column for the language
    language_column = language_column_map[language]

    query = f"""
        WITH monthly_data AS (
            SELECT
                DATE_TRUNC('month', ud.last_message_at) AS activity_month,
                SUM(ud.{language_column}) AS language_message_count,
                SUM(ud.total_message_count - ud.emoji_count) AS total_message_count
            FROM user_data ud
            JOIN channels c ON ud.channel_id = c.channel_id
            WHERE c.channel_name = %s
            GROUP BY activity_month
        ),
        video_durations AS (
            SELECT
                DATE_TRUNC('month', v.end_time) AS activity_month,
                SUM(EXTRACT(EPOCH FROM v.duration) / 60) AS total_minutes
            FROM videos v
            JOIN channels c ON v.channel_id = c.channel_id
            WHERE c.channel_name = %s AND v.has_chat_log = 't'
            GROUP BY activity_month
        )
        SELECT
            md.activity_month,
            ROUND((md.language_message_count::NUMERIC / NULLIF(md.total_message_count, 0)) * 100, 2) AS language_percent,
            ROUND(CAST(md.language_message_count::NUMERIC / NULLIF(vd.total_minutes, 0) AS NUMERIC), 2) AS language_message_rate
        FROM monthly_data md
        LEFT JOIN video_durations vd
        ON md.activity_month = vd.activity_month
        ORDER BY md.activity_month;
    """

    cursor = g.db_conn.cursor()
    cursor.execute(query, (channel_name, channel_name))
    results = cursor.fetchall()
    cursor.close()

    output = [
        {
            "month": row[0].strftime('%Y-%m'),
            "percent": float(row[1]),
            "message_rate": float(row[2])
        }
        for row in results
    ]

    g.redis_conn.set(redis_key, json.dumps(output))

    return jsonify(output)

@app.route('/api/get_attrition_rates', methods=['GET'])
def get_attrition_rates():
    """API to calculate attrition rates of a channel's top chatters.
    Returns percentage of top 1000 chatters (from last 3 months) who continue chatting in Hololive channels."""
    
    channel_name = request.args.get('channel')
    month = request.args.get('month')  # YYYY-MM format

    if not (channel_name and month):
        return jsonify({"error": "Missing required parameters"}), 400
    
    redis_key = f"attrition_rates_{channel_name}_{month}"
    cached_data = g.redis_conn.get(redis_key)
    if cached_data:
        inc_cache_hit_count()
        return jsonify(json.loads(cached_data))
    inc_cache_miss_count()

    try:
        # Calculate date ranges - include full given month plus previous 2 months
        end_date = datetime.strptime(month, "%Y-%m") + relativedelta(months=1)  # Start of next month
        start_date = end_date - relativedelta(months=3)  # 3 months back
        end_date_str = end_date.strftime("%Y-%m-01")
        start_date_str = start_date.strftime("%Y-%m-01")

        cursor = g.db_conn.cursor()

        # Get top 1000 users in the 3-month window
        cursor.execute("""
            WITH top_users AS (
                SELECT 
                    ud.user_id,
                    SUM(ud.total_message_count) AS total_messages
                FROM user_data ud
                JOIN channels c ON ud.channel_id = c.channel_id
                WHERE c.channel_name = %s
                  AND ud.last_message_at >= %s::DATE
                  AND ud.last_message_at < %s::DATE
                GROUP BY ud.user_id
                ORDER BY total_messages DESC
                LIMIT 1000
            )
            SELECT user_id FROM top_users
        """, (channel_name, start_date_str, end_date_str))
        
        top_users = [row[0] for row in cursor.fetchall()]
        if not top_users:
            return jsonify({"error": "No top chatters found for the given period"}), 404

        # For each subsequent month through current month, calculate percentage still active
        results = []
        current_month = end_date + relativedelta(months=1)
        today = datetime.utcnow()
        
        while current_month <= today:
            month_str = current_month.strftime("%Y-%m-01")
            
            cursor.execute("""
                SELECT COUNT(DISTINCT ud.user_id)
                FROM user_data ud
                JOIN channels c ON ud.channel_id = c.channel_id
                WHERE ud.user_id = ANY(%s)
                  AND c.channel_group = 'Hololive'
                  AND ud.last_message_at >= %s::DATE
                  AND ud.last_message_at < (%s::DATE + INTERVAL '1 month')
            """, (top_users, month_str, month_str))
            
            active_count = cursor.fetchone()[0] or 0
            percent_active = round((active_count / len(top_users)) * 100, 2)
            
            results.append({
                "month": current_month.strftime("%Y-%m"),
                "percent": percent_active
            })
            
            current_month = current_month + relativedelta(months=1)

        cursor.close()
        g.redis_conn.set(redis_key, json.dumps(results))
        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/get_jp_user_percent', methods=['GET'])
def get_jp_user_percent():
    """Returns the percentage of users using mostly Japanese (>50% of messages, excluding emoji-only messages) per month for a given channel."""
    channel_name = request.args.get('channel')

    if not channel_name:
        return jsonify({"error": "Missing required parameter: channel"}), 400
    
    redis_key = f"jp_user_percent_{channel_name}"
    cached_data = g.redis_conn.get(redis_key)
    if cached_data:
        inc_cache_hit_count()
        return jsonify(json.loads(cached_data))
    inc_cache_miss_count()

    query = """
        WITH user_language_usage AS (
            SELECT
                ud.user_id,
                DATE_TRUNC('month', ud.last_message_at) AS month,
                SUM(ud.jp_count) AS total_jp_messages,
                SUM(ud.total_message_count - ud.emoji_count) AS total_non_emoji_messages
            FROM user_data ud
            JOIN channels c ON ud.channel_id = c.channel_id
            WHERE c.channel_name = %s
            GROUP BY ud.user_id, month
        ),
        jp_users AS (
            SELECT
                month,
                COUNT(*) AS jp_user_count
            FROM user_language_usage
            WHERE total_jp_messages > total_non_emoji_messages * 0.5
            GROUP BY month
        ),
        total_users AS (
            SELECT
                DATE_TRUNC('month', last_message_at) AS month,
                COUNT(DISTINCT user_id) AS total_user_count
            FROM user_data ud
            JOIN channels c ON ud.channel_id = c.channel_id
            WHERE c.channel_name = %s
            GROUP BY month
        )
        SELECT 
            to_char(tu.month, 'YYYY-MM') AS month, 
            ROUND(100.0 * COALESCE(jp.jp_user_count, 0) / NULLIF(tu.total_user_count, 0), 2) AS jp_user_percent
        FROM total_users tu
        LEFT JOIN jp_users jp ON tu.month = jp.month
        ORDER BY month ASC;
    """

    cursor = g.db_conn.cursor()
    cursor.execute(query, (channel_name, channel_name))
    results = cursor.fetchall()
    cursor.close()

    # Format results for Chart.js
    response = [{"month": row[0], "jp_user_percent": float(row[1])} for row in results]

    g.redis_conn.set(redis_key, json.dumps(response))

    return jsonify(response)


@app.route('/api/get_latest_updates', methods=['GET'])
def get_latest_updates():
    """API to fetch the latest updates from news.txt for the front-end."""
    try:
        news_list = []
        with open("news.txt", "r") as file:
            for line in file:
                # Parse lines formatted as [date]: [text]
                if ": " in line:
                    date, message = line.split(": ", 1)
                    news_list.append({"date": date.strip(), "message": message.strip()})

        return jsonify(news_list)
    except FileNotFoundError:
        return jsonify([])  # Return an empty list if news.txt is not found
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/get_channel_names', methods=['GET'])
def get_channel_names():

    redis_key = "channel_names"
    cached_data = g.redis_conn.get(redis_key)
    if cached_data:
        inc_cache_hit_count()
        return jsonify(json.loads(cached_data))
    inc_cache_miss_count()
    
    cursor = g.db_conn.cursor()
    cursor.execute("SELECT channel_name FROM channels ORDER BY channel_name")
    channel_names = [row[0] for row in cursor.fetchall()]
    cursor.close()
    g.redis_conn.set(redis_key, json.dumps(channel_names))
    return jsonify(channel_names)

@app.route('/api/get_date_ranges', methods=['GET'])
def get_date_ranges():
    redis_key = "date_ranges"
    cached_data = g.redis_conn.get(redis_key)
    if cached_data:
        inc_cache_hit_count()
        return jsonify(json.loads(cached_data))
    inc_cache_miss_count()
    
    cursor = g.db_conn.cursor()

    # Gets first and last date from video table using end_time
    cursor.execute("SELECT MIN(end_time), MAX(end_time) FROM videos WHERE has_chat_log = 't'")
    
    date_range = cursor.fetchone()
    cursor.close()

    output = [
        str(date_range[0]),
        str(date_range[1])
    ]
    
    g.redis_conn.set(redis_key, json.dumps(output))
    return jsonify(output)

@app.route('/api/get_number_of_chat_logs', methods=['GET'])
def get_number_of_chat_logs():
    redis_key = "number_of_chat_logs"
    cached_data = g.redis_conn.get(redis_key)
    if cached_data:
        inc_cache_hit_count()
        return jsonify(json.loads(cached_data))
    inc_cache_miss_count()
    cursor = g.db_conn.cursor()
    cursor.execute("SELECT COUNT(*) from videos WHERE has_chat_log = 't'")
    num_chat_logs = cursor.fetchone()[0]
    cursor.close()
    g.redis_conn.set(redis_key, json.dumps(num_chat_logs))
    return jsonify(num_chat_logs)

@app.route('/api/get_num_messages', methods=['GET'])
def get_num_messages():
    redis_key = "num_messages"
    cached_data = g.redis_conn.get(redis_key)
    if cached_data:
        inc_cache_hit_count()
        return jsonify(json.loads(cached_data))
    inc_cache_miss_count()
    cursor = g.db_conn.cursor()
    cursor.execute("SELECT SUM(total_message_count) FROM user_data")
    num_messages = cursor.fetchone()[0]
    cursor.close()
    g.redis_conn.set(redis_key, json.dumps(num_messages))
    return jsonify(num_messages)

@app.route('/api/get_funniest_timestamps', methods=['GET'])
def get_funniest_timestamps():
    """API to fetch funniest timestamps using actual last chat message timestamp."""

    channel_name = request.args.get('channel')
    month = request.args.get('month')  # Expected format: YYYY-MM

    if not (channel_name and month):
        return jsonify({"error": "Missing required parameters"}), 400
    
    redis_key = f"funniest_timestamps_{channel_name}_{month}"
    cached_data = g.redis_conn.get(redis_key)
    if cached_data:
        inc_cache_hit_count()
        return jsonify(json.loads(cached_data))
    inc_cache_miss_count()

    # Convert month to a valid date range
    month_start = f"{month}-01"
    next_month_start = (datetime.strptime(month, "%Y-%m") + relativedelta(months=1)).strftime("%Y-%m-01")

    query = """
        WITH last_chat AS (
            SELECT 
                ud.video_id, 
                MAX(ud.last_message_at) AS last_message_at
            FROM user_data ud
            JOIN channels c ON ud.channel_id = c.channel_id
            WHERE c.channel_name = %s
              AND ud.last_message_at >= %s::DATE
              AND ud.last_message_at < %s::DATE
            GROUP BY ud.video_id
        )
        SELECT 
            v.title,
            v.video_id, 
            EXTRACT(EPOCH FROM (TO_TIMESTAMP(v.funniest_timestamp) - lc.last_message_at + v.duration)) AS relative_timestamp
        FROM videos v
        JOIN channels c ON v.channel_id = c.channel_id
        JOIN last_chat lc ON v.video_id = lc.video_id
        WHERE c.channel_name = %s 
          AND v.funniest_timestamp IS NOT NULL
        ORDER BY v.end_time ASC;
    """

    cursor = g.db_conn.cursor()
    cursor.execute(query, (channel_name, month_start, next_month_start, channel_name))
    results = cursor.fetchall()
    cursor.close()

    output = [
        {
            "title": row[0],
            "video_id": row[1],
            "timestamp": int(row[2])
        }
        for row in results if row[1] is not None and row[2] is not None
    ]

    g.redis_conn.set(redis_key, json.dumps(output))

    return jsonify(output)

@app.route('/api/get_user_info', methods=['GET'])
def get_user_info():
    """
    API to fetch all channels a user chatted on in a given month,
    their total messages on that channel, and their frequency percentile,
    using their unique user_id.
    """
    
    user_id = request.args.get('identifier')
    month = request.args.get('month')  # YYYY-MM format

    if not (user_id and month):
        return jsonify({"success": False, "error": "Missing required parameters: 'user_id' and 'month' are required."}), 400
    
    redis_key = f"user_info_{user_id}_{month}"
    cached_data = g.redis_conn.get(redis_key)
    if cached_data:
        inc_cache_hit_count()
        return jsonify(json.loads(cached_data))
    inc_cache_miss_count()

    # If user_id starts with '@', lookup user_id from database
    if user_id.startswith('@'):
        cursor = g.db_conn.cursor()
        cursor.execute("SELECT user_id FROM users WHERE username = %s", (user_id,))
        result = cursor.fetchone()
        cursor.close()
        if result:
            user_id = result[0]
        else:
            return jsonify({"success": False, "error": "User not found."}), 404

    month_start = f"{month}-01"
    next_month_start = (datetime.strptime(month, "%Y-%m") + relativedelta(months=1)).strftime("%Y-%m-01")

    cursor = g.db_conn.cursor()

    # Fetch user's chat activity per channel for the month
    cursor.execute("""
        SELECT 
            ud.channel_id, 
            c.channel_name, 
            SUM(ud.total_message_count) AS user_message_count
        FROM user_data ud
        JOIN channels c ON ud.channel_id = c.channel_id
        WHERE ud.user_id = %s
          AND ud.last_message_at >= %s::DATE
          AND ud.last_message_at < %s::DATE
        GROUP BY ud.channel_id, c.channel_name
    """, (user_id, month_start, next_month_start))
    
    user_chat_data = cursor.fetchall()

    if not user_chat_data:
        cursor.close()
        output = {"success": True, "data": []}
        g.redis_conn.set(redis_key, json.dumps(output)) 
        return jsonify(output)

    results = []

    for channel_id, channel_name, user_message_count in user_chat_data:
        cursor.execute("""
            WITH all_user_counts AS (
                SELECT user_id, SUM(total_message_count) AS total_messages
                FROM user_data
                WHERE channel_id = %s
                  AND last_message_at >= %s::DATE
                  AND last_message_at < %s::DATE
                GROUP BY user_id
            )
            SELECT 
                100.0 * (SELECT COUNT(*) FROM all_user_counts WHERE total_messages <= %s) 
                / NULLIF((SELECT COUNT(*) FROM all_user_counts), 0) AS percentile
        """, (channel_id, month_start, next_month_start, user_message_count))

        percentile_row = cursor.fetchone()
        percentile = percentile_row[0] if percentile_row and percentile_row[0] is not None else 0.0

        results.append({
            "channel_name": channel_name,
            "message_count": int(user_message_count),
            "percentile": round(float(percentile), 2)
        })

    cursor.close()
    
    output = {"success": True, "data": results}
    g.redis_conn.set(redis_key, json.dumps(output)) # Cache for 24 hours

    return jsonify(output)

@app.route('/api/get_chat_engagement', methods=['GET'])
def get_chat_engagement():
    """API to fetch chat engagement statistics per channel, with optional filtering."""
    month = request.args.get('month', datetime.utcnow().strftime('%Y-%m'))  # Default: current month
    group = request.args.get('group', None)  # "Hololive", "Indie", or None

    redis_key = f"chat_engagement_{month}_{group}"
    cached_data = g.redis_conn.get(redis_key)
    if cached_data:
        inc_cache_hit_count()
        return jsonify(json.loads(cached_data))
    inc_cache_miss_count()

    # Convert 'YYYY-MM' to 'YYYY-MM-01'
    start_month = datetime.strptime(month + "-01", '%Y-%m-%d').strftime('%Y-%m-01')

    query = """
        WITH chat_engagement AS (
            SELECT 
                ud.channel_id, 
                COUNT(DISTINCT ud.user_id) AS total_users,
                SUM(ud.total_message_count) AS total_messages
            FROM user_data ud
            JOIN channels c ON ud.channel_id = c.channel_id
            WHERE 
                DATE_TRUNC('month', ud.last_message_at) = %s
                AND (%s IS NULL OR c.channel_group = %s)
            GROUP BY ud.channel_id
        )
        SELECT 
            c.channel_name,
            ce.total_users,
            ce.total_messages,
            ROUND(ce.total_messages::DECIMAL / NULLIF(ce.total_users, 0), 2) AS avg_messages_per_user
        FROM chat_engagement ce
        JOIN channels c ON ce.channel_id = c.channel_id
        ORDER BY avg_messages_per_user DESC;
    """

    try:
        with g.db_conn.cursor() as cur:
            cur.execute(query, (start_month, group, group))
            results = cur.fetchall()
        data = [
            {"channel": row[0], "total_users": int(row[1]), "total_messages": int(row[2]), "avg_messages_per_user": float(row[3])}
            for row in results if row[3] is not None
        ]
        output = {"success": True, "data": data}
        g.redis_conn.set(redis_key, json.dumps(output))
        return jsonify(output)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/get_video_highlights', methods=['GET'])
def get_video_highlights():
    """
    API endpoint to fetch AI-generated highlights for a specific channel and month.
    The results are cached in Redis to improve performance.
    """
    # --- 1. Get and Validate Input Parameters ---
    channel_name = request.args.get('channel_name')
    month_str = request.args.get('month') # Expected format: "YYYY-MM"

    if not channel_name or not month_str:
        return jsonify({"success": False, "error": "Missing required parameters: 'channel_name' and 'month' are required."}), 400

    try:
        # Validate the month format and prepare for SQL query
        month_start_dt = datetime.strptime(month_str, '%Y-%m')
        month_start_sql = month_start_dt.strftime('%Y-%m-01')
    except ValueError:
        return jsonify({"success": False, "error": "Invalid month format. Please use 'YYYY-MM'."}), 400

    # --- 2. Check Redis Cache ---
    redis_key = f"video_highlights_{channel_name.replace(' ', '_')}_{month_str}"
    try:
        cached_data = g.redis_conn.get(redis_key)
        if cached_data:
             inc_cache_hit_count()
             return jsonify(json.loads(cached_data))
        inc_cache_miss_count()
    except Exception as e:
        # Log the error but proceed to query DB; cache is not critical.
        print(f"Redis cache check failed: {e}")


    # --- 3. Query the Database (if cache miss) ---
    # This query joins the three tables to get all the necessary information.
    query = """
        SELECT
            vh.video_id,
            v.title,
            vh.topic_tag,
            vh.generated_summary,
            -- Calculate the video's true start time (end_time - duration)
            -- Then find the difference between the highlight's time and the start time
            -- to get the elapsed seconds for the URL.
            EXTRACT(EPOCH FROM (TO_TIMESTAMP(vh.start_seconds) - (v.end_time - v.duration))) AS relative_seconds
        FROM
            video_highlights vh
        JOIN
            videos v ON vh.video_id = v.video_id
        JOIN
            channels c ON v.channel_id = c.channel_id
        WHERE
            c.channel_name = %s
            AND DATE_TRUNC('month', v.end_time) = %s
        ORDER BY
            v.end_time DESC, vh.start_seconds ASC;
    """

    try:
        with g.db_conn.cursor() as cur:
            cur.execute(query, (channel_name, month_start_sql))
            results = cur.fetchall()

        if not results:
            # Still cache the empty result to prevent repeated queries for months with no data
            output = {"success": True, "data": []}
            #g.redis_conn.set(redis_key, json.dumps(output), ex=3600) # Cache empty result for 1 hour
            return jsonify(output)

        # --- 4. Process and Structure the Data ---
        # Group the flat list of highlights by video_id
        videos_dict = {}
        for row in results:
            video_id, title, topic, summary, relative_seconds = row
            # Ensure timestamp is not negative or past the end of the video
            # (This is a safety check)
            if relative_seconds < 0: continue
            
            if video_id not in videos_dict:
                videos_dict[video_id] = { "video_id": video_id, "video_title": title, "timestamps": [] }
            
            videos_dict[video_id]["timestamps"].append({
                "topic": topic,
                "summary": summary,
                "youtube_url": f"https://www.youtube.com/watch?v={video_id}&t={int(relative_seconds)}s"
            })

        # Convert the dictionary of videos into a list for the final JSON
        data = list(videos_dict.values())
        output = {"success": True, "data": data}

        # --- 5. Store Result in Redis Cache ---
        #g.redis_conn.set(redis_key, json.dumps(output))

        return jsonify(output)

    except Exception as e:
        # Log the full error for debugging
        print(f"An error occurred in get_video_highlights: {e}")
        return jsonify({"success": False, "error": "An internal server error occurred."}), 500
    
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
    
    # Pattern to find channel:"quoted name" or channel:unquotedname
    channel_pattern = r'channel:"([^"]+)"|channel:(\S+)'
    from_pattern = r'from:(\d{4}-\d{2}-\d{2})'
    to_pattern = r'to:(\d{4}-\d{2}-\d{2})'
    
    # --- Extract Channel ---
    channel_match = re.search(channel_pattern, raw_query)
    if channel_match:
        # The first group is for quoted, the second for unquoted
        filters["channel_name"] = channel_match.group(1) or channel_match.group(2)
        raw_query = raw_query[:channel_match.start()] + raw_query[channel_match.end():]

    # --- Extract From Date ---
    from_match = re.search(from_pattern, raw_query)
    if from_match:
        try:
            datetime.strptime(from_match.group(1), '%Y-%m-%d')
            filters["from_date"] = from_match.group(1)
            raw_query = raw_query[:from_match.start()] + raw_query[from_match.end():]
        except ValueError:
            return None, None, f"Invalid 'from' date format: {from_match.group(1)}. Use YYYY-MM-DD."

    # --- Extract To Date ---
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
    
@app.route('/api/search_highlights', methods=['GET'])
def search_highlights():
    """
    API endpoint to search for highlights using vector similarity and search operators.
    Operators: channel:<name>, from:<YYYY-MM-DD>, to:<YYYY-MM-DD>
    """
    raw_query = request.args.get('query')
    if not raw_query:
        return jsonify({"success": False, "error": "Missing required parameter: 'query' is required."}), 400

    clean_query, filters, error = parse_search_query(raw_query)
    if error:
        return jsonify({"success": False, "error": error}), 400
    
    if not clean_query:
        return jsonify({"success": False, "error": "Search query cannot be empty after removing operators."}), 400

    try:
        query_vector = EMBEDDER.encode(clean_query).tolist()

        sql_select = """
        SELECT
            vh.generated_summary, vh.topic_tag, vh.video_id,
            v.title, v.end_time, v.duration, vh.start_seconds, 
            vh.summary_embedding <=> %s AS distance
        """
        params = [json.dumps(query_vector)]
        sql_from = "FROM video_highlights vh JOIN videos v ON vh.video_id = v.video_id"
        where_clauses = []

        if filters["channel_name"]:
            sql_from += " JOIN channels c ON v.channel_id = c.channel_id"
            where_clauses.append("c.channel_name ILIKE %s")
            params.append(f"%{filters['channel_name']}%")

        if filters["from_date"]:
            where_clauses.append("v.end_time >= %s")
            params.append(filters["from_date"])

        if filters["to_date"]:
            where_clauses.append("v.end_time < (%s::date + interval '1 day')")
            params.append(filters["to_date"])
        
        sql_where = ""
        if where_clauses:
            sql_where = "WHERE " + " AND ".join(where_clauses)

        sql_order_limit = "ORDER BY distance ASC LIMIT 10;"
        
        full_query = " ".join([sql_select, sql_from, sql_where, sql_order_limit])

        with g.db_conn.cursor() as cur:
            cur.execute(full_query, tuple(params))
            results = cur.fetchall()

        data = []
        for row in results:
            summary, topic, video_id, title, end_time, duration, start_seconds, distance = row
            
            # --- NEW: Calculate relative_seconds in Python for simplicity ---
            # (Alternatively, you can use the same SQL EXTRACT logic as above)
            video_start_time = end_time - duration
            highlight_time = datetime.fromtimestamp(start_seconds, tz=timezone.utc)
            relative_seconds = (highlight_time - video_start_time).total_seconds()

            if relative_seconds < 0: continue

            data.append({
                "summary": summary, "topic": topic, "video_id": video_id,
                "video_title": title, "date": end_time.strftime('%Y-%m-%d'),
                "youtube_url": f"https://www.youtube.com/watch?v={video_id}&t={int(relative_seconds)}s",
                "similarity_score": 1 - distance
            })

        output = {"success": True, "data": data}
        return jsonify(output)

    except Exception as e:
        print(f"An error occurred in search_highlights: {e}")
        return jsonify({"success": False, "error": "An internal server error occurred."}), 500


@app.route('/streaming_hours')
def streaming_hours_view():
    return render_template('streaming_hours.html', _=_, get_locale=get_locale)

@app.route('/streaming_hours_avg')
def streaming_hours_avg_view():
    return render_template('streaming_hours_avg.html', _=_, get_locale=get_locale)

@app.route('/streaming_hours_max')
def streaming_hours_max_view():
    return render_template('streaming_hours_max.html', _=_, get_locale=get_locale)

@app.route('/streaming_hours_diff')
def streaming_hours_diff_view():
    return render_template('streaming_hours_diff.html', _=_, get_locale=get_locale)

@app.route('/chat_makeup')
def chat_makeup_view():
    return render_template('chat_makeup.html', _=_, get_locale=get_locale)

@app.route('/common_users')
def common_users_view():
    return render_template('common_users.html', _=_, get_locale=get_locale)

@app.route('/common_user_heatmap')
def common_user_heatmap_view():
    return render_template('common_user_heatmap.html', _=_, get_locale=get_locale)

@app.route('/membership_counts')
def membership_counts_view():
    return render_template('membership_counts.html', _=_, get_locale=get_locale)

@app.route('/membership_percentages')
def membership_percentages_view():
    return render_template('membership_percentages.html', _=_, get_locale=get_locale)

@app.route('/membership_change')
def membership_expirations_view():
    return render_template('membership_change.html', _=_, get_locale=get_locale)

@app.route('/chat_leaderboards')
def chat_leaderboard_view():
    return render_template('chat_leaderboards.html', _=_, get_locale=get_locale)

@app.route('/user_change')
def user_changes_view():
    return render_template('user_change.html', _=_, get_locale=get_locale)

@app.route('/monthly_streaming_hours')
def monthly_streaming_hours_view():
    return render_template('monthly_streaming_hours.html', _=_, get_locale=get_locale)

@app.route('/exclusive_chat')
def exclusive_chat_users_view():
    return render_template('exclusive_chat.html', _=_, get_locale=get_locale)

@app.route('/message_types')
def message_types_view():
    return render_template('message_types.html', _=_, get_locale=get_locale)

@app.route('/funniest_timestamps')
def funniest_timestamps_view():
    return render_template('funniest_timestamps.html', _=_, get_locale=get_locale)

@app.route('/common_members')
def common_members_view():
    return render_template('common_members.html', _=_, get_locale=get_locale)

@app.route('/channel_clustering')
def channel_clustering_view():
    return render_template('channel_clustering.html', _=_, get_locale=get_locale)

@app.route('/jp_user_percents')
def jp_user_percents_view():
    return render_template('jp_user_percents.html', _=_, get_locale=get_locale)

@app.route('/user_info')
def user_info_view():
    return render_template('user_info.html', _=_, get_locale=get_locale)

@app.route('/engagement')
def engagement_redirect():
    return render_template('engagement.html', _=_, get_locale=get_locale)

@app.route('/site_metrics')
def site_metrics_view():
    return render_template('site_metrics.html', _=_, get_locale=get_locale)

@app.route('/recommendation_engine')
def recommendation_engine_view():
    return render_template('recommendation_engine.html', _=_, get_locale=get_locale)

@app.route('/highlights')
def highlights_view():
    return render_template('highlights.html', _=_, get_locale=get_locale)

@app.route('/highlight_search')
def highlight_search_view():
    return render_template('highlight_search.html', _=_, get_locale=get_locale)

@app.route('/heatmap')
def heatmap_view():
    return render_template('heatmap.html', _=_, get_locale=get_locale)

#v1 redirects
@app.route('/stream-time')
def stream_time_redirect():
    return redirect("/streaming_hours")

@app.route('/nonjp-holojp')
def nonjp_holojp_redirect():
    return redirect("/message_types")

@app.route('/jp-holoiden')
def holoiden_redirect():
    return redirect("/message_types")

@app.route('/chat-makeup')
def chat_makeup_redirect():
    return redirect("/chat_makeup")

@app.route('/langsum')
def langsum_redirect():
    return redirect("/message_types")

@app.route('/en-livetl')
def livetl_redirect():
    return redirect("https://old.holochatstats.info/en-livetl")

@app.route('/en-tl-stream')
def en_tl_stream_redirect():
    return redirect("https://old.holochatstats.info/en-tl-stream")

@app.route('/common-chat')
def common_chat_redirect():
    return redirect("/common_users")

@app.route('/excl-chat')
def excl_chat_redirect():
    return redirect("/exclusive_chat")

@app.route('/members')
def members_redirect():
    return redirect("/membership_counts")

@app.route('/member-percent')
def member_percent_redirect():
    return redirect("/membership_percentages")

@app.route('/stream-time-series')
def stream_time_series_redirect():
    return redirect("/monthly_streaming_hours")

@app.route('/coverage')
def coverage_redirect():
    return redirect("https://old.holochatstats.info/coverage")

@app.route('/robots.txt')
def robots_txt():
    lines = [
        "User-agent: GPTBot",
        "Disallow: /",
        "User-agent: ClaudeBot",
        "Disallow: /",
        "User-agent: ChatGPT-User",
        "Disallow: /",
        "User-agent: Amazonbot",
        "Disallow: /",
        "User-agent: CCBot",
        "Disallow: /",
        "User-agent: *",
        "Allow: /"
    ]
    return Response("\n".join(lines), mimetype="text/plain")

if __name__ == '__main__':
    app.run(debug=True)
