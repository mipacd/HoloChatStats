from flask import Flask, request, jsonify, render_template, session, redirect, Response, url_for
from flask_babel import Babel, _
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_caching import Cache
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import configparser
import os
import psycopg2
import logging
import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import community.community_louvain as community_louvain
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
import json
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import RunReportRequest, DateRange, Metric

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
# Setup session key and babel configuration
app.config["SECRET_KEY"] = get_config("Settings", "SecretKey")
app.config["SESSION_TYPE"] = "filesystem"
app.config["BABEL_DEFAULT_LOCALE"] = "en"
app.config["BABEL_TRANSLATION_DIRECTORIES"] = "translations"
app.config["JSON_AS_ASCII"] = False
app.config["GA_ID"] = get_config("Settings", "GoogleAnalyticsID")

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = get_config("API", "GAAPIKeyFile")

app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)

cache = Cache(app, config={"CACHE_TYPE": "filesystem", "CACHE_DIR": get_config("Settings", "CacheDir")})

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

# Setup logging
def setup_logging():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    app.logger.addHandler(handler)

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

@app.route('/api/visitor_count')
@cache.cached(timeout=3600)
def visitor_count():
    count = get_google_analytics_visitors()
    return jsonify({'count': count})

# Access logging and detect language
@app.before_request
def before_request():
    real_ip = request.headers.get("CF-Connecting-IP", request.remote_addr)
    app.logger.info(f"Request from {real_ip} to {request.path}")
    if 'language' not in session:
        user_lang = request.headers.get('Accept-Language', 'en').split(',')[0][:2]
        session['language'] = user_lang if user_lang in ['en', 'ja', 'ko'] else 'en'

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

def streaming_hours_query(aggregation_function, group=None):
    group = request.args.get('group', None)
    month = request.args.get('month', datetime.utcnow().strftime('%Y-%m'))
    timezone_offset = int(request.args.get('timezone', 0))
    month_start = f"{month}-01"

    base_query = f"""
        SELECT
            c.channel_name,
            DATE_TRUNC('month', v.end_time AT TIME ZONE 'UTC' + INTERVAL %s) AS month,
            {aggregation_function}(EXTRACT(EPOCH FROM v.duration)) / 3600 AS hours
        FROM videos v
        JOIN channels c ON v.channel_id = c.channel_id
        WHERE DATE_TRUNC('month', v.end_time AT TIME ZONE 'UTC' + INTERVAL %s) = %s::DATE
    """

    params = [f"{timezone_offset} hour", f"{timezone_offset} hour", month_start]

    if group and group != "All":
            base_query += " AND c.channel_group = %s"
            params.append(group)

    base_query += " GROUP BY c.channel_name, month ORDER BY hours DESC"

    return base_query, params

@app.route('/api/channel_clustering', methods=['GET'])
@cache.cached(timeout=86400, query_string=True)
def channel_clustering():
    try:
        filter_month = request.args.get("month")
        if not filter_month:
            return jsonify({"error": "Month filter (e.g., '2025-03') is required"}), 400

        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT ud.user_id, ch.channel_name, SUM(ud.total_message_count) AS message_weight
            FROM user_data ud
            JOIN channels ch ON ud.channel_id = ch.channel_id
            WHERE DATE_TRUNC('month', ud.last_message_at) = %s::DATE
            GROUP BY ud.user_id, ch.channel_name;
        """, (filter_month + "-01",))
        rows = cursor.fetchall()

        cursor.close()
        conn.close()

        data = pd.DataFrame(rows, columns=['user_id', 'channel_name', 'message_weight'])
        if data.empty:
            return jsonify({"error": "No data found for the specified month"}), 404

        user_channel_matrix = data.pivot(index='user_id', columns='channel_name', values='message_weight').fillna(0)
        similarity_matrix = cosine_similarity(user_channel_matrix.T)
        channel_names = user_channel_matrix.columns

        G = nx.Graph()
        threshold = 0.09  
        for i, channel_a in enumerate(channel_names):
            for j, channel_b in enumerate(channel_names):
                if i != j and similarity_matrix[i, j] > threshold:
                    G.add_edge(channel_a, channel_b, weight=similarity_matrix[i, j])

        partition = community_louvain.best_partition(G)
        community_colors = [partition[node] for node in G.nodes]

        # --- IMPROVED LAYOUT ---
        pos = nx.kamada_kawai_layout(G)  # Better spacing
        node_x, node_y = zip(*pos.values())

        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        fig = go.Figure()

        # --- EDGE VISUAL IMPROVEMENTS ---
        fig.add_trace(go.Scatter(
            x=edge_x,
            y=edge_y,
            mode='lines',
            line=dict(width=0.5, color='rgba(150,150,150,0.5)'),  # Transparent gray edges
            hoverinfo='none'
        ))

        # --- NODE VISUAL IMPROVEMENTS ---
        fig.add_trace(go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            text=list(G.nodes),
            textposition="top center",
            marker=dict(
                size=12,  # Larger nodes
                color=community_colors,
                colorscale='Viridis',  # Better contrast
                line=dict(color='black', width=1)  # Node border for clarity
            )
        ))

        fig.update_layout(
            title=f"Channel User Similarity Graph for {filter_month}",
            title_x=0.5,
            showlegend=False,
            margin=dict(l=10, r=10, t=50, b=10),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            plot_bgcolor='white'
        )

        graph_json = json.dumps(fig, cls=PlotlyJSONEncoder)

        return jsonify({"graph_json": graph_json})

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/api/get_monthly_streaming_hours', methods=['GET'])
@cache.cached(timeout=86400, query_string=True)
def get_monthly_streaming_hours():
    """API to fetch total streaming hours per month for a given channel."""

    channel_name = request.args.get('channel')

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

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(query, (channel_name,))
    results = cursor.fetchall()
    cursor.close()
    conn.close()

    # Convert results into JSON format
    return jsonify([{"month": row[0].strftime('%Y-%m'), "total_streaming_hours": row[1]} for row in results])


@app.route('/api/get_group_total_streaming_hours', methods=['GET'])
@cache.cached(timeout=86400, query_string=True)
def get_group_total_streaming_hours():
    query, params = streaming_hours_query('SUM')
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(query, params)
            results = cur.fetchall()
        data = [{"channel": row[0], "month": row[1].strftime('%Y-%m'), "hours": round(row[2], 2)} for row in results]
        return jsonify({"success": True, "data": data})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/get_group_avg_streaming_hours', methods=['GET'])
@cache.cached(timeout=86400, query_string=True)
def get_group_avg_streaming_hours():
    query, params = streaming_hours_query('AVG')
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(query, params)
            results = cur.fetchall()
        data = [{"channel": row[0], "month": row[1].strftime('%Y-%m'), "hours": round(row[2], 2)} for row in results]
        return jsonify({"success": True, "data": data})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/get_group_max_streaming_hours', methods=['GET'])
@cache.cached(timeout=86400, query_string=True)
def get_group_max_streaming_hours():
    query, params = streaming_hours_query('MAX')
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(query, params)
            results = cur.fetchall()
        data = [{"channel": row[0], "month": row[1].strftime('%Y-%m'), "hours": round(row[2], 2)} for row in results]
        return jsonify({"success": True, "data": data})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    
@app.route('/api/get_group_chat_makeup', methods=['GET'])
@cache.cached(timeout=86400, query_string=True)
def get_group_chat_makeup():
    """API to fetch chat makeup statistics per channel, with optional filtering."""
    group = request.args.get('group', None)  # "Hololive", "Indie", or None
    month = request.args.get('month', datetime.utcnow().strftime('%Y-%m'))  # Default: current month
    timezone_offset = int(request.args.get('timezone', 0))  # Default: UTC (0)

    # Convert 'YYYY-MM' to 'YYYY-MM-01' and the end of that month ('YYYY-MM-01')
    start_month = datetime.strptime(month + "-01", '%Y-%m-%d').strftime('%Y-%m-01')

    # Calculate time zone offset for INTERVAL
    timezone_interval = f"{timezone_offset} hours"

    # SQL query to calculate rates
    query = """
        WITH streaming_time AS (
            SELECT 
                v.channel_id,
                DATE_TRUNC('month', v.end_time AT TIME ZONE 'UTC' + INTERVAL %s) AS observed_month,
                SUM(EXTRACT(EPOCH FROM v.duration)) / 60 AS total_streaming_minutes
            FROM videos v
            WHERE duration IS NOT NULL 
            AND duration > INTERVAL '0 seconds'
            AND has_chat_log = 't'
            AND DATE_TRUNC('month', v.end_time AT TIME ZONE 'UTC' + INTERVAL %s) = %s::DATE
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
        FROM chat_language_stats cls
        JOIN channels c ON cls.channel_id = c.channel_id
        JOIN streaming_time st 
            ON st.channel_id = c.channel_id 
            AND CAST(st.observed_month AS DATE) = CAST(cls.observed_month AS DATE)
    """

    params = [timezone_interval, timezone_interval, start_month]

    # Group filtering
    if group:
        if group == "Indie":
            query += " WHERE c.channel_group IS NULL"
        else:
            query += " WHERE c.channel_group = %s"
            params.append(group)
    else:  # Default to All
        query += " WHERE (c.channel_group = 'Hololive' OR c.channel_group IS NULL)"

    query += " GROUP BY c.channel_name, st.observed_month ORDER BY SUM(st.total_streaming_minutes) DESC"

    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(query, params)
        results = cur.fetchall()
        cur.close()
        conn.close()

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

        return jsonify({"success": True, "data": data})

    except Exception as e:
        print(f"Error: {e}")  # Print error message for debugging
        return jsonify({"success": False, "error": str(e)}), 500
    
@app.route('/api/get_common_users', methods=['GET'])
@cache.cached(timeout=86400, query_string=True)
def get_common_users():
    """API to get common users between two (channel, month) pairs."""
    channel_a = request.args.get('channel_a')
    month_a = request.args.get('month_a')  # YYYY-MM format
    channel_b = request.args.get('channel_b')
    month_b = request.args.get('month_b')  # YYYY-MM format

    if not (channel_a and month_a and channel_b and month_b):
        return jsonify({"error": _("Missing required parameters")}), 400

    month_a += "-01"
    month_b += "-01"

    query = """
        SELECT 
            ca.channel_name AS channel_a,
            cb.channel_name AS channel_b,
            ua.observed_month AS month_a,
            ub.observed_month AS month_b,
            COUNT(DISTINCT ua.user_id) AS num_common_users,
            100.0 * COUNT(DISTINCT ua.user_id) / NULLIF(ua_count.total_users, 0) AS percent_A_to_B_users,
            100.0 * COUNT(DISTINCT ua.user_id) / NULLIF(ub_count.total_users, 0) AS percent_B_to_A_users
        FROM mv_user_monthly_activity ua
        JOIN mv_user_monthly_activity ub 
            ON ua.user_id = ub.user_id
            AND ua.channel_id <> ub.channel_id
        JOIN channels ca ON ua.channel_id = ca.channel_id
        JOIN channels cb ON ub.channel_id = cb.channel_id
        JOIN (
            SELECT channel_id, observed_month, COUNT(DISTINCT user_id) AS total_users
            FROM mv_user_monthly_activity
            GROUP BY channel_id, observed_month
        ) ua_count ON ua.channel_id = ua_count.channel_id AND ua.observed_month = ua_count.observed_month
        JOIN (
            SELECT channel_id, observed_month, COUNT(DISTINCT user_id) AS total_users
            FROM mv_user_monthly_activity
            GROUP BY channel_id, observed_month
        ) ub_count ON ub.channel_id = ub_count.channel_id AND ub.observed_month = ub_count.observed_month
        WHERE ca.channel_name = %s AND cb.channel_name = %s
          AND ua.observed_month = %s::DATE
          AND ub.observed_month = %s::DATE
        GROUP BY ca.channel_name, cb.channel_name, ua.observed_month, ub.observed_month, ua_count.total_users, ub_count.total_users;
    """

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(query, (channel_a, channel_b, month_a, month_b))
    results = cursor.fetchall()
    cursor.close()
    conn.close()

    if results:
        data = {
            "channel_a": results[0][0],
            "channel_b": results[0][1],
            "month_a": results[0][2].strftime('%Y-%m'),
            "month_b": results[0][3].strftime('%Y-%m'),
            "num_common_users": results[0][4],
            "percent_A_to_B_users": round(results[0][5], 2) if results[0][5] is not None else None,
            "percent_B_to_A_users": round(results[0][6], 2) if results[0][6] is not None else None
        }
    else:
        data = {}

    return jsonify(data)

@app.route('/api/get_common_members', methods=['GET'])
@cache.cached(timeout=86400, query_string=True)
def get_common_members():
    """API to get common members between two (channel, month) pairs."""
    channel_a = request.args.get('channel_a')
    month_a = request.args.get('month_a')  # YYYY-MM format
    channel_b = request.args.get('channel_b')
    month_b = request.args.get('month_b')  # YYYY-MM format

    if not (channel_a and month_a and channel_b and month_b):
        return jsonify({"error": _("Missing required parameters")}), 400

    month_a += "-01"
    month_b += "-01"

    # Simplified query with pre-computed total members
    query = """
        WITH user_activity AS (
        SELECT
            ud.user_id,
            ud.channel_id,
            DATE_TRUNC('month', ud.last_message_at AT TIME ZONE 'UTC') AS observed_month
        FROM user_data ud
        JOIN channels c ON ud.channel_id = c.channel_id
        WHERE (c.channel_name = %s OR c.channel_name = %s)
        AND ud.last_message_at >= %s::DATE
        AND ud.last_message_at < (%s::DATE + INTERVAL '1 month')
        GROUP BY ud.user_id, ud.channel_id, observed_month
    ),
    common_members AS (
        SELECT
            m1.channel_id AS channel_a,
            m2.channel_id AS channel_b,
            DATE_TRUNC('month', m1.last_message_at AT TIME ZONE 'UTC') AS observed_month_a,
            DATE_TRUNC('month', m2.last_message_at AT TIME ZONE 'UTC') AS observed_month_b,
            COUNT(DISTINCT m1.user_id) AS shared_members
        FROM user_data m1
        JOIN user_data m2 
            ON m1.user_id = m2.user_id
            AND m1.channel_id <> m2.channel_id
        JOIN channels ca ON m1.channel_id = ca.channel_id
        JOIN channels cb ON m2.channel_id = cb.channel_id
        WHERE (ca.channel_name = %s AND cb.channel_name = %s)
        AND m1.membership_rank >= 0
        AND m2.membership_rank >= 0
        AND DATE_TRUNC('month', m1.last_message_at AT TIME ZONE 'UTC') = %s::DATE
        AND DATE_TRUNC('month', m2.last_message_at AT TIME ZONE 'UTC') = %s::DATE
        GROUP BY m1.channel_id, m2.channel_id, observed_month_a, observed_month_b
    ),
    member_counts AS (
        SELECT
            channel_id,
            DATE_TRUNC('month', last_message_at AT TIME ZONE 'UTC') AS observed_month,
            COUNT(DISTINCT user_id) AS total_members
        FROM user_data
        WHERE membership_rank >= 0
        GROUP BY channel_id, observed_month
    )
    SELECT 
        ca.channel_name AS channel_a,
        cb.channel_name AS channel_b,
        cm.observed_month_a AS month_a,
        cm.observed_month_b AS month_b,
        cm.shared_members AS num_common_members,
        CASE WHEN ca.channel_name = %s THEN
            100.0 * cm.shared_members / NULLIF(ma_count.total_members, 0)
        ELSE
            100.0 * cm.shared_members / NULLIF(mb_count.total_members, 0)
        END AS percent_A_to_B_members,
        CASE WHEN cb.channel_name = %s THEN
            100.0 * cm.shared_members / NULLIF(mb_count.total_members, 0)
        ELSE
            100.0 * cm.shared_members / NULLIF(ma_count.total_members, 0)
        END AS percent_B_to_A_members
    FROM common_members cm
    JOIN channels ca ON cm.channel_a = ca.channel_id
    JOIN channels cb ON cm.channel_b = cb.channel_id
    LEFT JOIN member_counts ma_count ON ma_count.channel_id = cm.channel_a AND ma_count.observed_month = cm.observed_month_a
    LEFT JOIN member_counts mb_count ON mb_count.channel_id = cm.channel_b AND mb_count.observed_month = cm.observed_month_b;




    """

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(query, (channel_a, channel_b, month_a, month_b, channel_a, channel_b, month_a, month_b, channel_a, channel_b))
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    if results:
        data = {
            "channel_a": results[0][0],
            "channel_b": results[0][1],
            "month_a": results[0][2].strftime('%Y-%m'),
            "month_b": results[0][3].strftime('%Y-%m'),
            "num_common_members": results[0][4],
            "percent_A_to_B_members": round(results[0][5], 2) if results[0][5] is not None else None,
            "percent_B_to_A_members": round(results[0][6], 2) if results[0][6] is not None else None
        }
    else:
        data = {}

    return jsonify(data)



@app.route('/api/get_group_membership_data', methods=['GET'])
@cache.cached(timeout=86400, query_string=True)
def get_group_membership_counts():
    conn = get_db_connection()
    cursor = conn.cursor()
    channel_group = request.args.get('channel_group')
    month = request.args.get('month')  # Expected format: YYYY-MM

    if not channel_group or not month:
        return jsonify({"error": _("Missing required parameters")}), 400

    query = """SELECT channel_name, membership_rank, membership_count, percentage_total FROM mv_membership_data WHERE channel_group = %s AND observed_month = %s::DATE;"""

    cursor.execute(query, (channel_group, f"{month}-01"))
    results = cursor.fetchall()
    cursor.close()
    conn.close()

    return jsonify(results)

@app.route('/api/get_group_membership_changes', methods=['GET'])
@cache.cached(timeout=86400, query_string=True)
def get_group_membership_changes():
    """API to fetch membership gains, losses, and differential by group and month."""
    conn = get_db_connection()
    cursor = conn.cursor()

    channel_group = request.args.get('channel_group')
    month = request.args.get('month')  # Expected format: YYYY-MM
    

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
        conn.close()

    # Format the results as JSON
    response_data = [
        {
            "channel_name": row[0],
            "observed_month": row[1].strftime('%Y-%m'),
            "gains_count": row[2],
            "losses_count": row[3],
            "differential": row[4]
        }
        for row in results
    ]

    return jsonify(response_data)


@app.route('/api/get_group_streaming_hours_diff', methods=['GET'])
@cache.cached(timeout=86400, query_string=True)
def get_group_streaming_hours_diff():
    """API to fetch the change in streaming hours since the previous month for a given group and timezone."""

    # Get request parameters
    month = request.args.get('month')  # Expected format: YYYY-MM
    timezone_offset = request.args.get('timezone', '0')  # Default to UTC
    channel_group = request.args.get('group', None)

    if not month:
        return jsonify({"success": False, "error": "Missing required parameter: month"}), 400

    try:
        # Convert month into correct format
        month_date = datetime.strptime(month, "%Y-%m")
        prev_month_date = (month_date.replace(day=1) - timedelta(days=1)).replace(day=1)
    except ValueError:
        return jsonify({"success": False, "error": "Invalid month format. Use YYYY-MM."}), 400

    # SQL Query
    query = """
        WITH monthly_streaming AS (
            SELECT
                c.channel_name,
                DATE_TRUNC('month', v.end_time AT TIME ZONE 'UTC' + INTERVAL %s) AS observed_month,
                SUM(EXTRACT(EPOCH FROM v.duration)) / 3600 AS total_streaming_hours
            FROM videos v
            JOIN channels c ON v.channel_id = c.channel_id
            {group_filter}
            GROUP BY c.channel_name, observed_month
        )
        SELECT
            m1.channel_name,
            m1.observed_month,
            m1.total_streaming_hours,
            COALESCE((m1.total_streaming_hours - m2.total_streaming_hours), 0) AS change_from_previous_month
        FROM monthly_streaming m1
        LEFT JOIN monthly_streaming m2
            ON m1.channel_name = m2.channel_name AND m1.observed_month = m2.observed_month + INTERVAL '1 month'
        WHERE m1.observed_month = %s
        ORDER BY m1.channel_name;
    """

    # Apply channel group filtering if provided
    group_filter = ""
    params = [f"{timezone_offset} hours"]
    if channel_group:
        group_filter = "WHERE c.channel_group = %s"
        params.append(channel_group)

    params.append(f"{month_date.strftime('%Y-%m-01')}")

    # Replace placeholder with actual SQL filter
    query = query.format(group_filter=group_filter)

    # Execute the query
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(query, tuple(params))
    results = cursor.fetchall()

    cursor.close()
    conn.close()

    # Format the response
    data = [
        {
            "channel": row[0],
            "month": row[1].strftime('%Y-%m'),
            "hours": round(row[2], 2) if row[2] is not None else 0,
            "change": round(row[3], 2)
        }
        for row in results
    ]

    return jsonify({"success": True, "data": data})

@app.route('/api/get_chat_leaderboard', methods=['GET'])
@cache.cached(timeout=86400, query_string=True)
def get_chat_leaderboard():
    """API to fetch the top 10 chatters for a given channel and month."""

    channel_name = request.args.get('channel_name')
    month = request.args.get('month')  # Expected format: YYYY-MM

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

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(query, (channel_name, month_start, month_start))
    
    results = cursor.fetchall()
    cursor.close()
    conn.close()

    if not results:
        return jsonify({"error": _("No data found")}), 404

    return jsonify([
        {"user_name": row[0], "message_count": row[1]}
        for row in results
    ])

@app.route('/api/get_user_changes', methods=['GET'])
@cache.cached(timeout=86400, query_string=True)
def get_user_changes():
    """API to fetch user gains and losses per channel in a given group and month."""

    channel_group = request.args.get('group')
    month = request.args.get('month')  # YYYY-MM format

    if not (channel_group and month):
        return jsonify({"error": _("Missing required parameters")}), 400

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
        users_gained AS (
            SELECT user_id, channel_id
            FROM current_month_users
            EXCEPT
            SELECT user_id, channel_id
            FROM previous_month_users
        ),
        users_lost AS (
            SELECT user_id, channel_id
            FROM previous_month_users
            EXCEPT
            SELECT user_id, channel_id
            FROM current_month_users
        )
        SELECT
            c.channel_name,
            (SELECT COUNT(*) FROM users_gained WHERE channel_id = c.channel_id) AS users_gained,
            (SELECT COUNT(*) FROM users_lost WHERE channel_id = c.channel_id) AS users_lost,
            (SELECT COUNT(*) FROM users_gained WHERE channel_id = c.channel_id) -
            (SELECT COUNT(*) FROM users_lost WHERE channel_id = c.channel_id) AS net_change
        FROM channels c
        WHERE c.channel_group = %s
        ORDER BY net_change DESC;
    """

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(query, (current_month_start, previous_month_start, channel_group))
    results = cursor.fetchall()
    cursor.close()
    conn.close()

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

    return jsonify(filtered_results)

@app.route('/api/get_exclusive_chat_users', methods=['GET'])
@cache.cached(timeout=86400, query_string=True)
def get_exclusive_chat_users():
    """API to fetch the percentage of exclusive chat users for a given channel per month."""

    channel_name = request.args.get('channel')

    if not channel_name:
        return jsonify({"error": _("Missing required parameters")}), 400

    query = """
        WITH channel_specific_users AS (
            SELECT
                user_id,
                activity_month,
                channel_id
            FROM mv_user_activity
            WHERE channel_id = (SELECT channel_id FROM channels WHERE channel_name = %s)
        ),
        exclusive_users AS (
            SELECT
                csu.activity_month,
                COUNT(DISTINCT csu.user_id) AS exclusive_users_count
            FROM channel_specific_users csu
            LEFT JOIN mv_user_activity gu
                ON csu.user_id = gu.user_id
                AND gu.channel_id <> csu.channel_id
                AND gu.channel_group = (SELECT channel_group FROM channels WHERE channel_name = %s)
            WHERE gu.user_id IS NULL
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

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(query, (channel_name, channel_name))
    results = cursor.fetchall()
    cursor.close()
    conn.close()

    # Convert results into JSON format
    return jsonify([
        {"month": row[0].strftime('%Y-%m'), "percent": row[1]}
        for row in results
    ])

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

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(query, (channel_name, channel_name))
    results = cursor.fetchall()
    cursor.close()
    conn.close()

    # Convert results into JSON format
    return jsonify([
        {"month": row[0].strftime('%Y-%m'), "percent": row[1], "message_rate": row[2]}
        for row in results
    ])

@app.route('/api/get_jp_user_percent', methods=['GET'])
@cache.cached(timeout=86400, query_string=True)
def get_jp_user_percent():
    """Returns the percentage of users using mostly Japanese (>50% of messages, excluding emoji-only messages) per month for a given channel."""
    channel_name = request.args.get('channel')

    if not channel_name:
        return jsonify({"error": "Missing required parameter: channel"}), 400

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

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(query, (channel_name, channel_name))
    results = cursor.fetchall()
    cursor.close()
    conn.close()

    # Format results for Chart.js
    response = [{"month": row[0], "jp_user_percent": row[1]} for row in results]

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
@cache.cached(timeout=86400)
def get_channel_names():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT channel_name FROM channels ORDER BY channel_name")
    channel_names = [row[0] for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    return jsonify(channel_names)

@app.route('/api/get_date_ranges', methods=['GET'])
@cache.cached(timeout=86400)
def get_date_ranges():
    conn = get_db_connection()
    cursor = conn.cursor()

    # Gets first and last date from video table using end_time
    cursor.execute("SELECT MIN(end_time), MAX(end_time) FROM videos WHERE has_chat_log = 't'")
    
    date_range = cursor.fetchone()
    cursor.close()
    conn.close()
    return jsonify(date_range)

@app.route('/api/get_number_of_chat_logs', methods=['GET'])
@cache.cached(timeout=86400)
def get_number_of_chat_logs():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) from videos WHERE has_chat_log = 't'")
    num_chat_logs = cursor.fetchone()[0]
    cursor.close()
    conn.close()
    return jsonify(num_chat_logs)

@app.route('/api/get_funniest_timestamps', methods=['GET'])
@cache.cached(timeout=86400, query_string=True)
def get_funniest_timestamps():
    """API to fetch funniest timestamps using actual last chat message timestamp."""

    channel_name = request.args.get('channel')
    month = request.args.get('month')  # Expected format: YYYY-MM

    if not (channel_name and month):
        return jsonify({"error": "Missing required parameters"}), 400

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

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(query, (channel_name, month_start, next_month_start, channel_name))
    results = cursor.fetchall()
    cursor.close()
    conn.close()

    return jsonify([
        {"title": row[0], "video_id": row[1], "timestamp": int(row[2])}
        for row in results if row[1] is not None and 0 <= row[2]
    ])

@app.route('/api/get_user_info', methods=['GET'])
@cache.cached(timeout=86400, query_string=True)
def get_user_info():
    """API to fetch all channels a user chatted on in a given month,
    their total messages on that channel, and their frequency percentile."""
    
    username = request.args.get('username')
    month = request.args.get('month')  # YYYY-MM format

    if not (username and month):
        return jsonify({"error": "Missing required parameters"}), 400

    # Convert month to first day format for filtering
    month_start = f"{month}-01"
    next_month_start = (datetime.strptime(month, "%Y-%m") + relativedelta(months=1)).strftime("%Y-%m-01")

    conn = get_db_connection()
    cursor = conn.cursor()

    # Get the user's ID
    cursor.execute("SELECT user_id FROM users WHERE username = %s", (username,))
    user_row = cursor.fetchone()
    
    if not user_row:
        cursor.close()
        conn.close()
        return jsonify({"error": "User not found"}), 404

    user_id = user_row[0]

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

    # If user has no messages
    if not user_chat_data:
        cursor.close()
        conn.close()
        return jsonify({"error": "No data available for the given user and month"}), 404

    # Prepare data structure
    results = []

    for channel_id, channel_name, user_message_count in user_chat_data:
        # Compute user's percentile rank on this channel
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
                100 * (SELECT COUNT(*) FROM all_user_counts WHERE total_messages < %s) 
                / NULLIF((SELECT COUNT(*) FROM all_user_counts), 0) AS percentile
        """, (channel_id, month_start, next_month_start, user_message_count))

        percentile_row = cursor.fetchone()
        percentile = percentile_row[0] if percentile_row and percentile_row[0] is not None else 0.0

        # Append to results
        results.append({
            "channel_name": channel_name,
            "message_count": user_message_count,
            "percentile": round(percentile, 2)
        })

    cursor.close()
    conn.close()

    return jsonify(results)


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

if __name__ == '__main__':
    app.run(debug=True)