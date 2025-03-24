from flask import Flask, request, jsonify, render_template, session, redirect
from flask_babel import Babel, _
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_caching import Cache
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import configparser
import os
import psycopg2
import logging
import sys

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

app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)

cache = Cache(app, config={"CACHE_TYPE": "simple"})

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

# Access logging
@app.before_request
def before_request():
    real_ip = request.headers.get("CF-Connecting-IP", request.remote_addr)
    app.logger.info(f"Request from {real_ip} to {request.path}")

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
        print(query % tuple(params))  # Print the query with parameters for debugging
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
    
@app.route('/api/common_chatters', methods=['GET'])
@cache.cached(timeout=86400, query_string=True)
def get_common_chatters():
    """API to get common chatters and common members between two (channel, month) pairs using channel names."""

    channel_a = request.args.get('channel_a')
    month_a = request.args.get('month_a')  # YYYY-MM format
    channel_b = request.args.get('channel_b')
    month_b = request.args.get('month_b')  # YYYY-MM format
    timezone_offset = request.args.get('timezone', 0)  # Optional timezone offset

    timezone_interval = f"{timezone_offset} hours"

    if not (channel_a and month_a and channel_b and month_b):
        return jsonify({"error": _("Missing required parameters")}), 400

    month_a += "-01"
    month_b += "-01"

    query = """
        WITH user_activity AS (
            SELECT
                ud.user_id,
                ud.channel_id,
                DATE_TRUNC('month', ud.last_message_at AT TIME ZONE 'UTC' + INTERVAL %s) AS observed_month
            FROM user_data ud
            JOIN channels c ON ud.channel_id = c.channel_id
            WHERE (c.channel_name = %s AND ud.last_message_at >= %s::DATE AND ud.last_message_at < (%s::DATE + INTERVAL '1 month'))
            OR (c.channel_name = %s AND ud.last_message_at >= %s::DATE AND ud.last_message_at < (%s::DATE + INTERVAL '1 month'))
            GROUP BY ud.user_id, ud.channel_id, observed_month
        ),
        common_chatters AS (
            SELECT
                ua1.channel_id AS channel_a,
                ua2.channel_id AS channel_b,
                ua1.observed_month AS month_a,
                ua2.observed_month AS month_b,
                COUNT(DISTINCT ua1.user_id) AS shared_users
            FROM user_activity ua1
            JOIN user_activity ua2 
                ON ua1.user_id = ua2.user_id
                AND ua1.channel_id <> ua2.channel_id
                AND ua1.observed_month <= ua2.observed_month
            GROUP BY ua1.channel_id, ua2.channel_id, ua1.observed_month, ua2.observed_month
        ),
        user_counts AS (
            SELECT 
                channel_id, 
                observed_month, 
                COUNT(DISTINCT user_id) AS total_users
            FROM user_activity
            GROUP BY channel_id, observed_month
        ),
        common_members AS (
            SELECT
                m1.channel_id AS channel_a,
                m2.channel_id AS channel_b,
                DATE_TRUNC('month', m1.last_message_at AT TIME ZONE 'UTC' + INTERVAL %s) AS observed_month_a,
                DATE_TRUNC('month', m2.last_message_at AT TIME ZONE 'UTC' + INTERVAL %s) AS observed_month_b,
                COUNT(DISTINCT m1.user_id) AS shared_members
            FROM user_data m1
            JOIN user_data m2 
                ON m1.user_id = m2.user_id
                AND m1.channel_id <> m2.channel_id
            JOIN channels ca ON m1.channel_id = ca.channel_id
            JOIN channels cb ON m2.channel_id = cb.channel_id
            WHERE ca.channel_name = %s
              AND cb.channel_name = %s
              AND m1.membership_rank >= 0
              AND m2.membership_rank >= 0
              AND DATE_TRUNC('month', m1.last_message_at AT TIME ZONE 'UTC' + INTERVAL %s) = %s::DATE
              AND DATE_TRUNC('month', m2.last_message_at AT TIME ZONE 'UTC' + INTERVAL %s) = %s::DATE
            GROUP BY m1.channel_id, m2.channel_id, observed_month_a, observed_month_b
        ),
        member_counts AS (
            SELECT
                channel_id,
                DATE_TRUNC('month', last_message_at AT TIME ZONE 'UTC' + INTERVAL %s) AS observed_month,
                COUNT(DISTINCT user_id) AS total_members
            FROM user_data
            WHERE membership_rank >= 0
            GROUP BY channel_id, observed_month
        )
        SELECT 
            ca.channel_name AS channel_a,
            cb.channel_name AS channel_b,
            cg.month_a AS month_a,
            cg.month_b AS month_b,
            cg.shared_users AS num_common_users,
            100.0 * cg.shared_users / NULLIF(ua_count.total_users, 0) AS percent_A_to_B_users,
            100.0 * cg.shared_users / NULLIF(ub_count.total_users, 0) AS percent_B_to_A_users,
            COALESCE(cm.shared_members, 0) AS num_common_members,
            100.0 * COALESCE(cm.shared_members, 0) / NULLIF(ma_count.total_members, 0) AS percent_A_to_B_members,
            100.0 * COALESCE(cm.shared_members, 0) / NULLIF(mb_count.total_members, 0) AS percent_B_to_A_members
        FROM common_chatters cg
        JOIN channels ca ON cg.channel_a = ca.channel_id
        JOIN channels cb ON cg.channel_b = cb.channel_id
        JOIN user_counts ua_count ON ua_count.channel_id = cg.channel_a AND ua_count.observed_month = cg.month_a
        JOIN user_counts ub_count ON ub_count.channel_id = cg.channel_b AND ub_count.observed_month = cg.month_b
        LEFT JOIN common_members cm ON cm.channel_a = ca.channel_id AND cm.channel_b = cb.channel_id
        LEFT JOIN member_counts ma_count ON ma_count.channel_id = ca.channel_id AND ma_count.observed_month = cg.month_a
        LEFT JOIN member_counts mb_count ON mb_count.channel_id = cb.channel_id AND mb_count.observed_month = cg.month_b;
    """

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(query, (
        timezone_interval,
        channel_a, month_a, month_a,
        channel_b, month_b, month_b,
        timezone_interval, timezone_interval,
        channel_a, channel_b,
        timezone_interval, month_a,
        timezone_interval, month_b,
        timezone_interval
    ))

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
            "percent_B_to_A_users": round(results[0][6], 2) if results[0][6] is not None else None,
            "num_common_members": results[0][7],
            "percent_A_to_B_members": round(results[0][8], 2) if results[0][8] is not None else None,
            "percent_B_to_A_members": round(results[0][9], 2) if results[0][9] is not None else None
        }
    else:
        data = {}

    return jsonify(data)




@app.route('/api/get_group_common_chatters', methods=['GET'])
@cache.cached(timeout=86400, query_string=True)
def get_group_common_chatters():
    """API to fetch common chatters between channels in a given group and month from `mv_common_chatters`."""

    channel_group = request.args.get('channel_group')
    month = request.args.get('month')  # Expected format: YYYY-MM

    if not channel_group or not month:
        return jsonify({"error": _("Missing required parameters")}), 400

    query = """
        SELECT 
            channel_a, 
            channel_b, 
            percent_a_to_b
        FROM mv_common_chatters
        WHERE channel_group = %s 
          AND observed_month = %s::DATE;
    """

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(query, (channel_group, f"{month}-01"))
    results = cursor.fetchall()
    cursor.close()
    conn.close()

    if not results:
        return jsonify({"error": _("No data found")}), 404

    # Extract unique channels
    channels = list(set(row[0] for row in results) | set(row[1] for row in results))
    channels.sort()  # Alphabetical order

    # Create adjacency matrix
    matrix = [[0 for _ in range(len(channels))] for _ in range(len(channels))]

    for row in results:
        i = channels.index(row[0])  # A
        j = channels.index(row[1])  # B
        matrix[i][j] = round(row[2], 2)  # A->B

    return jsonify({
        "channels": channels,
        "matrix": matrix
    })

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
    cursor.execute("SELECT MIN(end_time), MAX(end_time) FROM videos")
    
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

if __name__ == '__main__':
    app.run(debug=True)