import json
import re
import urllib.parse
from datetime import datetime, timezone
from flask import Blueprint, request, jsonify, g, Response
from flask_babel import _
from dateutil.relativedelta import relativedelta
import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import leidenalg as la
import igraph as ig
import plotly.graph_objects as go
import numpy as np
from plotly.utils import PlotlyJSONEncoder
import requests

from utils import (
    get_db_connection, get_sqlite_connection, inc_cache_hit_count, inc_cache_miss_count,
    streaming_hours_query, parse_search_query, EMBEDDER, validate_month_format,
    format_month_for_sql
)

api_bp = Blueprint('api', __name__)


@api_bp.route('/api/channel_clustering', methods=['GET'])
def channel_clustering():
    """
Generates a channel clustering graph based on user similarity for a given month.

Args:
    month (str, required): Month filter in YYYY-MM format (query param)
    percentile (str, optional): Similarity threshold percentile, default "95" (query param)
    type (str, optional): Graph type "2d" or "3d", default "2d" (query param)

Returns:
    Success (200): JSON with "graph_json" containing Plotly figure data
    Failure (400): Missing month parameter
    Failure (404): No data found for specified month
    Failure (500): Internal server error
"""
    try:
        filter_month = request.args.get("month")
        percentile = request.args.get("percentile", "95")
        graph_type = request.args.get("type", "2d")
        
        if not filter_month:
            return jsonify({"error": "Month filter (e.g., '2025-03') is required"}), 400
        
        redis_key = f"channel_clustering_{filter_month}_{percentile}_{graph_type}"
        cursor = g.db_conn.cursor()

        cached_data = g.redis_conn.get(redis_key)
        if cached_data:
            inc_cache_hit_count()
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

        g_igraph = ig.Graph.from_networkx(G)
        partition = la.find_partition(
            g_igraph, 
            la.RBConfigurationVertexPartition,
            weights='weight',
            resolution_parameter=1.0
        )
        
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
            hover_traces = []
            num_hover_points = 10

            edge_x, edge_y, edge_z = [], [], []

            for u, v in G.edges():
                x0, y0, z0 = pos[u]
                x1, y1, z1 = pos[v]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edge_z.extend([z0, z1, None])

            for (u, v), norm_weight in zip(G.edges(), normalized_weights):
                x0, y0, z0 = pos[u]
                x1, y1, z1 = pos[v]
                adjusted_opacity = 0.1 + (norm_weight ** 1.1) * 0.9

                fig.add_trace(go.Scatter3d(
                    x=edge_x,
                    y=edge_y,
                    z=edge_z,
                    mode='lines',
                    line=dict(width=1.5, color=f'rgba(0,0,0,{adjusted_opacity})'),
                    hoverinfo='none'
                ))

                line_x = np.linspace(x0, x1, num_hover_points)
                line_y = np.linspace(y0, y1, num_hover_points)
                line_z = np.linspace(z0, z1, num_hover_points)

                hover_traces.append(go.Scatter3d(
                    x=line_x.tolist(),
                    y=line_y.tolist(),
                    z=line_z.tolist(),
                    mode='markers',
                    marker=dict(size=6, color='rgba(255,255,255,0)'),
                    hoverinfo='text',
                    hovertext=[f"{u} ↔ {v}<br>Score: {G[u][v]['weight'] * 100:.2f}"] * num_hover_points
                ))

            for trace in edge_traces:
                fig.add_trace(trace)
            for trace in hover_traces:
                fig.add_trace(trace)

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
            pos = nx.forceatlas2_layout(G, strong_gravity=True)
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
            hover_traces = []
            num_hover_points = 10

            for (u, v), norm_weight in zip(G.edges(), normalized_weights):
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                adjusted_opacity = 0.1 + (norm_weight ** 1.1) * 0.9

                edge_traces.append(go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=1.5, color=f'rgba(0,0,0,{adjusted_opacity})'),
                    hoverinfo='none'
                ))

                line_x = np.linspace(x0, x1, num_hover_points)
                line_y = np.linspace(y0, y1, num_hover_points)

                hover_traces.append(go.Scatter(
                    x=line_x.tolist(),
                    y=line_y.tolist(),
                    mode='markers',
                    marker=dict(size=6, color='rgba(255,255,255,0)'),
                    hoverinfo='text',
                    hovertext=[f"{u} ↔ {v}<br>Score: {G[u][v]['weight'] * 100:.2f}"] * num_hover_points
                ))

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
                title=f"Channel User Similarity Graph for {formatted_month}",
                title_x=0.5,
                showlegend=False,
                margin=dict(l=10, r=10, t=50, b=10),
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                plot_bgcolor='white'
            )

        graph_json = json.dumps(fig, cls=PlotlyJSONEncoder)
        output = {"graph_json": graph_json}
        g.redis_conn.set(redis_key, json.dumps(output))

        return jsonify(output)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route("/api/recommend", methods=["GET"])
def recommend_channels():
    """
Recommends channels for a user based on their recent activity patterns.

Args:
    identifier (str, required): User ID or @username handle (query param)

Returns:
    Success (200): JSON with "recommended_channels" list containing channel_name and score
    Failure (400): Missing identifier parameter
    Failure (404): User not found or no activity data available
    Failure (500): Internal server error
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
            user_id = identifier

        redis_key = f"channel_recommendations_{user_id}"
        cached_data = g.redis_conn.get(redis_key)
        if cached_data:
            inc_cache_hit_count()
            return jsonify(json.loads(cached_data))
        inc_cache_miss_count()

        last_month = datetime.utcnow().date().replace(day=1) - relativedelta(days=1)
        month_to_query = last_month.strftime("%Y-%m-01")

        redis_intermediate_key = f"channel_recommendation_data_{month_to_query}"
        cached_rows = g.redis_conn.get(redis_intermediate_key)
        
        if cached_rows:
            rows = json.loads(cached_rows)
        else:
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
        user_channel_matrix = data.pivot(index='user_id', columns='channel_name', values='message_weight').fillna(0)
        similarity_matrix = cosine_similarity(user_channel_matrix.T)
        channel_names = user_channel_matrix.columns
        similarity_df = pd.DataFrame(similarity_matrix, index=channel_names, columns=channel_names)

        if user_id not in user_channel_matrix.index:
            return jsonify({"error": "No activity found for this user in the last month."}), 404

        user_vector = user_channel_matrix.loc[user_id]
        user_channels = user_vector[user_vector > 0].index
        recommendation_scores = similarity_df[user_channels].sum(axis=1)
        recommendation_scores = recommendation_scores.drop(labels=user_channels, errors='ignore')
        top_recommendations = recommendation_scores.sort_values(ascending=False).head(5)
        
        ideal_max = len(user_channels)
        raw_normalized = (top_recommendations / ideal_max) * 100
        normalized_scores = np.log1p(raw_normalized) / np.log1p(100) * 100

        response = [
            {"channel_name": channel, "score": round(score, 2)}
            for channel, score in normalized_scores.items()
        ]

        output = {"recommended_channels": response}
        g.redis_conn.set(redis_key, json.dumps(output))

        return jsonify(output)

    except Exception as e:
        print(f"An error occurred in recommend_channels: {e}")
        return jsonify({"error": "An internal server error occurred."}), 500
    finally:
        if 'conn' in locals() and conn:
            cursor.close()
            conn.close()


@api_bp.route('/api/get_monthly_streaming_hours', methods=['GET'])
def get_monthly_streaming_hours():
    """
    Fetches total streaming hours per month for a specific channel.

    Args:
        channel (str, required): Channel name (query param)

    Returns:
        Success (200): JSON array of objects with "month" and "total_streaming_hours"
        Failure (400): Missing channel parameter
    """
    channel_name = request.args.get('channel')
    redis_key = f"monthly_streaming_hours_{channel_name}"
    
    cached_data = g.redis_conn.get(redis_key)
    if cached_data:
        inc_cache_hit_count()
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

    output = jsonify([{"month": row[0].strftime('%Y-%m'), "total_streaming_hours": row[1]} for row in results])
    g.redis_conn.set(redis_key, output.get_data(as_text=True))
    return output


@api_bp.route('/api/get_group_total_streaming_hours', methods=['GET'])
def get_group_total_streaming_hours():
    """
    Fetch total streaming hours per channel for a given month, optionally filtered by group.

    Args:
        month (str, optional): Month in YYYY-MM format, defaults to current month (query parameter)
        group (str, optional): Channel group name filter (query parameter)

    Returns:
        Success (200): JSON with success=True and data array containing channel, month, hours
        Error (500): JSON with success=False and error message
    """
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


@api_bp.route('/api/get_group_avg_streaming_hours', methods=['GET'])
def get_group_avg_streaming_hours():
    """
    Fetch average streaming hours per stream for channels in a given month and optional group.

    Args:
        month (str, optional): Month in YYYY-MM format, defaults to current month (query parameter)
        group (str, optional): Channel group name filter (query parameter)

    Returns:
        Success (200): JSON with success=True and data array containing channel, month, hours
        Error (500): JSON with success=False and error message
    """
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


@api_bp.route('/api/get_group_max_streaming_hours', methods=['GET'])
def get_group_max_streaming_hours():
    """
    Fetch maximum single stream duration for channels in a given month and optional group.

    Args:
        month (str, optional): Month in YYYY-MM format, defaults to current month (query parameter)
        group (str, optional): Channel group name filter (query parameter)

    Returns:
        Success (200): JSON with success=True and data array containing channel, month, hours
        Error (500): JSON with success=False and error message
    """
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


@api_bp.route('/api/get_group_chat_makeup', methods=['GET'])
def get_group_chat_makeup():
    """
    Fetch chat language composition rates per minute for channels, optionally filtered by group.

    Args:
        month (str, optional): Month in YYYY-MM format, defaults to current month (query parameter)
        group (str, optional): Channel group name filter (query parameter)

    Returns:
        Success (200): JSON with success=True and data array containing language rates per minute
        Error (500): JSON with success=False and error message
    """
    group = request.args.get('group', None)
    month = request.args.get('month', datetime.utcnow().strftime('%Y-%m'))

    redis_key = f"group_chat_makeup_{group}_{month}"
    cached_data = g.redis_conn.get(redis_key)
    if cached_data:
        inc_cache_hit_count()
        return jsonify(json.loads(cached_data))
    inc_cache_miss_count()

    start_month = datetime.strptime(month + "-01", '%Y-%m-%d').strftime('%Y-%m-01')

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

    if group:
        query += " WHERE c.channel_group = %s"
        params.append(group)

    query += " GROUP BY c.channel_name, st.observed_month ORDER BY SUM(st.total_streaming_minutes) DESC"

    try:
        cur = g.db_conn.cursor()
        cur.execute(query, params)
        results = cur.fetchall()
        cur.close()

        data = [
            {
                "channel_name": row[0],
                "observed_month": row[1].strftime('%Y-%m'),
                "es_en_id_rate_per_minute": round(float(row[2]) if row[2] is not None else 0, 2),
                "jp_rate_per_minute": round(float(row[3]) if row[3] is not None else 0, 2),
                "kr_rate_per_minute": round(float(row[4]) if row[4] is not None else 0, 2),
                "ru_rate_per_minute": round(float(row[5]) if row[5] is not None else 0, 2),
                "emoji_rate_per_minute": round(float(row[6]) if row[6] is not None else 0, 2)
            }
            for row in results
        ]

        output = jsonify({"success": True, "data": data})
        g.redis_conn.set(redis_key, output.get_data(as_text=True))
        return output

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@api_bp.route('/api/get_common_users', methods=['GET'])
def get_common_users():
    """
    Calculate common users between two channel-month pairs using set intersection.

    Args:
        channel_a (str, required): First channel name (query parameter)
        month_a (str, required): First month in YYYY-MM format (query parameter)
        channel_b (str, required): Second channel name (query parameter)
        month_b (str, required): Second month in YYYY-MM format (query parameter)

    Returns:
        Success (200): JSON with num_common_users and percentage overlaps for both directions
        Error (400): Missing required parameters
        Error (500): Database query failed
    """
    channel_a_name = request.args.get('channel_a')
    month_a_str = request.args.get('month_a')
    channel_b_name = request.args.get('channel_b')
    month_b_str = request.args.get('month_b')

    if not (channel_a_name and month_a_str and channel_b_name and month_b_str):
        return jsonify({"error": _("Missing required parameters")}), 400
    
    redis_key = f"common_users_{channel_a_name}_{month_a_str}_{channel_b_name}_{month_b_str}"
    cached_data = g.redis_conn.get(redis_key)
    if cached_data:
        inc_cache_hit_count()
        return jsonify(json.loads(cached_data))
    inc_cache_miss_count()
    
    cursor = g.db_conn.cursor()
    
    query = """
        SELECT DISTINCT mua.user_id
        FROM mv_user_monthly_activity mua
        JOIN channels c ON mua.channel_id = c.channel_id
        WHERE c.channel_name = %s
          AND mua.observed_month = %s::DATE;
    """
    
    try:
        month_a_date = f"{month_a_str}-01"
        cursor.execute(query, (channel_a_name, month_a_date))
        users_a = {row[0] for row in cursor.fetchall()}

        month_b_date = f"{month_b_str}-01"
        cursor.execute(query, (channel_b_name, month_b_date))
        users_b = {row[0] for row in cursor.fetchall()}
        
    except Exception as e:
        cursor.close()
        return jsonify({"error": f"Database query failed: {e}"}), 500
    finally:
        cursor.close()

    if not users_a or not users_b:
        return jsonify({})

    common_users = users_a.intersection(users_b)
    
    num_common_users = len(common_users)
    total_users_a = len(users_a)
    total_users_b = len(users_b)

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

    g.redis_conn.set(redis_key, json.dumps(data), ex=86400)
    
    return jsonify(data)


@api_bp.route('/api/get_common_users_matrix', methods=['GET'])
def get_common_users_matrix():
    """
    Generates a matrix of common user/member percentages for multiple channels.

    Args:
        month (str, required): Month in YYYY-MM format (query param)
        channels (str, required): Comma-separated list of channel names (query param)
        members_only (str, optional): Filter for members only, default "false" (query param)

    Returns:
        Success (200): JSON with "labels" array and "matrix" 2D array of percentages
        Failure (400): Missing parameters or fewer than 2 channels
        Failure (500): Database query error
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
        query = """
            SELECT DISTINCT ud.user_id
            FROM user_data ud
            JOIN channels c ON ud.channel_id = c.channel_id
            WHERE c.channel_name = %s
              AND DATE_TRUNC('month', ud.last_message_at) = %s::DATE
              AND ud.membership_rank >= 0;
        """
    else:
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


@api_bp.route('/api/get_common_members', methods=['GET'])
def get_common_members():
    """
    Calculate common members between two channel-month pairs using set intersection.

    Args:
        channel_a (str, required): First channel name (query parameter)
        month_a (str, required): First month in YYYY-MM format (query parameter)
        channel_b (str, required): Second channel name (query parameter)
        month_b (str, required): Second month in YYYY-MM format (query parameter)

    Returns:
        Success (200): JSON with num_common_members and percentage overlaps for both directions
        Error (400): Missing required parameters
        Error (500): Database query failed
    """
    channel_a_name = request.args.get('channel_a')
    month_a_str = request.args.get('month_a')
    channel_b_name = request.args.get('channel_b')
    month_b_str = request.args.get('month_b')

    if not (channel_a_name and month_a_str and channel_b_name and month_b_str):
        return jsonify({"error": _("Missing required parameters")}), 400
    
    redis_key = f"common_members_{channel_a_name}_{month_a_str}_{channel_b_name}_{month_b_str}"
    cached_data = g.redis_conn.get(redis_key)
    if cached_data:
        inc_cache_hit_count()
        return jsonify(json.loads(cached_data))
    inc_cache_miss_count()
    
    cursor = g.db_conn.cursor()
    
    query = """
        SELECT DISTINCT ud.user_id
        FROM user_data ud
        JOIN channels c ON ud.channel_id = c.channel_id
        WHERE c.channel_name = %s
          AND DATE_TRUNC('month', ud.last_message_at) = %s::DATE
          AND ud.membership_rank >= 0;
    """
    
    try:
        month_a_date = f"{month_a_str}-01"
        cursor.execute(query, (channel_a_name, month_a_date))
        members_a = {row[0] for row in cursor.fetchall()}

        month_b_date = f"{month_b_str}-01"
        cursor.execute(query, (channel_b_name, month_b_date))
        members_b = {row[0] for row in cursor.fetchall()}
        
    except Exception as e:
        cursor.close()
        return jsonify({"error": f"Database query failed: {e}"}), 500
    finally:
        cursor.close()

    if not members_a or not members_b:
        return jsonify({})

    common_members = members_a.intersection(members_b)
    
    num_common_members = len(common_members)
    total_members_a = len(members_a)
    total_members_b = len(members_b)

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

    g.redis_conn.set(redis_key, json.dumps(data), ex=86400)
    
    return jsonify(data)


@api_bp.route('/api/get_group_membership_data', methods=['GET'])
def get_group_membership_counts():
    """
    Fetches membership tier distribution data for a channel group and month.

    Args:
        channel_group (str, required): Channel group name (query param)
        month (str, required): Month in YYYY-MM format (query param)

    Returns:
        Success (200): JSON array with channel_name, membership_rank, count, percentage
        Failure (400): Missing required parameters
    """
    cursor = g.db_conn.cursor()
    channel_group = request.args.get('channel_group')
    month = request.args.get('month')

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


@api_bp.route('/api/get_group_membership_summary', methods=['GET'])
def get_group_membership_summary():
    """
    Returns membership summary by tier or total for channels in a group.

    Args:
        channel_group (str, required): Channel group name (query param)
        month (str, required): Month in YYYY-MM format (query param)
        membership_rank (str, required): Tier number or "total" (query param)

    Returns:
        Success (200): JSON array with membership counts per channel
        Failure (400): Missing required parameters
    """
    channel_group = request.args.get("channel_group")
    month = request.args.get("month")
    membership_rank = request.args.get("membership_rank", type=str)
    
    if membership_rank.lower() == "total":
        total = True
    else:
        total = False
        membership_rank = int(membership_rank)

    if not channel_group or not month:
        return jsonify({"error": "Missing required parameters: channel_group and month"}), 400

    redis_key = f"group_membership_summary_{channel_group}_{month}_{membership_rank}_{total}"
    cached_data = g.redis_conn.get(redis_key)
    if cached_data:
        inc_cache_hit_count()
        return jsonify(json.loads(cached_data))
    inc_cache_miss_count()

    cursor = g.db_conn.cursor()

    if total:
        query = """
            SELECT channel_name, SUM(membership_count) AS total_members
            FROM mv_membership_data
            WHERE channel_group = %s 
              AND observed_month = %s::DATE
              AND membership_rank >= 0
            GROUP BY channel_name
            ORDER BY total_members DESC
        """
        cursor.execute(query, (channel_group, f"{month}-01"))
        results = cursor.fetchall()
        output = [
            {"channel_name": row[0], "total_members": int(row[1])}
            for row in results
        ]
    else:
        query = """
            SELECT channel_name, membership_rank, membership_count, percentage_total
            FROM mv_membership_data
            WHERE channel_group = %s 
              AND observed_month = %s::DATE 
              AND membership_rank = %s
            ORDER BY membership_count DESC
        """
        cursor.execute(query, (channel_group, f"{month}-01", membership_rank))
        results = cursor.fetchall()
        output = [
            [row[0], int(row[1]), float(row[2]), float(row[3])]
            for row in results
        ]

    cursor.close()
    g.redis_conn.set(redis_key, json.dumps(output))

    return jsonify(output)


@api_bp.route('/api/get_group_membership_changes', methods=['GET'])
def get_group_membership_changes():
    """
    Fetch membership gains, losses, and net differential for channels in a group.

    Args:
        channel_group (str, required): VTuber group name (query parameter)
        month (str, required): Month in YYYY-MM format (query parameter)

    Returns:
        Success (200): JSON array with gains_count, losses_count, and differential per channel
        Error (400): Missing required parameters
        Error (500): Database query failed
    """
    cursor = g.db_conn.cursor()

    channel_group = request.args.get('channel_group')
    month = request.args.get('month')

    redis_key = f"group_membership_changes_{channel_group}_{month}"
    cached_data = g.redis_conn.get(redis_key)
    if cached_data:
        inc_cache_hit_count()
        return jsonify(json.loads(cached_data))
    inc_cache_miss_count()
    
    if not channel_group or not month:
        return jsonify({"error": _("Missing required parameters")}), 400

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
        cursor.execute(query, (channel_group, f"{month}-01", f"{month}-01", f"{month}-01"))
        results = cursor.fetchall()
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        cursor.close()

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


@api_bp.route('/api/get_group_streaming_hours_diff', methods=['GET'])
def get_group_streaming_hours_diff():
    """
    Fetches the change in streaming hours compared to the previous month.

    Args:
        month (str, required): Month in YYYY-MM format (query param)
        group (str, optional): Channel group name to filter by (query param)

    Returns:
        Success (200): JSON with "success" and "data" containing hours and change values
        Failure (400): Missing month parameter or invalid format
    """
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
    g.redis_conn.set(redis_key, json.dumps(output))

    return jsonify(output)


@api_bp.route('/api/get_chat_leaderboard', methods=['GET'])
def get_chat_leaderboard():
    """
    Fetch top 10 chatters by message count for a channel in a specific month.

    Args:
        channel_name (str, required): Channel name (query parameter)
        month (str, required): Month in YYYY-MM format (query parameter)

    Returns:
        Success (200): JSON array with user_name and message_count
        Error (400): Missing required parameters
        Error (404): No data found
    """
    channel_name = request.args.get('channel_name')
    month = request.args.get('month')

    redis_key = f"chat_leaderboard_{channel_name}_{month}"
    cached_data = g.redis_conn.get(redis_key)
    if cached_data:
        inc_cache_hit_count()
        return jsonify(json.loads(cached_data))
    inc_cache_miss_count()

    if not channel_name or not month:
        return jsonify({"error": _("Missing required parameters")}), 400

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


@api_bp.route('/api/get_user_changes', methods=['GET'])
def get_user_changes():
    """
    Fetch user gains and losses per channel for a group comparing to previous month.

    Args:
        group (str, required): Channel group name (query parameter)
        month (str, required): Month in YYYY-MM format (query parameter)

    Returns:
        Success (200): JSON array with users_gained, users_lost, and net_change per channel
        Error (400): Missing required parameters
    """
    channel_group = request.args.get('group')
    month = request.args.get('month')

    if not (channel_group and month):
        return jsonify({"error": _("Missing required parameters")}), 400
    
    redis_key = f"user_changes_{channel_group}_{month}"
    cached_data = g.redis_conn.get(redis_key)
    if cached_data:
        inc_cache_hit_count()
        return jsonify(json.loads(cached_data))
    inc_cache_miss_count()

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

    filtered_results = [
        {
            "channel": row[0],
            "users_gained": row[1],
            "users_lost": row[2],
            "net_change": row[3]
        }
        for row in results
        if row[1] > 0 and row[2] > 0
    ]

    g.redis_conn.set(redis_key, json.dumps(filtered_results))

    return jsonify(filtered_results)


@api_bp.route('/api/get_exclusive_chat_users', methods=['GET'])
def get_exclusive_chat_users():
    """
    Calculate percentage of users who only chat in a specific channel within its group.

    Args:
        channel (str, required): Channel name (query parameter)

    Returns:
        Success (200): JSON array with month and exclusive user percentage
        Error (400): Missing channel parameter or invalid channel
    """
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


@api_bp.route('/api/get_message_type_percents', methods=['GET'])
def get_message_type_percents():
    """
    Fetch language-specific message percentages and rates per minute for a channel.

    Args:
        channel (str, required): Channel name (query parameter)
        language (str, required): Language code: EN, JP, KR, or RU (query parameter)

    Returns:
        Success (200): JSON array with month, percent, and message_rate
        Error (400): Missing parameters or invalid language code
    """
    channel_name = request.args.get('channel')
    language = request.args.get('language').upper()

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

    language_column_map = {
        "EN": "es_en_id_count",
        "JP": "jp_count",
        "KR": "kr_count",
        "RU": "ru_count"
    }

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


@api_bp.route('/api/get_attrition_rates', methods=['GET'])
def get_attrition_rates():
    """
    Calculate retention rates of top 1000 chatters over time after baseline period.

    Args:
        channel (str, required): Channel name (query parameter)
        month (str, optional): Baseline month in YYYY-MM format (query parameter)
        announce_date (str, optional): Announcement date YYYY-MM-DD for graduation mode (query parameter)
        graduation_date (str, optional): Graduation date YYYY-MM-DD for graduation mode (query parameter)

    Returns:
        Success (200): JSON array with month and percent of users still active
        Error (400): Missing channel or invalid parameter combination
        Error (404): No top chatters found
        Error (500): Server error
    """
    channel_name = request.args.get('channel')
    month = request.args.get('month')
    announce_date = request.args.get('announce_date')
    graduation_date = request.args.get('graduation_date')

    if not channel_name:
        return jsonify({"error": "Missing required parameter: channel"}), 400

    try:
        if announce_date and graduation_date:
            announce_dt = datetime.strptime(announce_date, "%Y-%m-%d")
            graduation_dt = datetime.strptime(graduation_date, "%Y-%m-%d")
            baseline_month = (announce_dt - relativedelta(months=1)).strftime("%Y-%m")
            start_from_month = (graduation_dt + relativedelta(months=1)).strftime("%Y-%m")
            redis_key = f"attrition_rates_{channel_name}_{baseline_month}_{start_from_month}"

        elif month:
            baseline_month = month
            start_from_month = (datetime.strptime(month, "%Y-%m") + relativedelta(months=1)).strftime("%Y-%m")
            redis_key = f"attrition_rates_{channel_name}_{baseline_month}"
        else:
            return jsonify({"error": "Must provide either month or both announce_date and graduation_date"}), 400

        cached_data = g.redis_conn.get(redis_key)
        if cached_data:
            inc_cache_hit_count()
            return jsonify(json.loads(cached_data))
        inc_cache_miss_count()

        end_date = datetime.strptime(baseline_month, "%Y-%m") + relativedelta(months=1)
        start_date = end_date - relativedelta(months=3)
        end_date_str = end_date.strftime("%Y-%m-01")
        start_date_str = start_date.strftime("%Y-%m-01")

        cursor = g.db_conn.cursor()

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

        results = []
        current_month = datetime.strptime(start_from_month, "%Y-%m")
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

            current_month += relativedelta(months=1)

        cursor.close()
        g.redis_conn.set(redis_key, json.dumps(results))
        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route('/api/get_jp_user_percent', methods=['GET'])
def get_jp_user_percent():
    """
    Returns percentage of users with >50% Japanese messages per month.

    Args:
        channel (str, required): Channel name (query param)

    Returns:
        Success (200): JSON array with month and jp_user_percent
        Failure (400): Missing channel parameter
    """
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

    response = [{"month": row[0], "jp_user_percent": float(row[1])} for row in results]

    g.redis_conn.set(redis_key, json.dumps(response))

    return jsonify(response)


@api_bp.route('/api/get_latest_updates', methods=['GET'])
def get_latest_updates():
    """
    Fetches the latest news updates from the news.txt file.

    Args:
        None

    Returns:
        Success (200): JSON array with date and message objects
        Failure (500): File read error
    """
    try:
        news_list = []
        with open("news.txt", "r") as file:
            for line in file:
                if ": " in line:
                    date, message = line.split(": ", 1)
                    news_list.append({"date": date.strip(), "message": message.strip()})

        return jsonify(news_list)
    except FileNotFoundError:
        return jsonify([])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route('/api/get_channel_names', methods=['GET'])
def get_channel_names():
    """
    Fetch all channel names from database sorted alphabetically.

    Args:
        None

    Returns:
        Success (200): JSON array of channel name strings
    """
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


@api_bp.route('/api/get_date_ranges', methods=['GET'])
def get_date_ranges():
    """
    Fetch earliest and latest video end times from database for videos with chat logs.

    Args:
        None

    Returns:
        Success (200): JSON array with [min_date, max_date] as strings
    """
    redis_key = "date_ranges"
    cached_data = g.redis_conn.get(redis_key)
    if cached_data:
        inc_cache_hit_count()
        return jsonify(json.loads(cached_data))
    inc_cache_miss_count()
    
    cursor = g.db_conn.cursor()
    cursor.execute("SELECT MIN(end_time), MAX(end_time) FROM videos WHERE has_chat_log = 't'")
    
    date_range = cursor.fetchone()
    cursor.close()

    output = [
        str(date_range[0]),
        str(date_range[1])
    ]
    
    g.redis_conn.set(redis_key, json.dumps(output))
    return jsonify(output)


@api_bp.route('/api/get_number_of_chat_logs', methods=['GET'])
def get_number_of_chat_logs():
    """
    Fetches the total count of videos with chat logs.

    Args:
        None

    Returns:
        Success (200): JSON number representing total chat log count
    """
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


@api_bp.route('/api/get_num_messages', methods=['GET'])
def get_num_messages():
    """
    Fetches the total number of messages across all user data.

    Args:
        None

    Returns:
        Success (200): JSON number representing total message count
    """
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


@api_bp.route('/api/get_funniest_timestamps', methods=['GET'])
def get_funniest_timestamps():
    """
    Fetches funniest moment timestamps for videos in a channel and month.

    Args:
        channel (str, required): Channel name (query param)
        month (str, required): Month in YYYY-MM format (query param)

    Returns:
        Success (200): JSON array with title, video_id, and relative timestamp
        Failure (400): Missing required parameters
    """
    channel_name = request.args.get('channel')
    month = request.args.get('month')

    if not (channel_name and month):
        return jsonify({"error": "Missing required parameters"}), 400
    
    redis_key = f"funniest_timestamps_{channel_name}_{month}"
    cached_data = g.redis_conn.get(redis_key)
    if cached_data:
        inc_cache_hit_count()
        return jsonify(json.loads(cached_data))
    inc_cache_miss_count()

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


@api_bp.route('/api/get_user_info', methods=['GET'])
def get_user_info():
    """
    Fetch user's chat activity across channels with message counts and percentile rankings.

    Args:
        identifier (str, required): User ID or @username handle (query parameter)
        month (str, required): Month in YYYY-MM format (query parameter)

    Returns:
        Success (200): JSON with success=True and data array of channel activity
        Error (400): Missing required parameters
        Error (404): User not found
    """
    user_id = request.args.get('identifier')
    month = request.args.get('month')

    if not (user_id and month):
        return jsonify({"success": False, "error": "Missing required parameters: 'user_id' and 'month' are required."}), 400
    
    redis_key = f"user_info_{user_id}_{month}"
    cached_data = g.redis_conn.get(redis_key)
    if cached_data:
        inc_cache_hit_count()
        return jsonify(json.loads(cached_data))
    inc_cache_miss_count()

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
    g.redis_conn.set(redis_key, json.dumps(output))

    return jsonify(output)


@api_bp.route('/api/get_chat_engagement', methods=['GET'])
def get_chat_engagement():
    """
    Fetches chat engagement statistics including average messages per user.

    Args:
        month (str, optional): Month in YYYY-MM format, defaults to current (query param)
        group (str, optional): Channel group name to filter by (query param)

    Returns:
        Success (200): JSON with "success" and "data" containing engagement metrics
        Failure (500): Database query error
    """
    month = request.args.get('month', datetime.utcnow().strftime('%Y-%m'))
    group = request.args.get('group', None)

    redis_key = f"chat_engagement_{month}_{group}"
    cached_data = g.redis_conn.get(redis_key)
    if cached_data:
        inc_cache_hit_count()
        return jsonify(json.loads(cached_data))
    inc_cache_miss_count()

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


@api_bp.route('/api/get_video_highlights', methods=['GET'])
def get_video_highlights():
    """
    Fetch AI-generated video highlights with timestamps for a channel and month.

    Args:
        channel_name (str, required): Channel name (query parameter)
        month (str, required): Month in YYYY-MM format (query parameter)

    Returns:
        Success (200): JSON with success=True and data array grouped by video
        Error (400): Missing parameters or invalid month format
        Error (500): Internal server error
    """
    channel_name = request.args.get('channel_name')
    month_str = request.args.get('month')

    if not channel_name or not month_str:
        return jsonify({"success": False, "error": "Missing required parameters: 'channel_name' and 'month' are required."}), 400

    try:
        month_start_dt = datetime.strptime(month_str, '%Y-%m')
        month_start_sql = month_start_dt.strftime('%Y-%m-01')
    except ValueError:
        return jsonify({"success": False, "error": "Invalid month format. Please use 'YYYY-MM'."}), 400

    redis_key = f"video_highlights_{channel_name.replace(' ', '_')}_{month_str}"
    try:
        cached_data = g.redis_conn.get(redis_key)
        if cached_data:
             inc_cache_hit_count()
             return jsonify(json.loads(cached_data))
        inc_cache_miss_count()
    except Exception as e:
        print(f"Redis cache check failed: {e}")

    query = """
        SELECT
            vh.video_id,
            v.title,
            vh.topic_tag,
            vh.generated_summary,
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
            output = {"success": True, "data": []}
            g.redis_conn.set(redis_key, json.dumps(output))
            return jsonify(output)

        videos_dict = {}
        for row in results:
            video_id, title, topic, summary, relative_seconds = row
            if relative_seconds < 0: continue
            
            if video_id not in videos_dict:
                videos_dict[video_id] = { "video_id": video_id, "video_title": title, "timestamps": [] }
            
            videos_dict[video_id]["timestamps"].append({
                "topic": topic,
                "summary": summary,
                "youtube_url": f"https://www.youtube.com/watch?v={video_id}&t={int(relative_seconds)}s"
            })

        data = list(videos_dict.values())
        output = {"success": True, "data": data}

        g.redis_conn.set(redis_key, json.dumps(output))

        return jsonify(output)

    except Exception as e:
        print(f"An error occurred in get_video_highlights: {e}")
        return jsonify({"success": False, "error": "An internal server error occurred."}), 500


@api_bp.route('/api/search_highlights', methods=['GET'])
def search_highlights():
    """
    Searches highlights using vector similarity with optional filters.

    Args:
        query (str, required): Search query with optional operators (query param)
            Operators: channel:<name>, from:<YYYY-MM-DD>, to:<YYYY-MM-DD>

    Returns:
        Success (200): JSON with "success" and "data" array of matching highlights
        Failure (400): Missing query or empty after parsing
        Failure (500): Internal server error
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


@api_bp.route('/api/search_merchandise', methods=['GET'])
def search_merchandise():
    """
    Search shops for products matching VTuber name.

    Args:
        vtuber_name (str, required): VTuber name to search (query parameter)
        language (str, optional): Shop language "en" or "jp", default "en" (query parameter)

    Returns:
        Success (200): JSON with success=True and results array of products
        Error (400): Missing vtuber_name or invalid language
        Error (404): No products found
        Error (500): Failed to fetch or parse shop data
    """
    vtuber_name = request.args.get('vtuber_name')
    language = request.args.get('language', 'en')

    if not vtuber_name:
        return jsonify({"success": False, "error": "Missing required parameter: vtuber_name"}), 400

    if language not in ['en', 'jp']:
        return jsonify({"success": False, "error": "Invalid language parameter. Must be 'en' or 'jp'."}), 400

    base_url = "https://shop.hololivepro.com"
    lang_path = "/en" if language == "en" else ""

    query_params = {
        "q": vtuber_name,
        "options[prefix]": "last",
        "filter.p.m.sales.status": "販売中",
        "sort_by": "relevance"
    }
    search_url = f"{base_url}{lang_path}/search?{urllib.parse.urlencode(query_params)}"

    try:
        response = requests.get(search_url, timeout=15)
        response.raise_for_status()
    except Exception as e:
        return jsonify({"success": False, "error": f"Failed to fetch Hololive shop: {e}"}), 500

    match = re.search(
        r'<script id="web-pixels-manager-setup"[^>]*>\s*(.*?)\s*</script>',
        response.text,
        re.DOTALL
    )
    if not match:
        return jsonify({"success": False, "error": "Could not find web-pixels-manager-setup script"}), 500

    script_content = match.group(1)

    events_start = script_content.find('"events":"')
    
    if events_start == -1:
        return jsonify({"success": False, "error": "Could not find events data"}), 404

    i = events_start + len('"events":"')
    
    events_str = ""
    while i < len(script_content):
        char = script_content[i]
        
        if char == '\\' and i + 1 < len(script_content):
            events_str += char + script_content[i + 1]
            i += 2
        elif char == '"':
            break
        else:
            events_str += char
            i += 1
    
    if not events_str:
        return jsonify({"success": False, "error": "Could not extract events string"}), 404

    try:
        events_str_decoded = json.loads('"' + events_str + '"')
        events = json.loads(events_str_decoded)
        
        product_variants = []
        for event in events:
            if isinstance(event, list) and len(event) >= 2:
                event_name = event[0]
                event_data = event[1] if len(event) > 1 else {}
                
                if event_name == "search_submitted":
                    search_result = event_data.get("searchResult", {})
                    product_variants = search_result.get("productVariants", [])
                    break
        
        if not product_variants:
            return jsonify({"success": False, "error": "No product variants found in search results"}), 404
            
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Error at position: {e.pos if hasattr(e, 'pos') else 'unknown'}")
        if 'events_str_decoded' in locals():
            error_pos = getattr(e, 'pos', 2839)
            print(f"Context around error: {events_str_decoded[max(0, error_pos-100):error_pos+100]}")
        return jsonify({"success": False, "error": f"Failed to parse events JSON: {e}"}), 500
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Failed to extract data: {e}"}), 500

    results = []
    for p in product_variants:
        product = p.get("product", {})
        image = p.get("image", {})
        price = p.get("price", {})
        results.append({
            "title": product.get("title"),
            "vendor": product.get("vendor"),
            "url": f"{base_url}{product.get('url')}" if product.get('url') else None,
            "price": f"{price.get('amount')} {price.get('currencyCode')}" if price.get('amount') else None,
            "image": f"https:{image.get("src")}" if image.get("src") else None,
            "sku": p.get("sku"),
            "variant_title": p.get("title"),
            "type": product.get("type"),
        })

    return jsonify({
        "success": True,
        "query": vtuber_name,
        "language": language,
        "count": len(results),
        "results": results
    })