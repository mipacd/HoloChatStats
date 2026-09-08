import json
import os
import re
import urllib.parse
from datetime import datetime, timezone, timedelta
from flask import Blueprint, request, jsonify, g, current_app
from flask_babel import _
from dateutil.relativedelta import relativedelta
import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import leidenalg as la
import igraph as ig
import numpy as np
import requests
import boto3
import time
import math
import logging
from scipy.sparse import csr_matrix
from googleapiclient.errors import HttpError
from sklearn.feature_extraction.text import TfidfVectorizer
from sqlalchemy import func, extract, cast, and_, case, or_, literal_column, collate, text
from sqlalchemy.orm import aliased
from sqlalchemy.types import Numeric, BigInteger, Date
from zoneinfo import ZoneInfo
try:
    from web.utils import (
        cached_json, parse_month, month_range, resolve_user_id, get_or_compute_cached,
        inc_cache_hit_count, inc_cache_miss_count,
        streaming_hours_query, parse_search_query, EMBEDDER, load_channel_mapping,
        get_youtube_service, fetch_live_and_upcoming_streams, fetch_past_videos
    )
    from web.models import (
        db, User, UserData, Channel, Video,
        StreamingForecast, ChatLanguageStatsMv, MvUserMonthlyActivity,
        MembershipDataSummary, MvUserActivity, VideoHighlight
    )
except ImportError:
    from utils import (
        cached_json, parse_month, month_range, resolve_user_id, get_or_compute_cached,
        inc_cache_hit_count, inc_cache_miss_count,
        streaming_hours_query, parse_search_query, EMBEDDER, load_channel_mapping,
        get_youtube_service, fetch_live_and_upcoming_streams, fetch_past_videos
    )
    from models import (
        db, User, UserData, Channel, Video,
        StreamingForecast, ChatLanguageStatsMv, MvUserMonthlyActivity,
        MembershipDataSummary, MvUserActivity, VideoHighlight
    )

from utils import (
    inc_cache_hit_count, inc_cache_miss_count,
    streaming_hours_query, parse_search_query, EMBEDDER, load_channel_mapping,
    get_youtube_service, fetch_live_and_upcoming_streams, fetch_past_videos
)

api_bp = Blueprint('api', __name__)
logger = logging.getLogger(__name__)



@api_bp.route('/api/get_channel_streams', methods=['GET', 'POST'])
def get_channel_streams():
    """
    Get past, current, or upcoming streams/videos for one or more VTuber channels.
    Includes live streams, premieres, and regular video uploads.
    
    Query Parameters / JSON Body:
        channel (str): Channel name or comma-separated list of names (required)
        stream_type (str): 'past', 'live', 'upcoming', or 'all' (default: 'all')
        limit (int): Maximum items to return per channel (default: 5)
    
    Returns:
        JSON response with stream/video data for each requested channel
    """
    # Get parameters from request
    if request.method == 'POST':
        data = request.get_json() or {}
    else:
        data = request.args.to_dict()
    
    channel_names = data.get('channel', '')
    stream_type = data.get('stream_type', 'all').lower()
    
    try:
        limit = int(data.get('limit', 5))
        limit = max(1, min(limit, 50))  # Clamp between 1 and 50
    except (ValueError, TypeError):
        limit = 5
    
    # Validate required parameters
    if not channel_names:
        return jsonify({
            'success': False,
            'error': 'Channel name is required'
        }), 400
    
    # Validate stream_type
    valid_stream_types = ['past', 'live', 'upcoming', 'all']
    if stream_type not in valid_stream_types:
        return jsonify({
            'success': False,
            'error': f'Invalid stream_type. Must be one of: {", ".join(valid_stream_types)}'
        }), 400
    
    # Load channel mapping
    try:
        channel_mapping = load_channel_mapping()
    except FileNotFoundError:
        return jsonify({
            'success': False,
            'error': 'Channel configuration file not found'
        }), 500
    except json.JSONDecodeError:
        return jsonify({
            'success': False,
            'error': 'Invalid channel configuration file'
        }), 500
    
    # Initialize YouTube service
    try:
        youtube = get_youtube_service(current_app)
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    
    # Parse channel names (comma-separated)
    channels = [c.strip() for c in channel_names.split(',') if c.strip()]
    
    results = []
    
    for channel_name in channels:
        # Look up channel in mapping
        channel_info = channel_mapping.get(channel_name.lower())
        
        if not channel_info:
            # Try partial match
            partial_matches = [
                v for k, v in channel_mapping.items() 
                if channel_name.lower() in k
            ]
            if partial_matches:
                channel_info = partial_matches[0]
            else:
                results.append({
                    'channel': channel_name,
                    'error': f'Channel "{channel_name}" not found in database',
                    'streams': []
                })
                continue
        
        channel_id = channel_info['id']
        all_streams = []
        
        try:
            # Fetch based on stream_type
            if stream_type == 'live':
                streams = fetch_live_and_upcoming_streams(
                    youtube, channel_id, 'live', limit
                )
                all_streams.extend(streams)
                
            elif stream_type == 'upcoming':
                streams = fetch_live_and_upcoming_streams(
                    youtube, channel_id, 'upcoming', limit
                )
                all_streams.extend(streams)
                
            elif stream_type == 'past':
                videos = fetch_past_videos(youtube, channel_id, limit)
                all_streams.extend(videos)
                
            elif stream_type == 'all':
                # Fetch all types
                try:
                    live_streams = fetch_live_and_upcoming_streams(
                        youtube, channel_id, 'live', limit
                    )
                    all_streams.extend(live_streams)
                except HttpError as e:
                    print(f"Error fetching live streams for {channel_name}: {e}")
                
                try:
                    upcoming_streams = fetch_live_and_upcoming_streams(
                        youtube, channel_id, 'upcoming', limit
                    )
                    all_streams.extend(upcoming_streams)
                except HttpError as e:
                    print(f"Error fetching upcoming streams for {channel_name}: {e}")
                
                try:
                    past_videos = fetch_past_videos(youtube, channel_id, limit)
                    all_streams.extend(past_videos)
                except HttpError as e:
                    print(f"Error fetching past videos for {channel_name}: {e}")
            
            # Sort streams by most relevant criteria
            def get_sort_key(stream):
                # Prioritize by status, then by date
                status_priority = {
                    'live': 0,
                    'upcoming': 1,
                    'completed': 2
                }
                priority = status_priority.get(stream.get('status'), 3)
                
                # Get the most relevant timestamp
                timestamp = (
                    stream.get('scheduled_start') or 
                    stream.get('started_at') or 
                    stream.get('published_at') or 
                    ''
                )
                return (priority, timestamp)
            
            all_streams.sort(key=get_sort_key, reverse=True)
            
            # Remove duplicates based on video_id
            seen_ids = set()
            unique_streams = []
            for stream in all_streams:
                if stream['video_id'] not in seen_ids:
                    seen_ids.add(stream['video_id'])
                    unique_streams.append(stream)
            
            # Limit total results
            unique_streams = unique_streams[:limit]
            
            results.append({
                'channel': channel_info['name'],
                'organization': channel_info['organization'],
                'channel_id': channel_id,
                'streams': unique_streams
            })
            
        except HttpError as e:
            error_reason = e.resp.get('status', 'Unknown')
            results.append({
                'channel': channel_info['name'],
                'error': f'YouTube API error: {error_reason}',
                'streams': []
            })
        except Exception as e:
            results.append({
                'channel': channel_info['name'],
                'error': f'Unexpected error: {str(e)}',
                'streams': []
            })
    
    return jsonify({
        'success': True,
        'data': results
    })

@api_bp.route('/api/get_channel_metrics', methods=['GET', 'POST'])
def get_channel_metrics():
    """
    Get channel metrics including subscriber count, total views, and video count
    for one or more VTuber channels.
    
    Query Parameters / JSON Body:
        channel (str): Channel name or comma-separated list of names (required)
    
    Returns:
        JSON response with channel metrics for each requested channel
    """
    # Get parameters from request
    if request.method == 'POST':
        data = request.get_json() or {}
    else:
        data = request.args.to_dict()
    
    channel_names = data.get('channel', '')
    
    # Validate required parameters
    if not channel_names:
        return jsonify({
            'success': False,
            'error': 'Channel name is required'
        }), 400
    
    # Load channel mapping
    try:
        channel_mapping = load_channel_mapping()
    except FileNotFoundError:
        return jsonify({
            'success': False,
            'error': 'Channel configuration file not found'
        }), 500
    except json.JSONDecodeError:
        return jsonify({
            'success': False,
            'error': 'Invalid channel configuration file'
        }), 500
    
    # Initialize YouTube service
    try:
        youtube = get_youtube_service(current_app)
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    
    # Parse channel names (comma-separated)
    channels = [c.strip() for c in channel_names.split(',') if c.strip()]
    
    results = []
    
    # Collect all valid channel IDs for batch request
    channel_lookup = {}  # Maps channel_id to channel_info
    
    for channel_name in channels:
        # Look up channel in mapping
        channel_info = channel_mapping.get(channel_name.lower())
        
        if not channel_info:
            # Try partial match
            partial_matches = [
                v for k, v in channel_mapping.items() 
                if channel_name.lower() in k
            ]
            if partial_matches:
                channel_info = partial_matches[0]
            else:
                results.append({
                    'channel': channel_name,
                    'error': f'Channel "{channel_name}" not found in database'
                })
                continue
        
        channel_lookup[channel_info['id']] = channel_info
    
    if not channel_lookup:
        return jsonify({
            'success': True,
            'data': results
        })
    
    try:
        # Batch request for all channels
        channel_ids = list(channel_lookup.keys())
        
        # YouTube API allows up to 50 channel IDs per request
        batch_size = 50
        all_channel_data = []
        
        for i in range(0, len(channel_ids), batch_size):
            batch_ids = channel_ids[i:i + batch_size]
            
            channel_response = youtube.channels().list(
                part='snippet,statistics,brandingSettings',
                id=','.join(batch_ids)
            ).execute()
            
            all_channel_data.extend(channel_response.get('items', []))
        
        # Process each channel's data
        for channel_data in all_channel_data:
            channel_id = channel_data['id']
            channel_info = channel_lookup.get(channel_id)
            
            if not channel_info:
                continue
            
            snippet = channel_data.get('snippet', {})
            statistics = channel_data.get('statistics', {})
            
            # Get subscriber count (may be hidden)
            subscriber_count = None
            subscriber_hidden = statistics.get('hiddenSubscriberCount', False)
            
            if not subscriber_hidden:
                sub_count_str = statistics.get('subscriberCount')
                if sub_count_str:
                    subscriber_count = int(sub_count_str)
            
            # Get total view count
            total_views = None
            view_count_str = statistics.get('viewCount')
            if view_count_str:
                total_views = int(view_count_str)
            
            # Get video count
            video_count = None
            video_count_str = statistics.get('videoCount')
            if video_count_str:
                video_count = int(video_count_str)
            
            # Get channel thumbnail (prefer high quality)
            thumbnails = snippet.get('thumbnails', {})
            thumbnail_url = (
                thumbnails.get('high', {}).get('url') or
                thumbnails.get('medium', {}).get('url') or
                thumbnails.get('default', {}).get('url') or
                ''
            )
            
            # Get description (truncate if too long)
            description = snippet.get('description', '')
            if len(description) > 500:
                description = description[:497] + '...'
            
            result = {
                'channel': channel_info['name'],
                'organization': channel_info['organization'],
                'channel_id': channel_id,
                'custom_url': snippet.get('customUrl', ''),
                'description': description,
                'thumbnail': thumbnail_url,
                'created_at': snippet.get('publishedAt', ''),
                'subscriber_count': subscriber_count,
                'subscriber_count_hidden': subscriber_hidden,
                'total_view_count': total_views,
                'video_count': video_count
            }
            
            results.append(result)
        
        # Check for any channels that weren't found in the API response
        found_ids = {item['channel_id'] for item in results if 'channel_id' in item}
        for channel_id, channel_info in channel_lookup.items():
            if channel_id not in found_ids:
                results.append({
                    'channel': channel_info['name'],
                    'error': f'Channel data not found on YouTube'
                })
        
    except HttpError as e:
        error_reason = e.resp.get('status', 'Unknown')
        error_content = e.content.decode('utf-8') if hasattr(e, 'content') else str(e)
        return jsonify({
            'success': False,
            'error': f'YouTube API error: {error_reason}',
            'details': error_content
        }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Unexpected error: {str(e)}'
        }), 500
    
    return jsonify({
        'success': True,
        'data': results
    })

def _build_similarity_graph(channel_names, similarity_matrix, percentile):
    """
    Build a NetworkX graph connecting channels whose pairwise similarity
    exceeds the given percentile threshold of the similarity matrix.
    """
    G = nx.Graph()
    threshold = np.percentile(similarity_matrix, float(percentile))
    n = len(channel_names)
    for i in range(n):
        for j in range(i + 1, n):
            weight = similarity_matrix[i, j]
            if weight > threshold:
                G.add_edge(channel_names[i], channel_names[j], weight=weight)
    return G


def _detect_communities(G):
    """Run Leiden community detection on G; returns community ids aligned
    with G.nodes() iteration order, for use as node marker colors."""
    g_igraph = ig.Graph.from_networkx(G)
    partition = la.find_partition(
        g_igraph,
        la.RBConfigurationVertexPartition,
        weights='weight',
        resolution_parameter=1.0,
    )
    partition_dict = {
        g_igraph.vs[node]['_nx_name']: partition.membership[node]
        for node in range(g_igraph.vcount())
    }
    return [partition_dict[node] for node in G.nodes]


def _compute_graph_layout(G, community_ids, dim=2):
    """
    Compute ForceAtlas2 layout for G, then post-process for readability.
    Two adjustments are applied after the base layout converges:
    1. Cluster expansion: each community's nodes are pushed outward from their
       shared centroid, increasing intra-cluster spacing so labels don't pile
       on top of each other. The centroids themselves don't move, so the
       overall community structure is preserved.
    2. Low-degree nudge: degree-1 and degree-2 nodes (typically bridges between
       clusters) are pulled toward the centroid of their neighbors. This
       prevents FA2's global repulsion from flinging them to the far side of
       the viewport, away from the very nodes they're connected to.
    Both EXPANSION_FACTOR and the pull strengths below are tuning knobs --
    adjust based on how the graph looks with your actual channel count.
    """
    pos = nx.forceatlas2_layout(
        G,
        dim=dim,
        max_iter=300,
        scaling_ratio=2.0,
        gravity=1.0,
        strong_gravity=False,
        jitter_tolerance=1.0,
    )
    nodes = list(G.nodes())
    # ── Stage 1: expand clusters ──
    communities = {}
    for node, comm_id in zip(nodes, community_ids):
        communities.setdefault(comm_id, []).append(node)
    EXPANSION_FACTOR = 4
    for members in communities.values():
        if len(members) <= 1:
            continue
        centroid = np.mean([pos[n] for n in members], axis=0)
        for node in members:
            node_pos = np.array(pos[node])
            pos[node] = tuple(centroid + EXPANSION_FACTOR * (node_pos - centroid))
    # ── Stage 2: pull leaf / bridge nodes toward their neighbors ──
    for node in nodes:
        degree = G.degree(node)
        if degree > 2:
            continue
        neighbors = list(G.neighbors(node))
        neighbor_centroid = np.mean([pos[n] for n in neighbors], axis=0)
        node_pos = np.array(pos[node])
        pull_strength = 0.6 if degree == 1 else 0.3
        pos[node] = tuple(node_pos + pull_strength * (neighbor_centroid - node_pos))
    return pos


def _build_similarity_payload(G, pos, community_colors, graph_type, title):
    """
    Build a JSON-serializable payload describing a channel similarity graph,
    for client-side rendering (react-force-graph) instead of Plotly.
    """
    is_3d = graph_type == "3d"
    nodes = list(G.nodes())
    community_map = dict(zip(nodes, community_colors))
    node_payload = []
    for node in nodes:
        coords = pos[node]
        entry = {
            "id": node,
            "community": int(community_map[node]),
            "degree": G.degree(node),
            "neighbors": list(G.neighbors(node)),
            "x": float(coords[0]),
            "y": float(coords[1]),
        }
        if is_3d:
            entry["z"] = float(coords[2])
        node_payload.append(entry)
    link_payload = [
        {"source": u, "target": v, "weight": float(G[u][v]["weight"])}
        for u, v in G.edges()
    ]
    return {
        "title": title,
        "is_3d": is_3d,
        "nodes": node_payload,
        "links": link_payload,
    }


@api_bp.route('/api/channel_clustering', methods=['GET'])
@cached_json(lambda: f"channel_clustering_vxd_{request.args.get('month')}_{request.args.get('percentile', '95')}_{request.args.get('type', '2d')}")
def channel_clustering():
    filter_month = request.args.get("month")
    percentile = request.args.get("percentile", "95")
    graph_type = request.args.get("type", "2d")
    if not filter_month:
        return jsonify({"error": "Month filter (e.g., '2025-03') is required"}), 400
    try:
        filter_month_start = parse_month(filter_month)
        rows = (
            db.session.query(
                UserData.user_id,
                Channel.channel_name,
                func.sum(UserData.total_message_count).label("message_weight"),
            )
            .join(Channel, UserData.channel_id == Channel.channel_id)
            .filter(
                func.date_trunc("month", UserData.last_message_at) == filter_month_start,
                UserData.total_message_count > 0,
            )
            .group_by(UserData.user_id, Channel.channel_name)
            .all()
        )
        data = pd.DataFrame(rows, columns=['user_id', 'channel_name', 'message_weight'])
        if data.empty:
            return jsonify({"error": "No data found for the specified month"}), 404
        user_channel_matrix = data.pivot(index='user_id', columns='channel_name', values='message_weight').fillna(0)
        similarity_matrix = cosine_similarity(user_channel_matrix.T)
        channel_names = list(user_channel_matrix.columns)
        G = _build_similarity_graph(channel_names, similarity_matrix, percentile)
        if G.number_of_nodes() == 0:
            return jsonify({"error": "No connections found with current percentile threshold"}), 404
        community_colors = _detect_communities(G)
        dim = 3 if graph_type == '3d' else 2                                 
        pos = _compute_graph_layout(G, community_colors, dim=dim)             
        formatted_month = filter_month_start.strftime("%B %Y")
        title = f"Channel User Similarity Graph{' (3D)' if graph_type == '3d' else ''} for {formatted_month}"
        payload = _build_similarity_payload(G, pos, community_colors, graph_type, title)
        return payload
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@api_bp.route('/api/content_clustering', methods=['GET'])
@cached_json(lambda: f"content_clustering_v2x_{request.args.get('month')}_{request.args.get('percentile', '95')}_{request.args.get('type', '2d')}")
def content_clustering():
    filter_month = request.args.get("month")
    percentile = request.args.get("percentile", "95")
    graph_type = request.args.get("type", "2d")
    if not filter_month:
        return jsonify({"error": "Month filter (e.g., '2025-03') is required"}), 400
    try:
        filter_month_start = parse_month(filter_month)
        rows = (
            db.session.query(Video.video_id, Channel.channel_name, Video.title)
            .join(Channel, Video.channel_id == Channel.channel_id)
            .filter(
                func.date_trunc("month", Video.end_time) == filter_month_start,
                Video.title.isnot(None),
                Video.title != "",
            )
            .group_by(Video.video_id, Channel.channel_name, Video.title)
            .all()
        )
        if not rows:
            return jsonify({"error": "No data found for the specified month"}), 404
        data = pd.DataFrame(rows, columns=['video_id', 'channel_name', 'title'])
        channel_titles = data.groupby('channel_name')['title'].apply(lambda x: ' '.join(x)).to_dict()
        channel_titles = {k: v for k, v in channel_titles.items() if len(v.strip()) > 0}
        if len(channel_titles) < 2:
            return jsonify({"error": "Insufficient channels with content for clustering"}), 404
        channel_names = list(channel_titles.keys())
        title_corpus = [channel_titles[channel] for channel in channel_names]
        vectorizer = TfidfVectorizer(max_features=500, stop_words='english', ngram_range=(1, 2), min_df=1)
        tfidf_matrix = vectorizer.fit_transform(title_corpus)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        G = _build_similarity_graph(channel_names, similarity_matrix, percentile)
        if G.number_of_nodes() == 0:
            return jsonify({"error": "No connections found with current percentile threshold"}), 404
        community_colors = _detect_communities(G)
        dim = 3 if graph_type == '3d' else 2                                 
        pos = _compute_graph_layout(G, community_colors, dim=dim)            
        formatted_month = filter_month_start.strftime("%B %Y")
        title = f"Channel Content Similarity Graph{' (3D)' if graph_type == '3d' else ''} for {formatted_month}"
        payload = _build_similarity_payload(G, pos, community_colors, graph_type, title)
        return payload
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
MAX_COMMUNITIES = 3         # module-level constant, not user-facing
# ── Channel layout ────────────────────────────────────────────────────────
CH_PERSONAL_MIN = 5.0      # world-space "personal space" of the smallest channel
CH_PERSONAL_MAX = 16.0     # ...and of the largest
CH_GAP_INTRA    = 6.0      # extra clearance between same-community channels
CH_GAP_INTER    = 14.0     # ...and across communities
CH_PULL         = 0.12
CH_PULL_STEPS   = 18
CH_RELAX_STEPS  = 45       # pure separation pass — guarantees the min distance holds
CH_MIN_EXTENT   = 200.0
CH_PACKING      = 2.4      # how loosely the discs are packed into the layout disc
# ── User placement ────────────────────────────────────────────────────────
U_GAMMA         = 2.5      # sharpening of the centroid toward the dominant channel
U_LOYALTY_POW   = 1.0      # exponent on "share of this user's messages"
U_VOLUME_POW    = 0.75     # exponent on "how heavy a commenter they are here"
U_ORBIT_MIN     = 1.2      # world units — maximally loyal + high volume
U_ORBIT_MAX     = 14.0     # world units — drive-by commenter
U_TERRITORY     = 0.55     # never orbit past this fraction of the way to a foreign channel
U_JITTER        = 0.18     # fraction of the orbit radius
U_KEEPOUT       = 0.60     # fraction of a channel's personal radius that non-members must clear
U_MEMBER_BUFFER = 0.5      # don't let a dot sit exactly on a channel centre
U_VOL_PCTL      = 95       # per-channel reference volume
U_CHUNK         = 32768

_MAX_CANDIDATES = 12

def _channel_sizes(ch_degree):
    """Map user-count → (personal space radius, inertia).  Log-scaled: a
    200 k-user channel should not claim 4000× the space of a 50-user one."""
    m = np.log1p(np.asarray(ch_degree, dtype=np.float64))
    spread = m.max() - m.min()
    t = (m - m.min()) / spread if spread > 1e-9 else np.full_like(m, 0.5)
    personal = CH_PERSONAL_MIN + t * (CH_PERSONAL_MAX - CH_PERSONAL_MIN)
    mass = 0.25 + 0.75 * t          # heavier channels move less when they collide
    return personal, mass

def _resolve_overlaps(x, y, min_d, mass, rng, iterations, damping=0.9):
    """
    Hard minimum-distance constraint, solved like d3-force's collide.
    Correction is split between the two channels in inverse proportion to
    their mass, so a big channel pushes a small one out of the way rather
    than meeting it halfway.
    """
    # w[i, j] = fraction of the correction absorbed by i  (→ 1 when j is heavy)
    w = mass[None, :] / (mass[:, None] + mass[None, :])
    eye = np.eye(len(x), dtype=bool)
    for _ in range(iterations):
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.hypot(dx, dy)
        # Keep the diagonal finite so no inf/inf or 0/0 is ever evaluated,
        # then mark it inactive explicitly.
        d[eye] = 1.0
        overlap = min_d - d
        overlap[eye] = -1.0
        active = overlap > 0.0
        if not active.any():
            break
        # coincident channels: pick an arbitrary separation axis
        coincide = active & (d < 1e-6)
        if coincide.any():
            ang = rng.uniform(0, 2 * np.pi, int(coincide.sum()))
            dx[coincide] = np.cos(ang)
            dy[coincide] = np.sin(ang)
            d[coincide] = 1.0
        # `where=` skips the inactive cells entirely → no invalid-value warning
        scale = np.zeros_like(d)
        np.divide(overlap, d, out=scale, where=active)
        x += damping * (scale * dx * w).sum(axis=1)
        y += damping * (scale * dy * w).sum(axis=1)
    return x, y

def _find_channel_communities(sim_matrix, n_channels, max_communities=MAX_COMMUNITIES, seed=42):
    """
    Binary-search the Leiden resolution on the channel-similarity graph
    so the number of communities is as close to max_communities as
    possible without exceeding it.  Much better results than running
    Leiden on the full 267 K-node bipartite graph and force-merging.
    """
    ri, ci = np.triu_indices(n_channels, k=1)
    vals = sim_matrix[ri, ci]
    keep = vals > 0
    g = ig.Graph(
        n=n_channels,
        edges=list(zip(ri[keep].tolist(), ci[keep].tolist())),
        directed=False,
    )
    g.es["weight"] = vals[keep].tolist()
    lo, hi = 0.001, 10.0
    best_mem, best_n = None, 0
    for _ in range(30):
        mid = (lo + hi) / 2
        part = la.find_partition(
            g, la.RBConfigurationVertexPartition,
            weights="weight", resolution_parameter=mid, seed=seed,
        )
        mem = np.asarray(part.membership, dtype=np.int32)
        n = len(np.unique(mem))
        if n <= max_communities:
            if best_mem is None or n > best_n:
                best_mem, best_n = mem.copy(), n
            lo = mid
        else:
            hi = mid
    if best_mem is None:
        part = la.find_partition(
            g, la.RBConfigurationVertexPartition,
            weights="weight", resolution_parameter=0.001, seed=seed,
        )
        best_mem = np.asarray(part.membership, dtype=np.int32)
    return best_mem

def _assign_user_communities(edge_u, edge_ch, edge_w, ch_communities, n_users):
    """Assign each user the community of its highest-weight channel."""
    sort_idx = np.lexsort((-edge_w, edge_u))
    sorted_u = edge_u[sort_idx]
    sorted_ch = edge_ch[sort_idx]
    _, first_idx = np.unique(sorted_u, return_index=True)
    return ch_communities[sorted_ch[first_idx]]
    
def _remap_communities_by_size(membership):
    """
    Renumber community IDs so that the largest community is 0, the
    next-largest is 1, and so on.  Returns (remapped_array, n_communities).
    """
    uniq, counts = np.unique(membership, return_counts=True)
    rank = np.empty_like(uniq)
    rank[np.argsort(-counts)] = np.arange(len(uniq))
    lut = np.empty(uniq.max() + 1, dtype=np.int32)
    lut[uniq] = rank
    return lut[membership], len(uniq)

def _layout_channels(sim_matrix, n_channels, ch_communities, ch_degree):
    """
    FR layout → normalise → {community pull + size-aware separation} →
    a final separation-only relaxation so the minimum distances actually hold.
    Minimum distance between i and j is
        personal_i + personal_j + (GAP_INTRA | GAP_INTER)
    so large channels keep a bigger bubble around themselves and
    cross-community pairs stay further apart than same-community pairs.
    """
    personal, mass = _channel_sizes(ch_degree)
    same = ch_communities[:, None] == ch_communities[None, :]
    min_d = personal[:, None] + personal[None, :] + np.where(same, CH_GAP_INTRA, CH_GAP_INTER)
    # Size the canvas so the constraints are actually satisfiable.
    eff = personal + CH_GAP_INTRA / 2.0
    extent = max(CH_MIN_EXTENT, 2.0 * math.sqrt(CH_PACKING * float(np.sum(eff ** 2))))
    ri, ci = np.triu_indices(n_channels, k=1)
    vals = sim_matrix[ri, ci]
    keep = vals > 0
    g = ig.Graph(
        n=n_channels,
        edges=list(zip(ri[keep].tolist(), ci[keep].tolist())),
        directed=False,
    )
    g.es["weight"] = vals[keep].tolist()
    coords = g.layout_fruchterman_reingold(weights="weight", niter=500)
    x = np.array([c[0] for c in coords], dtype=np.float64)
    y = np.array([c[1] for c in coords], dtype=np.float64)
    span = max(x.max() - x.min(), y.max() - y.min(), 1e-9)
    x = (x - x.mean()) / span * extent
    y = (y - y.mean()) / span * extent
    rng = np.random.default_rng(42)
    for _ in range(CH_PULL_STEPS):
        for cid in np.unique(ch_communities):
            m = np.where(ch_communities == cid)[0]
            if len(m) <= 1:
                continue
            x[m] += CH_PULL * (x[m].mean() - x[m])
            y[m] += CH_PULL * (y[m].mean() - y[m])
        _resolve_overlaps(x, y, min_d, mass, rng, iterations=2)
    # Pull is off from here on, so nothing can re-introduce an overlap.
    _resolve_overlaps(x, y, min_d, mass, rng, iterations=CH_RELAX_STEPS, damping=0.85)
    x -= x.mean()
    y -= y.mean()
    return x, y, personal

def _position_users(uc_sparse, ch_x, ch_y, personal_r, n_users):
    """
    Orbit model.
    Every user is anchored at a *sharpened* centroid of the channels they
    actually commented in (share ** GAMMA), then pushed out along a random
    ray by an orbit radius that shrinks with their affinity to that anchor:
        loyalty  = w_max / w_total                 (share of their messages)
        volume   = log1p(w_max) / log1p(p95_j)     (heavy commenter *here*?)
        affinity = loyalty**a * volume**b           (AND, not OR)
        radius   = ORBIT_MIN + (ORBIT_MAX - ORBIT_MIN) * (1 - affinity)
    200 on A + 1 on B  → loyalty 0.995, volume ~1   → hugs A
      1 on A           → loyalty 1.0,   volume ~0.1 → far outer halo of A
     50 on A + 50 on B → anchor at the A–B midpoint, mid-size orbit
    Finally a keep-out pass evicts every dot from the vicinity of channels it
    never commented in, so the 50/50 A–B crowd can never sit on top of C.
    """
    n_ch = len(ch_x)
    rng = np.random.default_rng(42)
    csr = uc_sparse.tocsr()
    csr.sort_indices()
    indptr, indices, data = csr.indptr, csr.indices, csr.data.astype(np.float64)
    counts = np.diff(indptr)
    if counts.min() < 1:                                   # reduceat needs non-empty rows
        raise ValueError("every user must have at least one channel edge")
    row_id = np.repeat(np.arange(n_users, dtype=np.int64), counts)
    totals = np.maximum(np.bincount(row_id, weights=data, minlength=n_users), 1e-9)
    shares = data / totals[row_id]
    # ── 1. Sharpened centroid ───────────────────────────────────────────
    # share**GAMMA collapses the anchor onto the dominant channel when the
    # split is lopsided, but leaves a genuine 50/50 user on the midpoint.
    q = shares ** U_GAMMA
    qsum = np.maximum(np.bincount(row_id, weights=q, minlength=n_users), 1e-12)
    cx = np.bincount(row_id, weights=q * ch_x[indices], minlength=n_users) / qsum
    cy = np.bincount(row_id, weights=q * ch_y[indices], minlength=n_users) / qsum
    # ── 2. Dominant channel per user (first edge attaining the row max) ──
    rowmax = np.maximum.reduceat(data, indptr[:-1])
    hit = np.flatnonzero(data == rowmax[row_id])
    _, first = np.unique(row_id[hit], return_index=True)
    dom_edge = hit[first]
    dom_ch = indices[dom_edge]
    w_max = data[dom_edge]
    loyalty = w_max / totals
    # ── 3. Volume, normalised per channel (channel sizes differ wildly) ──
    csc = uc_sparse.tocsc()
    cap = np.ones(n_ch)
    for j in range(n_ch):
        col = csc.data[csc.indptr[j]:csc.indptr[j + 1]]
        if col.size:
            cap[j] = max(float(np.percentile(col, U_VOL_PCTL)), 2.0)
    volume = np.clip(np.log1p(w_max) / np.log1p(cap[dom_ch]), 0.0, 1.0)
    affinity = np.clip(loyalty ** U_LOYALTY_POW * volume ** U_VOLUME_POW, 0.0, 1.0)
    r = U_ORBIT_MIN + (U_ORBIT_MAX - U_ORBIT_MIN) * (1.0 - affinity)
    r *= rng.uniform(0.85, 1.15, n_users)                  # break up concentric rings
    # ── 4. Territory cap: never orbit into a foreign channel's space ─────
    d_out = np.empty(n_users)
    for s in range(0, n_users, U_CHUNK):
        e = min(s + U_CHUNK, n_users)
        dd = np.hypot(cx[s:e, None] - ch_x[None, :], cy[s:e, None] - ch_y[None, :])
        member = np.zeros((e - s, n_ch), dtype=bool)
        lo, hi = indptr[s], indptr[e]
        member[row_id[lo:hi] - s, indices[lo:hi]] = True
        dd[member] = np.inf                                # own channels don't constrain
        d_out[s:e] = dd.min(axis=1)
    d_out = np.where(np.isfinite(d_out), d_out, 1e6)       # member of everything → no cap
    r = np.minimum(r, np.maximum(U_ORBIT_MIN, U_TERRITORY * d_out))
    # ── 5. Scatter onto the orbit ───────────────────────────────────────
    theta = rng.uniform(0, 2 * np.pi, n_users)
    ux = cx + r * np.cos(theta) + rng.normal(0, U_JITTER, n_users) * r
    uy = cy + r * np.sin(theta) + rng.normal(0, U_JITTER, n_users) * r
    # ── 6. Keep-out pass ────────────────────────────────────────────────
    # Non-members must clear KEEPOUT × personal_r; members only need a tiny
    # buffer so they don't land exactly on the centre.  Randomised overshoot
    # prevents a hard ring forming at the keep-out boundary.
    keepout = U_KEEPOUT * personal_r
    for _ in range(2):
        for j in range(n_ch):
            member = np.zeros(n_users, dtype=bool)
            member[csc.indices[csc.indptr[j]:csc.indptr[j + 1]]] = True
            dx = ux - ch_x[j]
            dy = uy - ch_y[j]
            dist = np.hypot(dx, dy)
            required = np.where(member, U_MEMBER_BUFFER, keepout[j])
            mask = dist < required
            if not mask.any():
                continue
            k = int(mask.sum())
            dxm, dym, dm = dx[mask], dy[mask], dist[mask]
            # dots sitting exactly on the centre get an arbitrary ray
            dead = dm < 1e-6
            if dead.any():
                ang = rng.uniform(0, 2 * np.pi, int(dead.sum()))
                dxm[dead], dym[dead] = np.cos(ang), np.sin(ang)
                dm[dead] = 1.0
            req = required[mask]
            out = req + np.abs(rng.normal(0, 0.25 * req, k))
            ux[mask] = ch_x[j] + dxm / dm * out
            uy[mask] = ch_y[j] + dym / dm * out
    return ux, uy


@api_bp.route('/api/community_graph', methods=['GET'])
@cached_json(
    lambda: "community_graph_xcAZSdf"
            f"_{request.args.get('month')}"
            f"_{request.args.get('include_edges', 'true')}"
            f"_{request.args.get('channel_group', 'all')}"
)
def community_graph():
    """
    Full bipartite user ↔ channel community graph for a single month.
    Query parameters
    ----------------
    month           str   (required)  e.g. "2025-03"
    resolution      float (optional, default 1.0)
        Leiden resolution.  Lower → fewer, larger communities.
    include_edges   bool  (optional, default true)
        Set to "false" to omit the edges object and roughly halve
        the payload size.
    channel_group   str   (optional)
        Filter to channels in this Channel.channel_group.
        Omit or pass "all" to include every channel.
    """
    month_str = request.args.get("month")
    if not month_str:
        return jsonify({"error": "month parameter is required (e.g. '2025-03')"}), 400
    include_edges = request.args.get("include_edges", "true").lower() != "false"
    channel_group = request.args.get("channel_group")          # None → no filter
    try:
        t0 = time.perf_counter()
        month_start = parse_month(month_str)
        # ── 1. Query user–channel interactions for the month ────────────
        query = (
            db.session.query(
                UserData.user_id,
                Channel.channel_name,
                func.sum(UserData.total_message_count).label("weight"),
            )
            .join(Channel, UserData.channel_id == Channel.channel_id)
            .filter(
                func.date_trunc("month", UserData.last_message_at) == month_start,
                UserData.total_message_count > 0,
            )
        )
        # Optional channel_group filter
        if channel_group and channel_group.lower() != "all":
            query = query.filter(Channel.channel_group == channel_group)
        rows = (
            query
            .group_by(UserData.user_id, Channel.channel_name)
            .all()
        )
        if not rows:
            return jsonify({"error": "No data found for the specified month"}), 404
        df = pd.DataFrame(rows, columns=["user_id", "channel_name", "weight"])
        logger.info("[community_graph] query: %d interaction rows (%.1fs)",
                     len(df), time.perf_counter() - t0)
        # ── 2. Build stable integer indices ─────────────────────────────
        channel_names = sorted(df["channel_name"].unique().tolist())
        user_ids      = np.sort(df["user_id"].unique())
        ch_to_idx = {c: i for i, c in enumerate(channel_names)}
        u_to_idx  = {u: i for i, u in enumerate(user_ids)}
        n_ch    = len(channel_names)
        n_users = len(user_ids)
        # Vectorised mapping — no Python-level row iteration
        edge_ch = df["channel_name"].map(ch_to_idx).values.astype(np.int32)
        edge_u  = df["user_id"].map(u_to_idx).values.astype(np.int32)
        edge_w  = df["weight"].values.astype(np.float64)
        n_edges = len(edge_w)
        # ── 3. Community detection (channel-similarity Leiden) ──────────
        t1 = time.perf_counter()
        uc_sparse = csr_matrix(
            (edge_w, (edge_u, edge_ch)), shape=(n_users, n_ch),
        )
        sim = cosine_similarity(uc_sparse.T)
        ch_comm_raw = _find_channel_communities(sim, n_ch)
        usr_comm_raw = _assign_user_communities(
            edge_u, edge_ch, edge_w, ch_comm_raw, n_users,
        )
        all_mem = np.concatenate([ch_comm_raw, usr_comm_raw])
        membership, n_communities = _remap_communities_by_size(all_mem)
        ch_communities  = membership[:n_ch]
        usr_communities = membership[n_ch:]
        logger.info("[community_graph] %d communities (%.1fs)",
                     n_communities, time.perf_counter() - t1)
        # ── 4. Degree counts (needed by the layout) ─────────────────────
        ch_degree  = np.bincount(edge_ch, minlength=n_ch)
        usr_degree = np.bincount(edge_u,  minlength=n_users)
        # ── 5. Spatial layout ───────────────────────────────────────────
        ch_x, ch_y, ch_personal = _layout_channels(sim, n_ch, ch_communities, ch_degree)
        usr_x, usr_y = _position_users(uc_sparse, ch_x, ch_y, ch_personal, n_users)
        # ── 6. Assemble compact payload ────────────────────────────────
        formatted_month = month_start.strftime("%B %Y")
        payload = {
            "title": f"Community Graph for {formatted_month}",
            "channels": [
                {
                    "name":      channel_names[i],
                    "x":         round(float(ch_x[i]), 2),
                    "y":         round(float(ch_y[i]), 2),
                    "community": int(ch_communities[i]),
                    "degree":    int(ch_degree[i]),
                }
                for i in range(n_ch)
            ],
            "users": {
                "count":     n_users,
                "x":         np.round(usr_x, 2).tolist(),
                "y":         np.round(usr_y, 2).tolist(),
                "community": usr_communities.tolist(),
                "degree":    usr_degree.tolist(),
            },
            "stats": {
                "user_count":      n_users,
                "channel_count":   n_ch,
                "edge_count":      n_edges,
                "community_count": n_communities,
            },
        }
        if include_edges:
            payload["edges"] = {
                "source": edge_u.tolist(),          # → index into users.*[]
                "target": edge_ch.tolist(),         # → index into channels[]
                "weight": np.round(edge_w, 1).tolist(),
            }
        logger.info(
            "[community_graph] payload ready – %d users, %d channels, "
            "%d edges, %d communities (%.1fs total)",
            n_users, n_ch, n_edges, n_communities,
            time.perf_counter() - t0,
        )
        return payload
    except Exception as e:
        logger.exception("community_graph failed")
        return jsonify({"error": str(e)}), 500
    
def _graph_user_query(month_start, channel_group):
    """The exact same row set /community_graph builds its user index from."""
    q = (
        db.session.query(UserData.user_id)
        .join(Channel, UserData.channel_id == Channel.channel_id)
        .filter(
            func.date_trunc("month", UserData.last_message_at) == month_start,
            UserData.total_message_count > 0,
        )
    )
    if channel_group and channel_group.lower() != "all":
        q = q.filter(Channel.channel_group == channel_group)
    return q

_MAX_CANDIDATES = 12
_CHANNEL_ID_RE = re.compile(r"^UC[A-Za-z0-9_-]{22}$")

def _handle_key(s: str) -> str:
    """Normalise a handle/username for comparison: trim, drop a leading @, lower."""
    return s.strip().lstrip("@").strip().lower()

def _username_key(col):
    """The exact same normalisation, expressed in SQL."""
    return func.lower(func.btrim(func.ltrim(func.btrim(col), "@")))

def _escape_like(s: str) -> str:
    """A pasted display name must not be interpreted as a LIKE pattern."""
    return s.replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_")

def _resolve_user(term):
    """
    Resolve free text to a User row.
    Returns (user, candidates) — at most one of them is non-None.
    """
    term = (term or "").strip()
    if not term:
        return None, None
    # 1 ▪ exact channel id (case-sensitive — YouTube ids are)
    user = User.query.filter(User.user_id == term).first()
    if user:
        return user, None
    key = _handle_key(term)
    if not key:
        return None, None
    norm = _username_key(User.username)
    # 2 ▪ exact username/handle, ignoring case, surrounding space AND a leading '@'
    #     on *either* side, so "@Gura", "Gura" and " gura " all match "@GawrGura"→no,
    #     but all match a stored "@Gura" / "Gura" / " Gura ".
    exact = User.query.filter(norm == key).limit(_MAX_CANDIDATES + 1).all()
    if len(exact) == 1:
        return exact[0], None
    if exact:
        return None, exact[:_MAX_CANDIDATES]
    # 3 ▪ prefix, then substring — both on the normalised column
    pat = _escape_like(key)
    for expr in (norm.like(f"{pat}%", escape="\\"), norm.like(f"%{pat}%", escape="\\")):
        hits = (
            User.query.filter(expr)
            .order_by(func.length(User.username), User.username)
            .limit(_MAX_CANDIDATES)
            .all()
        )
        if len(hits) == 1:
            return hits[0], None
        if hits:
            return None, hits
    return None, None

@api_bp.route('/api/community_graph/find_user', methods=['GET'])
def community_graph_find_user():
    month_str = request.args.get("month")
    term = (request.args.get("q") or "").strip()
    channel_group = request.args.get("channel_group")
    if not month_str:
        return jsonify({"error": "month parameter is required"}), 400
    if not term:
        return jsonify({"error": "q parameter is required"}), 400
    try:
        month_start = parse_month(month_str)
        user, candidates = _resolve_user(term)
        if candidates:
            return jsonify({
                "candidates": [
                    {"user_id": u.user_id, "username": u.username} for u in candidates
                ]
            })
        if user is not None:
            user_id, username = user.user_id, user.username
        elif _CHANNEL_ID_RE.match(term):
            # `users` can lag behind `user_data`; a well-formed id is still plottable
            user_id, username = term, term
        else:
            return jsonify({"error": f"No user matching '{term}'."}), 404
        base = _graph_user_query(month_start, channel_group)
        if not base.filter(UserData.user_id == user_id).first():
            return jsonify({
                "error": f"{username} has no messages in "
                         f"{month_start.strftime('%B %Y')}"
                         + (f" for {channel_group}." if channel_group
                            and channel_group.lower() != "all" else ".")
            }), 404
        index = (
            base.filter(collate(UserData.user_id, "C") < user_id)
                .distinct()
                .count()
        )
        return jsonify({"user_id": user_id, "username": username, "index": int(index)})
    except Exception as e:
        logger.exception("community_graph_find_user failed")
        return jsonify({"error": str(e)}), 500
    
def _compute_channel_recommendations(user_id, months, participation_exclusion_threshold):
    today = datetime.utcnow().date()
    current_month_start = today.replace(day=1)
    all_data = []
    for i in range(months):
        month_start = current_month_start - relativedelta(months=i + 1)
        month_end = current_month_start - relativedelta(months=i)
        month_key = month_start.strftime("%Y-%m")
        monthly_data = _get_monthly_recommendation_data(
            g.redis_conn, month_start, month_end, month_key,
        )
        all_data.extend(monthly_data)
    if not all_data:
        return jsonify({"error": "Not enough recent data available to generate recommendations."}), 404
    df = pd.DataFrame(all_data, columns=['user_id', 'channel_name', 'message_weight'])
    df['message_weight'] = pd.to_numeric(df['message_weight'], errors='coerce').fillna(0)
    df = df.groupby(['user_id', 'channel_name'], as_index=False)['message_weight'].sum()
    user_channel_matrix = df.pivot(index='user_id', columns='channel_name', values='message_weight').fillna(0)
    if user_id not in user_channel_matrix.index:
        return jsonify({"error": f"No activity found for this user in the last {months} month(s)."}), 404
    user_vector = user_channel_matrix.loc[user_id]
    user_channels = list(user_vector[user_vector > 0].index)
    if not user_channels:
        return jsonify({"error": f"No channel activity found for this user in the last {months} month(s)."}), 404
    channels_to_exclude = list(user_vector[user_vector > participation_exclusion_threshold].index)
    similarity_matrix = cosine_similarity(user_channel_matrix.T)
    channel_names = user_channel_matrix.columns
    similarity_df = pd.DataFrame(similarity_matrix, index=channel_names, columns=channel_names)
    recommendation_scores = similarity_df[user_channels].sum(axis=1)
    recommendation_scores = recommendation_scores.drop(labels=channels_to_exclude, errors='ignore')
    top_recommendations = recommendation_scores.sort_values(ascending=False).head(10)
    ideal_max = len(user_channels)
    raw_normalized = (top_recommendations / ideal_max) * 100
    normalized_scores = np.log1p(raw_normalized) / np.log1p(100) * 100
    response = [
        {"channel_name": channel, "score": round(float(score), 2)}
        for channel, score in normalized_scores.items()
    ]
    return {"recommended_channels": response}


@api_bp.route("/api/recommend", methods=["GET"])
def recommend_channels():
    PARTICIPATION_EXCLUSION_THRESHOLD = 3
    try:
        identifier = request.args.get("identifier", "")
        if not identifier:
            return jsonify({"error": "Missing required parameter: 'identifier' is required."}), 400
        try:
            months = int(request.args.get("months", 1))
            if not 1 <= months <= 6:
                return jsonify({"error": "Parameter 'months' must be between 1 and 6."}), 400
        except ValueError:
            return jsonify({"error": "Parameter 'months' must be a valid integer."}), 400
        user_id = resolve_user_id(identifier)
        if user_id is None:
            return jsonify({"error": f"User handle '{identifier}' not found"}), 404
        redis_key = f"channel_recommendations:{user_id}:{months}m"
        return get_or_compute_cached(
            redis_key,
            lambda: _compute_channel_recommendations(user_id, months, PARTICIPATION_EXCLUSION_THRESHOLD),
        )
    except Exception as e:
        print(f"An error occurred in recommend_channels: {e}")
        return jsonify({"error": "An internal server error occurred."}), 500


def _get_monthly_recommendation_data(redis_conn, month_start, month_end, month_key):
    """
    Fetch aggregated user-channel activity data for a specific month.
    Uses Redis caching to avoid repeated expensive database queries.
    Args:
        redis_conn: Redis connection
        month_start: First day of the month (inclusive)
        month_end: First day of next month (exclusive)
        month_key: String key for the month (YYYY-MM format)
    Returns:
        List of [user_id, channel_name, message_weight] entries
    """
    redis_monthly_key = f"recommendation_monthly_data:{month_key}"
    cached_data = redis_conn.get(redis_monthly_key)
    if cached_data:
        return json.loads(cached_data)
    rows = (
        db.session.query(
            UserData.user_id,
            Channel.channel_name,
            cast(func.sum(UserData.total_message_count), BigInteger)
                .label("message_weight"),
        )
        .join(Channel, UserData.channel_id == Channel.channel_id)
        .filter(
            UserData.last_message_at >= month_start,
            UserData.last_message_at < month_end,
            UserData.total_message_count > 0,
        )
        .group_by(UserData.user_id, Channel.channel_name)
        .all()
    )
    data = [[r.user_id, r.channel_name, int(r.message_weight)] for r in rows]
    redis_conn.set(redis_monthly_key, json.dumps(data))
    return data

def _get_distinct_users_for_channel(channel_name, month_date):
    """Distinct user IDs from mv_user_monthly_activity for a channel + month."""
    rows = (
        db.session.query(MvUserMonthlyActivity.user_id)
        .join(Channel, MvUserMonthlyActivity.channel_id == Channel.channel_id)
        .filter(
            Channel.channel_name == channel_name,
            MvUserMonthlyActivity.observed_month == month_date,
        )
        .distinct()
        .all()
    )
    return {r[0] for r in rows}

def _get_distinct_members_for_channel(channel_name, month_date):
    """Distinct member user IDs (membership_rank >= 0) from user_data for a channel + month."""
    rows = (
        db.session.query(UserData.user_id)
        .join(Channel, UserData.channel_id == Channel.channel_id)
        .filter(
            Channel.channel_name == channel_name,
            func.date_trunc('month', UserData.last_message_at) == month_date,
            UserData.membership_rank >= 0,
        )
        .distinct()
        .all()
    )
    return {r[0] for r in rows}

@api_bp.route('/api/get_monthly_streaming_hours', methods=['GET'])
@cached_json(lambda: f"monthly_streaming_hours_{request.args.get('channel')}_forecast_{request.args.get('include_forecast', 'true').lower() == 'true'}")
def get_monthly_streaming_hours():
    channel_name = request.args.get('channel')
    include_forecast = request.args.get('include_forecast', 'true').lower() == 'true'
    if not channel_name:
        return jsonify({"error": _("Missing required parameters")}), 400
    month_col = cast(func.date_trunc('month', Video.end_time), Date).label('month')
    hours_col = func.round(
        cast(func.sum(extract('epoch', Video.duration)) / 3600, Numeric), 2
    ).label('total_streaming_hours')
    historical_results = (
        db.session.query(month_col, hours_col)
        .join(Channel, Video.channel_id == Channel.channel_id)
        .filter(Channel.channel_name == channel_name)
        .group_by(month_col)
        .order_by(month_col)
        .all()
    )
    output_data = [
        {
            "month": row.month.strftime('%Y-%m'),
            "total_streaming_hours": float(row.total_streaming_hours),
            "is_forecast": False,
        }
        for row in historical_results
        if row.total_streaming_hours is not None
    ]
    if include_forecast and historical_results:
        sf = aliased(StreamingForecast)
        latest_created = (
            db.session.query(func.max(sf.created_at))
            .filter(sf.channel_id == Channel.channel_id)
            .correlate(Channel)
            .scalar_subquery()
        )
        forecast_results = (
            db.session.query(
                StreamingForecast.forecast_month,
                StreamingForecast.forecasted_hours,
                StreamingForecast.confidence_p25,
                StreamingForecast.confidence_p75,
            )
            .join(Channel, StreamingForecast.channel_id == Channel.channel_id)
            .filter(
                Channel.channel_name == channel_name,
                StreamingForecast.created_at == latest_created,
            )
            .order_by(StreamingForecast.forecast_month)
            .all()
        )
        for row in forecast_results:
            if row.forecasted_hours is None:
                continue
            predicted = float(row.forecasted_hours)
            output_data.append({
                "month": row.forecast_month.strftime('%Y-%m'),
                "total_streaming_hours": predicted,
                "is_forecast": True,
                "confidence_low": float(row.confidence_p25) if row.confidence_p25 is not None else predicted * 0.8,
                "confidence_high": float(row.confidence_p75) if row.confidence_p75 is not None else predicted * 1.2,
            })
    return output_data

def _aggregate_streaming_hours(agg_func):
    """Execute streaming_hours_query with the given aggregate and format the response."""
    results = streaming_hours_query(agg_func).all()
    return [
        {"channel": r.channel_name, "month": r.month.strftime('%Y-%m'), "hours": round(float(r.hours), 2)}
        for r in results if r.hours is not None
    ]
def _streaming_hours_key(prefix):
    return lambda: f"{prefix}_{request.args.get('group')}_{request.args.get('month', datetime.utcnow().strftime('%Y-%m'))}"

@api_bp.route('/api/get_group_total_streaming_hours', methods=['GET'])
@cached_json(_streaming_hours_key('group_total_streaming_hours'))
def get_group_total_streaming_hours():
    try:
        return {"success": True, "data": _aggregate_streaming_hours(func.sum)}
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@api_bp.route('/api/get_group_avg_streaming_hours', methods=['GET'])
@cached_json(_streaming_hours_key('group_avg_streaming_hours'))
def get_group_avg_streaming_hours():
    try:
        return {"success": True, "data": _aggregate_streaming_hours(func.avg)}
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@api_bp.route('/api/get_group_max_streaming_hours', methods=['GET'])
@cached_json(_streaming_hours_key('group_max_streaming_hours'))
def get_group_max_streaming_hours():
    try:
        return {"success": True, "data": _aggregate_streaming_hours(func.max)}
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@api_bp.route('/api/get_group_chat_makeup', methods=['GET'])
@cached_json(lambda: f"group_chat_makeup_{request.args.get('group')}_{request.args.get('month', datetime.utcnow().strftime('%Y-%m'))}")
def get_group_chat_makeup():
    group = request.args.get('group', None)
    month = request.args.get('month', datetime.utcnow().strftime('%Y-%m'))
    start_month = parse_month(month)
    utc_month = func.date_trunc('month', func.timezone('UTC', Video.end_time))
    streaming_time = (
        db.session.query(
            Video.channel_id.label('channel_id'),
            utc_month.label('observed_month'),
            (func.sum(extract('epoch', Video.duration)) / 60).label('total_streaming_minutes'),
        )
        .filter(
            Video.duration.isnot(None),
            Video.duration > timedelta(0),
            Video.has_chat_log.is_(True),
            utc_month == start_month,
        )
        .group_by(Video.channel_id, utc_month)
        .cte('streaming_time')
    )
    st = streaming_time.c
    def _rate(col):
        return cast(func.sum(col), Numeric) / func.nullif(func.sum(st.total_streaming_minutes), 0)
    query = (
        db.session.query(
            Channel.channel_name,
            st.observed_month,
            _rate(ChatLanguageStatsMv.es_en_id_count).label('es_en_id_rate_per_minute'),
            _rate(ChatLanguageStatsMv.jp_count).label('jp_rate_per_minute'),
            _rate(ChatLanguageStatsMv.kr_count).label('kr_rate_per_minute'),
            _rate(ChatLanguageStatsMv.ru_count).label('ru_rate_per_minute'),
            _rate(ChatLanguageStatsMv.emoji_count).label('emoji_rate_per_minute'),
        )
        .select_from(ChatLanguageStatsMv)
        .join(Channel, ChatLanguageStatsMv.channel_id == Channel.channel_id)
        .join(
            streaming_time,
            and_(
                st.channel_id == Channel.channel_id,
                cast(st.observed_month, Date) == cast(ChatLanguageStatsMv.observed_month, Date),
            ),
        )
    )
    if group:
        query = query.filter(Channel.channel_group == group)
    try:
        results = (
            query
            .group_by(Channel.channel_name, st.observed_month)
            .order_by(func.sum(st.total_streaming_minutes).desc())
            .all()
        )
        data = [
            {
                "channel_name": row.channel_name,
                "observed_month": row.observed_month.strftime('%Y-%m'),
                "es_en_id_rate_per_minute": round(float(row.es_en_id_rate_per_minute or 0), 2),
                "jp_rate_per_minute": round(float(row.jp_rate_per_minute or 0), 2),
                "kr_rate_per_minute": round(float(row.kr_rate_per_minute or 0), 2),
                "ru_rate_per_minute": round(float(row.ru_rate_per_minute or 0), 2),
                "emoji_rate_per_minute": round(float(row.emoji_rate_per_minute or 0), 2),
            }
            for row in results
        ]
        return {"success": True, "data": data}
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
    
def _build_set_overlap_response(set_a, set_b, channel_a, channel_b, month_a, month_b, noun):
    """
    Build the overlap comparison response for two ID sets (users or members)
    between two channel/month pairs. `noun` is "users" or "members" and
    controls output key naming.
    """
    if not set_a or not set_b:
        return {}
    common = set_a & set_b
    total_a, total_b = len(set_a), len(set_b)
    num_common = len(common)
    return {
        "channel_a": channel_a,
        "channel_b": channel_b,
        "month_a": month_a,
        "month_b": month_b,
        f"num_common_{noun}": num_common,
        f"percent_A_to_B_{noun}": round(100.0 * num_common / total_a, 2) if total_a else 0,
        f"percent_B_to_A_{noun}": round(100.0 * num_common / total_b, 2) if total_b else 0,
    }

@api_bp.route('/api/get_common_users', methods=['GET'])
@cached_json(
    lambda: f"common_users_{request.args.get('channel_a')}_{request.args.get('month_a')}_"
            f"{request.args.get('channel_b')}_{request.args.get('month_b')}",
)
def get_common_users():
    channel_a_name = request.args.get('channel_a')
    month_a_str = request.args.get('month_a')
    channel_b_name = request.args.get('channel_b')
    month_b_str = request.args.get('month_b')
    if not (channel_a_name and month_a_str and channel_b_name and month_b_str):
        return jsonify({"error": _("Missing required parameters")}), 400
    try:
        users_a = _get_distinct_users_for_channel(channel_a_name, f"{month_a_str}-01")
        users_b = _get_distinct_users_for_channel(channel_b_name, f"{month_b_str}-01")
    except Exception as e:
        return jsonify({"error": f"Database query failed: {e}"}), 500
    return _build_set_overlap_response(users_a, users_b, channel_a_name, channel_b_name, month_a_str, month_b_str, "users")


@api_bp.route('/api/get_common_users_matrix', methods=['GET'])
@cached_json(
    lambda: (
        f"common_matrix_percent_{'members' if request.args.get('members_only', 'false').lower() in ('true', '1', 'yes') else 'users'}_"
        f"{','.join(sorted(n.strip() for n in request.args.get('channels', '').split(',')))}_"
        f"{request.args.get('month')}"
    ),
)
def get_common_users_matrix():
    month_str = request.args.get('month')
    channels_str = request.args.get('channels')
    members_only = request.args.get('members_only', 'false').lower() in ('true', '1', 'yes')
    if not (month_str and channels_str):
        return jsonify({"error": "Missing 'month' or 'channels' parameter"}), 400
    channel_names = [n.strip() for n in channels_str.split(',')]
    if len(channel_names) < 2:
        return jsonify({"error": "Please provide at least two channel names."}), 400
    month_date = f"{month_str}-01"
    lookup_fn = _get_distinct_members_for_channel if members_only else _get_distinct_users_for_channel
    try:
        user_sets = {name: lookup_fn(name, month_date) for name in channel_names}
    except Exception as e:
        return jsonify({"error": f"Database query failed: {e}"}), 500
    n = len(channel_names)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        set_i = user_sets[channel_names[i]]
        total_i = len(set_i)
        for j in range(n):
            if i == j:
                matrix[i][j] = 100.0 if total_i else 0.0
            elif total_i:
                matrix[i][j] = round(100.0 * len(set_i & user_sets[channel_names[j]]) / total_i, 2)
    return {"labels": channel_names, "matrix": matrix}


@api_bp.route('/api/get_common_members', methods=['GET'])
@cached_json(
    lambda: f"common_members_{request.args.get('channel_a')}_{request.args.get('month_a')}_"
            f"{request.args.get('channel_b')}_{request.args.get('month_b')}",
)
def get_common_members():
    channel_a_name = request.args.get('channel_a')
    month_a_str = request.args.get('month_a')
    channel_b_name = request.args.get('channel_b')
    month_b_str = request.args.get('month_b')
    if not (channel_a_name and month_a_str and channel_b_name and month_b_str):
        return jsonify({"error": _("Missing required parameters")}), 400
    try:
        members_a = _get_distinct_members_for_channel(channel_a_name, f"{month_a_str}-01")
        members_b = _get_distinct_members_for_channel(channel_b_name, f"{month_b_str}-01")
    except Exception as e:
        return jsonify({"error": f"Database query failed: {e}"}), 500
    return _build_set_overlap_response(members_a, members_b, channel_a_name, channel_b_name, month_a_str, month_b_str, "members")


@api_bp.route('/api/get_group_membership_data', methods=['GET'])
@cached_json(lambda: f"group_membership_data_{request.args.get('channel_group')}_{request.args.get('month')}")
def get_group_membership_counts():
    channel_group = request.args.get('channel_group')
    month = request.args.get('month')
    if not channel_group or not month:
        return jsonify({"error": _("Missing required parameters")}), 400
    results = (
        db.session.query(
            MembershipDataSummary.channel_name,
            MembershipDataSummary.membership_rank,
            MembershipDataSummary.membership_count,
            MembershipDataSummary.percentage_total,
        )
        .filter(
            MembershipDataSummary.channel_group == channel_group,
            MembershipDataSummary.observed_month == parse_month(month),
        )
        .all()
    )
    return [
        [row.channel_name, int(row.membership_rank), float(row.membership_count), float(row.percentage_total)]
        for row in results
    ]


@api_bp.route('/api/get_group_membership_summary', methods=['GET'])
@cached_json(lambda: (
    f"group_membership_summary_{request.args.get('channel_group')}_{request.args.get('month')}_"
    f"{request.args.get('membership_rank')}"
))
def get_group_membership_summary():
    channel_group = request.args.get("channel_group")
    month = request.args.get("month")
    membership_rank = request.args.get("membership_rank", type=str)
    if not channel_group or not month:
        return jsonify({"error": "Missing required parameters: channel_group and month"}), 400
    total = membership_rank.lower() == "total"
    if not total:
        membership_rank = int(membership_rank)
    month_date = parse_month(month)
    if total:
        results = (
            db.session.query(
                MembershipDataSummary.channel_name,
                func.sum(MembershipDataSummary.membership_count).label('total_members'),
            )
            .filter(
                MembershipDataSummary.channel_group == channel_group,
                MembershipDataSummary.observed_month == month_date,
                MembershipDataSummary.membership_rank != -1,
            )
            .group_by(MembershipDataSummary.channel_name)
            .order_by(func.sum(MembershipDataSummary.membership_count).desc())
            .all()
        )
        return [
            {"channel_name": row.channel_name, "total_members": int(row.total_members)}
            for row in results
        ]
    results = (
        db.session.query(
            MembershipDataSummary.channel_name,
            MembershipDataSummary.membership_rank,
            MembershipDataSummary.membership_count,
            MembershipDataSummary.percentage_total,
        )
        .filter(
            MembershipDataSummary.channel_group == channel_group,
            MembershipDataSummary.observed_month == month_date,
            MembershipDataSummary.membership_rank == membership_rank,
        )
        .order_by(MembershipDataSummary.membership_count.desc())
        .all()
    )
    return [
        [row.channel_name, int(row.membership_rank), float(row.membership_count), float(row.percentage_total)]
        for row in results
    ]

@api_bp.route('/api/get_group_membership_changes', methods=['GET'])
@cached_json(lambda: f"group_membership_changes_{request.args.get('channel_group')}_{request.args.get('month')}")
def get_group_membership_changes():
    channel_group = request.args.get('channel_group')
    month = request.args.get('month')
    if not channel_group or not month:
        return jsonify({"error": _("Missing required parameters")}), 400
    month_date = parse_month(month)
    last_message_month = func.date_trunc('month', UserData.last_message_at)
    membership_changes = (
        db.session.query(
            UserData.user_id.label('user_id'),
            UserData.channel_id.label('channel_id'),
            cast(last_message_month, Date).label('observed_month'),
            UserData.membership_rank.label('membership_rank'),
            func.lag(UserData.membership_rank).over(
                partition_by=[UserData.user_id, UserData.channel_id],
                order_by=UserData.last_message_at,
            ).label('previous_membership_rank'),
        )
        .join(Channel, UserData.channel_id == Channel.channel_id)
        .filter(
            Channel.channel_group == channel_group,
            last_message_month == month_date,
        )
        .cte('membership_changes')
    )
    mc = membership_changes.c
    gains = (
        db.session.query(mc.user_id, mc.channel_id, mc.observed_month)
        .filter(
            mc.previous_membership_rank == -1,
            mc.membership_rank.is_distinct_from(-1),
            mc.membership_rank.isnot(None),
        )
        .cte('gains')
    )
    expirations = (
        db.session.query(mc.user_id, mc.channel_id, mc.observed_month)
        .filter(
            mc.previous_membership_rank.is_distinct_from(-1),
            mc.previous_membership_rank.isnot(None),
            mc.membership_rank == -1,
        )
        .cte('expirations')
    )
    observed_month_col = func.coalesce(gains.c.observed_month, expirations.c.observed_month).label('observed_month')
    gains_count_expr = func.count(func.distinct(gains.c.user_id))
    losses_count_expr = func.count(func.distinct(expirations.c.user_id))
    differential_expr = (gains_count_expr - losses_count_expr).label('differential')
    try:
        results = (
            db.session.query(
                Channel.channel_name,
                observed_month_col,
                gains_count_expr.label('gains_count'),
                losses_count_expr.label('losses_count'),
                differential_expr,
            )
            .select_from(Channel)
            .outerjoin(gains, gains.c.channel_id == Channel.channel_id)
            .outerjoin(expirations, expirations.c.channel_id == Channel.channel_id)
            .filter(
                Channel.channel_group == channel_group,
                or_(
                    gains.c.observed_month == month_date,
                    expirations.c.observed_month == month_date,
                ),
            )
            .group_by(Channel.channel_name, observed_month_col)
            .order_by(differential_expr.desc())
            .all()
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return [
        {
            "channel_name": row.channel_name,
            "observed_month": row.observed_month.strftime('%Y-%m') if row.observed_month else None,
            "gains_count": row.gains_count,
            "losses_count": row.losses_count,
            "differential": row.differential,
        }
        for row in results if row.observed_month is not None
    ]


@api_bp.route('/api/get_group_streaming_hours_diff', methods=['GET'])
@cached_json(lambda: f"group_streaming_hours_diff_{request.args.get('group') or 'all'}_{request.args.get('month')}")
def get_group_streaming_hours_diff():
    month = request.args.get('month')
    channel_group = request.args.get('group', None)
    if not month:
        return jsonify({"success": False, "error": "Missing required parameter: month"}), 400
    try:
        month_dt = parse_month(month)
    except ValueError:
        return jsonify({"success": False, "error": "Invalid month format. Use YYYY-MM."}), 400
    utc_month = func.date_trunc('month', func.timezone('UTC', Video.end_time))
    monthly_streaming_q = (
        db.session.query(
            Channel.channel_name.label('channel_name'),
            utc_month.label('observed_month'),
            (func.sum(extract('epoch', Video.duration)) / 3600).label('total_streaming_hours'),
        )
        .join(Channel, Video.channel_id == Channel.channel_id)
    )
    if channel_group:
        monthly_streaming_q = monthly_streaming_q.filter(Channel.channel_group == channel_group)
    monthly_streaming = (
        monthly_streaming_q
        .group_by(Channel.channel_name, utc_month)
        .cte('monthly_streaming')
    )
    m1 = monthly_streaming
    m2 = monthly_streaming.alias('m2')
    one_month = literal_column("INTERVAL '1 month'")
    change_expr = func.coalesce(
        m1.c.total_streaming_hours - m2.c.total_streaming_hours,
        m1.c.total_streaming_hours,
    ).label('change_from_previous_month')
    results = (
        db.session.query(
            m1.c.channel_name,
            m1.c.observed_month,
            m1.c.total_streaming_hours,
            change_expr,
        )
        .select_from(m1)
        .outerjoin(
            m2,
            and_(
                m1.c.channel_name == m2.c.channel_name,
                m1.c.observed_month == (m2.c.observed_month + one_month),
            ),
        )
        .filter(m1.c.observed_month == month_dt)
        .order_by(change_expr.desc())
        .all()
    )
    data = [
        {
            "channel": row.channel_name,
            "month": row.observed_month.strftime('%Y-%m'),
            "hours": float(round(row.total_streaming_hours, 2)) if row.total_streaming_hours is not None else 0.0,
            "change": float(round(row.change_from_previous_month, 2)) if row.change_from_previous_month is not None else 0.0,
        }
        for row in results
    ]
    return {"success": True, "data": data}


@api_bp.route('/api/get_chat_leaderboard', methods=['GET'])
@cached_json(lambda: f"chat_leaderboard_{request.args.get('channel_name')}_{request.args.get('month')}")
def get_chat_leaderboard():
    channel_name = request.args.get('channel_name')
    month = request.args.get('month')
    if not channel_name or not month:
        return jsonify({"error": _("Missing required parameters")}), 400
    month_start, month_end = month_range(month)
    results = (
        db.session.query(
            User.username.label('user_name'),
            func.sum(UserData.total_message_count).label('message_count'),
        )
        .join(Channel, UserData.channel_id == Channel.channel_id)
        .join(User, UserData.user_id == User.user_id)
        .filter(
            Channel.channel_name == channel_name,
            UserData.last_message_at >= month_start,
            UserData.last_message_at < month_end,
        )
        .group_by(UserData.user_id, User.username)
        .order_by(func.sum(UserData.total_message_count).desc())
        .limit(10)
        .all()
    )
    if not results:
        return jsonify({"error": _("No data found")}), 404
    return [{"user_name": row.user_name, "message_count": row.message_count} for row in results]



@api_bp.route('/api/get_user_changes', methods=['GET'])
@cached_json(lambda: f"user_changes_{request.args.get('group')}_{request.args.get('month')}")
def get_user_changes():
    """
    Get user changes for channels in a specific group.
    
    This function analyzes user activity changes for channels within a specific
    channel group, tracking gained and lost users over time.
    
    Args:
        group (str): Name of the channel group
        month (str): Month for analysis (YYYY-MM)
        
    Returns:
        List of user change entries with gained, lost, and net change counts
        
    Raises:
        ValueError: If required parameters are missing
        InternalError: If an unexpected error occurs
    """
    channel_group = request.args.get('group')
    month = request.args.get('month')
    if not (channel_group and month):
        return jsonify({"error": _("Missing required parameters")}), 400
    current_month_start, _unused = month_range(month)
    previous_month_start = current_month_start - relativedelta(months=1)
    current_month_users = (
        db.session.query(
            MvUserMonthlyActivity.user_id.label('user_id'),
            MvUserMonthlyActivity.channel_id.label('channel_id'),
        )
        .filter(
            MvUserMonthlyActivity.observed_month == current_month_start,
            MvUserMonthlyActivity.monthly_message_count >= 5,
        )
        .cte('current_month_users')
    )
    previous_month_users = (
        db.session.query(
            MvUserMonthlyActivity.user_id.label('user_id'),
            MvUserMonthlyActivity.channel_id.label('channel_id'),
        )
        .filter(
            MvUserMonthlyActivity.observed_month == previous_month_start,
            MvUserMonthlyActivity.monthly_message_count >= 5,
        )
        .cte('previous_month_users')
    )
    uma1 = current_month_users.c
    uma2 = previous_month_users.c
    gained_case = case((uma1.user_id.isnot(None), 1), else_=0)
    lost_case = case((uma2.user_id.isnot(None), 1), else_=0)
    users_gained_sum = func.sum(gained_case)
    users_lost_sum = func.sum(lost_case)
    net_change_sum = (users_gained_sum - users_lost_sum).label('net_change')
    results = (
        db.session.query(
            Channel.channel_name,
            users_gained_sum.label('users_gained'),
            users_lost_sum.label('users_lost'),
            net_change_sum,
        )
        .select_from(Channel)
        .outerjoin(current_month_users, Channel.channel_id == uma1.channel_id)
        .outerjoin(
            previous_month_users,
            and_(
                Channel.channel_id == uma2.channel_id,
                uma1.user_id == uma2.user_id,
            ),
        )
        .filter(Channel.channel_group == channel_group)
        .group_by(Channel.channel_name)
        .all()
    )
    return [
        {"channel": row.channel_name, "users_gained": row.users_gained, "users_lost": row.users_lost, "net_change": row.net_change}
        for row in results
        if row.users_gained > 0 and row.users_lost > 0
    ]


@api_bp.route('/api/get_exclusive_chat_users', methods=['GET'])
@cached_json(lambda: f"exclusive_chat_users_{request.args.get('channel')}")
def get_exclusive_chat_users():
    channel_name = request.args.get('channel')
    if not channel_name:
        return jsonify({"error": _("Missing required parameters")}), 400
    channel_info = (
        db.session.query(Channel.channel_id, Channel.channel_group)
        .filter(Channel.channel_name == channel_name)
        .first()
    )
    if not channel_info:
        return jsonify({"error": "Invalid channel"}), 400
    channel_id, channel_group = channel_info
    channel_specific_users = (
        db.session.query(
            MvUserActivity.user_id.label('user_id'),
            MvUserActivity.activity_month.label('activity_month'),
            MvUserActivity.channel_id.label('channel_id'),
        )
        .filter(MvUserActivity.channel_id == channel_id)
        .cte('channel_specific_users')
    )
    csu = channel_specific_users.c
    other_channel_activity = (
        db.session.query(MvUserActivity)
        .filter(
            MvUserActivity.user_id == csu.user_id,
            MvUserActivity.channel_group == channel_group,
            MvUserActivity.channel_id != csu.channel_id,
        )
        .correlate(channel_specific_users)
        .exists()
    )
    exclusive_users = (
        db.session.query(
            csu.activity_month,
            func.count(func.distinct(csu.user_id)).label('exclusive_users_count'),
        )
        .filter(~other_channel_activity)
        .group_by(csu.activity_month)
        .cte('exclusive_users')
    )
    total_users = (
        db.session.query(
            csu.activity_month,
            func.count(func.distinct(csu.user_id)).label('total_users_count'),
        )
        .group_by(csu.activity_month)
        .cte('total_users_per_month')
    )
    eu = exclusive_users.c
    tu = total_users.c
    results = (
        db.session.query(
            tu.activity_month,
            func.round(
                (cast(eu.exclusive_users_count, Numeric) / tu.total_users_count) * 100, 2
            ).label('exclusive_percent'),
        )
        .select_from(total_users)
        .join(exclusive_users, tu.activity_month == eu.activity_month)
        .order_by(tu.activity_month)
        .all()
    )
    return [
        {"month": row.activity_month.strftime('%Y-%m'), "percent": float(row.exclusive_percent)}
        for row in results
    ]


@api_bp.route('/api/get_message_type_percents', methods=['GET'])
@cached_json(lambda: f"message_type_percents_{request.args.get('channel')}_{(request.args.get('language') or '').upper()}")
def get_message_type_percents():
    channel_name = request.args.get('channel')
    language = request.args.get('language').upper()
    if not channel_name or not language:
        return jsonify({"error": _("Missing required parameters")}), 400
    if language not in ["EN", "JP", "KR", "RU"]:
        return jsonify({"error": _("Invalid language parameter. Must be one of: EN, JP, KR, RU.")}), 400
    language_column_map = {
        "EN": UserData.es_en_id_count,
        "JP": UserData.jp_count,
        "KR": UserData.kr_count,
        "RU": UserData.ru_count,
    }
    language_col = language_column_map[language]
    activity_month_ud = func.date_trunc('month', UserData.last_message_at)
    monthly_data = (
        db.session.query(
            activity_month_ud.label('activity_month'),
            func.sum(language_col).label('language_message_count'),
            func.sum(UserData.total_message_count - UserData.emoji_count).label('total_message_count'),
        )
        .join(Channel, UserData.channel_id == Channel.channel_id)
        .filter(Channel.channel_name == channel_name)
        .group_by(activity_month_ud)
        .cte('monthly_data')
    )
    activity_month_v = func.date_trunc('month', Video.end_time)
    video_durations = (
        db.session.query(
            activity_month_v.label('activity_month'),
            (func.sum(extract('epoch', Video.duration)) / 60).label('total_minutes'),
        )
        .join(Channel, Video.channel_id == Channel.channel_id)
        .filter(Channel.channel_name == channel_name, Video.has_chat_log.is_(True))
        .group_by(activity_month_v)
        .cte('video_durations')
    )
    md = monthly_data.c
    vd = video_durations.c
    language_percent = func.round(
        (cast(md.language_message_count, Numeric) / func.nullif(md.total_message_count, 0)) * 100, 2
    ).label('language_percent')
    language_message_rate = func.round(
        cast(cast(md.language_message_count, Numeric) / func.nullif(vd.total_minutes, 0), Numeric), 2
    ).label('language_message_rate')
    results = (
        db.session.query(md.activity_month, language_percent, language_message_rate)
        .select_from(monthly_data)
        .outerjoin(video_durations, md.activity_month == vd.activity_month)
        .order_by(md.activity_month)
        .all()
    )
    return [
        {"month": row.activity_month.strftime('%Y-%m'), "percent": float(row.language_percent), "message_rate": float(row.language_message_rate)}
        for row in results
    ]


@api_bp.route('/api/get_attrition_rates', methods=['GET'])
def get_attrition_rates():
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
        top_users_rows = (
            db.session.query(
                UserData.user_id,
                func.sum(UserData.total_message_count).label('total_messages'),
            )
            .join(Channel, UserData.channel_id == Channel.channel_id)
            .filter(
                Channel.channel_name == channel_name,
                UserData.last_message_at >= start_date,
                UserData.last_message_at < end_date,
            )
            .group_by(UserData.user_id)
            .order_by(func.sum(UserData.total_message_count).desc())
            .limit(1000)
            .all()
        )
        top_users = [row.user_id for row in top_users_rows]
        if not top_users:
            return jsonify({"error": "No top chatters found for the given period"}), 404
        results = []
        current_month = datetime.strptime(start_from_month, "%Y-%m")
        today = datetime.utcnow()
        while current_month <= today:
            month_end = current_month + relativedelta(months=1)
            active_count = (
                db.session.query(func.count(func.distinct(UserData.user_id)))
                .join(Channel, UserData.channel_id == Channel.channel_id)
                .filter(
                    UserData.user_id.in_(top_users),
                    Channel.channel_group == 'Hololive',  # hardcoded in original query
                    UserData.last_message_at >= current_month,
                    UserData.last_message_at < month_end,
                )
                .scalar()
            ) or 0
            percent_active = round((active_count / len(top_users)) * 100, 2)
            results.append({
                "month": current_month.strftime("%Y-%m"),
                "percent": percent_active,
            })
            current_month += relativedelta(months=1)
        g.redis_conn.set(redis_key, json.dumps(results))
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route('/api/get_jp_user_percent', methods=['GET'])
@cached_json(lambda: f"jp_user_percent_{request.args.get('channel')}")
def get_jp_user_percent():
    channel_name = request.args.get('channel')
    if not channel_name:
        return jsonify({"error": "Missing required parameter: channel"}), 400
    month_col = func.date_trunc('month', UserData.last_message_at)
    user_language_usage = (
        db.session.query(
            UserData.user_id.label('user_id'),
            month_col.label('month'),
            func.sum(UserData.jp_count).label('total_jp_messages'),
            func.sum(UserData.total_message_count - UserData.emoji_count).label('total_non_emoji_messages'),
        )
        .join(Channel, UserData.channel_id == Channel.channel_id)
        .filter(Channel.channel_name == channel_name, UserData.total_message_count > 0)
        .group_by(UserData.user_id, month_col)
        .cte('user_language_usage')
    )
    ulu = user_language_usage.c
    jp_users = (
        db.session.query(ulu.month.label('month'), func.count().label('jp_user_count'))
        .filter(
            ulu.total_non_emoji_messages > 0,
            ulu.total_jp_messages > ulu.total_non_emoji_messages * 0.5,
        )
        .group_by(ulu.month)
        .cte('jp_users')
    )
    total_users = (
        db.session.query(
            month_col.label('month'),
            func.count(func.distinct(UserData.user_id)).label('total_user_count'),
        )
        .join(Channel, UserData.channel_id == Channel.channel_id)
        .filter(Channel.channel_name == channel_name, UserData.total_message_count > 0)
        .group_by(month_col)
        .cte('total_users')
    )
    jp = jp_users.c
    tu = total_users.c
    jp_user_percent = func.round(
        100.0 * func.coalesce(jp.jp_user_count, 0) / func.nullif(tu.total_user_count, 0), 2
    ).label('jp_user_percent')
    results = (
        db.session.query(func.to_char(tu.month, 'YYYY-MM').label('month'), jp_user_percent)
        .select_from(total_users)
        .outerjoin(jp_users, tu.month == jp.month)
        .order_by('month')
        .all()
    )
    return [
        {"month": row.month, "jp_user_percent": float(row.jp_user_percent) if row.jp_user_percent else 0.0}
        for row in results
    ]


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
        text = None
        bucket = os.getenv("CONFIG_BUCKET")
        if bucket:
            try:
                s3 = boto3.client(
                    "s3", endpoint_url=os.getenv("AWS_ENDPOINT_URL") or None,
                    region_name=os.getenv("AWS_REGION", "us-east-1"))
                text = s3.get_object(Bucket=bucket, Key="news.txt")["Body"].read().decode("utf-8")
            except Exception as exc:
                logger.warning("Could not load managed news from S3: %s", exc)
        if text is None:
            try:
                with open("news.txt", "r", encoding="utf-8") as file:
                    text = file.read()
            except FileNotFoundError:
                text = ""
        for line in text.splitlines():
            if ": " in line:
                date, message = line.split(": ", 1)
                news_list.append({"date": date.strip(), "message": message.strip()})

        return jsonify(news_list)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route('/api/get_channel_names', methods=['GET'])
@cached_json(lambda: "channel_names")
def get_channel_names():
    """Get a list of all channel names.

    Retrieves all unique channel names from the database and returns them
    as a sorted list. This endpoint is cached for performance.

    Returns:
        List[str]: A list of channel names sorted alphabetically.
    """
    rows = db.session.query(Channel.channel_name).order_by(Channel.channel_name).all()
    return [r.channel_name for r in rows]

@api_bp.route('/api/get_date_ranges', methods=['GET'])
@cached_json(lambda: "date_ranges")
def get_date_ranges():
    min_date, max_date = (
        db.session.query(func.min(Video.end_time), func.max(Video.end_time))
        .filter(Video.has_chat_log.is_(True))
        .one()
    )
    return [str(min_date), str(max_date)]


@api_bp.route('/api/get_number_of_chat_logs', methods=['GET'])
@cached_json(lambda: "number_of_chat_logs")
def get_number_of_chat_logs():
    return (
        db.session.query(func.count(Video.video_id))
        .filter(Video.has_chat_log.is_(True))
        .scalar()
    )


@api_bp.route('/api/get_publication_progress', methods=['GET'])
def get_publication_progress():
    """Return known work for the next sequential unpublished month."""
    row = db.session.execute(text("""
        WITH publication AS (
            SELECT MAX(observed_month) FILTER (WHERE status = 'merged') AS latest,
                   COALESCE(
                       NULLIF((SELECT value FROM service_config
                               WHERE key = 'backlog_floor'), '')::timestamptz,
                       '2026-07-01 00:00:00+00'::timestamptz
                   ) AS backlog_floor
            FROM monthly_merge_state
        ), target AS (
            SELECT COALESCE(
                       (latest + INTERVAL '1 month')::date,
                       date_trunc('month', backlog_floor AT TIME ZONE 'UTC')::date
                   ) AS target_month,
                   (date_trunc('month', NOW() AT TIME ZONE 'UTC')
                       - INTERVAL '1 month')::date AS previous_month
            FROM publication
        )
        SELECT t.target_month,
               t.target_month <= t.previous_month AS behind,
               COUNT(j.video_id) FILTER (
                   WHERE j.status NOT IN ('done', 'skipped')) AS remaining
        FROM target t
        LEFT JOIN videos v
          ON v.end_time >= (t.target_month::timestamp AT TIME ZONE 'UTC')
         AND v.end_time < ((t.target_month + INTERVAL '1 month')::timestamp
                           AT TIME ZONE 'UTC')
        LEFT JOIN ingest_jobs j ON j.video_id = v.video_id
        GROUP BY t.target_month, t.previous_month
    """)).one()
    return {
        "behind": bool(row.behind),
        "target_month": str(row.target_month),
        "remaining_chat_logs": int(row.remaining or 0),
        "approximate": True,
    }

@api_bp.route('/api/get_num_messages', methods=['GET'])
@cached_json(lambda: "num_messages")
def get_num_messages():
    return db.session.query(func.sum(UserData.total_message_count)).scalar()

@api_bp.route('/api/get_funniest_timestamps', methods=['GET'])
@cached_json(lambda: f"funniest_timestamps_{request.args.get('channel')}_{request.args.get('month')}")
def get_funniest_timestamps():
    channel_name = request.args.get('channel')
    month = request.args.get('month')
    if not (channel_name and month):
        return jsonify({"error": "Missing required parameters"}), 400
    month_start, next_month_start = month_range(month)
    last_chat = (
        db.session.query(
            UserData.video_id.label('video_id'),
            func.max(UserData.last_message_at).label('last_message_at'),
        )
        .join(Channel, UserData.channel_id == Channel.channel_id)
        .filter(
            Channel.channel_name == channel_name,
            UserData.last_message_at >= month_start,
            UserData.last_message_at < next_month_start,
        )
        .group_by(UserData.video_id)
        .cte('last_chat')
    )
    lc = last_chat.c
    relative_timestamp = extract(
        'epoch', func.to_timestamp(Video.funniest_timestamp) - lc.last_message_at + Video.duration
    ).label('relative_timestamp')
    results = (
        db.session.query(Video.title, Video.video_id, relative_timestamp)
        .join(Channel, Video.channel_id == Channel.channel_id)
        .join(last_chat, Video.video_id == lc.video_id)
        .filter(Channel.channel_name == channel_name, Video.funniest_timestamp.isnot(None))
        .order_by(Video.end_time.asc())
        .all()
    )
    return [
        {"title": row.title, "video_id": row.video_id, "timestamp": int(row.relative_timestamp)}
        for row in results
        if row.video_id is not None and row.relative_timestamp is not None
    ]


@api_bp.route('/api/get_user_info', methods=['GET'])
@cached_json(lambda: f"user_info_{request.args.get('identifier')}_{request.args.get('month')}")
def get_user_info():
    identifier = request.args.get('identifier')
    month = request.args.get('month')
    if not (identifier and month):
        return jsonify({"success": False, "error": "Missing required parameters: 'user_id' and 'month' are required."}), 400
    user_id = resolve_user_id(identifier)
    if user_id is None:
        return jsonify({"success": False, "error": "User not found."}), 404
    month_start, next_month_start = month_range(month)
    user_chat_data = (
        db.session.query(
            UserData.channel_id,
            Channel.channel_name,
            func.sum(UserData.total_message_count).label('user_message_count'),
        )
        .join(Channel, UserData.channel_id == Channel.channel_id)
        .filter(
            UserData.user_id == user_id,
            UserData.last_message_at >= month_start,
            UserData.last_message_at < next_month_start,
            UserData.total_message_count > 0,
        )
        .group_by(UserData.channel_id, Channel.channel_name)
        .all()
    )
    if not user_chat_data:
        return {"success": True, "data": []}
    results = []
    for row in user_chat_data:
        channel_id = row.channel_id
        channel_name = row.channel_name
        user_message_count = row.user_message_count
        all_user_counts = (
            db.session.query(
                UserData.user_id.label('user_id'),
                func.sum(UserData.total_message_count).label('total_messages'),
            )
            .filter(
                UserData.channel_id == channel_id,
                UserData.last_message_at >= month_start,
                UserData.last_message_at < next_month_start,
                UserData.total_message_count > 0,
            )
            .group_by(UserData.user_id)
            .cte('all_user_counts')
        )
        auc = all_user_counts.c
        percentile_expr = (
            100.0 * func.count().filter(auc.total_messages <= user_message_count)
            / func.nullif(func.count(), 0)
        )
        percentile = db.session.query(percentile_expr).select_from(all_user_counts).scalar()
        percentile = percentile if percentile is not None else 0.0
        results.append({
            "channel_name": channel_name,
            "message_count": int(user_message_count),
            "percentile": round(float(percentile), 2),
        })
    return {"success": True, "data": results}


@api_bp.route('/api/get_chat_engagement', methods=['GET'])
@cached_json(lambda: f"chat_engagement_{request.args.get('month', datetime.utcnow().strftime('%Y-%m'))}_{request.args.get('group')}")
def get_chat_engagement():
    month = request.args.get('month', datetime.utcnow().strftime('%Y-%m'))
    group = request.args.get('group', None)
    start_month = parse_month(month)
    chat_engagement_q = (
        db.session.query(
            UserData.channel_id.label('channel_id'),
            func.count(func.distinct(UserData.user_id)).label('total_users'),
            func.sum(UserData.total_message_count).label('total_messages'),
        )
        .join(Channel, UserData.channel_id == Channel.channel_id)
        .filter(
            func.date_trunc('month', UserData.last_message_at) == start_month,
            UserData.total_message_count > 0,
        )
    )
    if group:
        chat_engagement_q = chat_engagement_q.filter(Channel.channel_group == group)
    chat_engagement = chat_engagement_q.group_by(UserData.channel_id).cte('chat_engagement')
    ce = chat_engagement.c
    avg_messages_per_user = func.round(
        cast(ce.total_messages, Numeric) / func.nullif(ce.total_users, 0), 2
    ).label('avg_messages_per_user')
    try:
        results = (
            db.session.query(Channel.channel_name, ce.total_users, ce.total_messages, avg_messages_per_user)
            .select_from(chat_engagement)
            .join(Channel, ce.channel_id == Channel.channel_id)
            .order_by(avg_messages_per_user.desc())
            .all()
        )
        data = [
            {"channel": row.channel_name, "total_users": int(row.total_users), "total_messages": int(row.total_messages), "avg_messages_per_user": float(row.avg_messages_per_user)}
            for row in results if row.avg_messages_per_user is not None
        ]
        return {"success": True, "data": data}
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    

@api_bp.route('/api/get_stream_frequency', methods=['GET'])
@cached_json(
    lambda: f"stream_frequency_{request.args.get('channel_name')}_"
            f"{request.args.get('timezone', 'UTC')}_"
            f"{request.args.get('mode', 'span')}_"
            f"{request.args.get('resolution', '60')}_"
            f"{request.args.get('exclude_shorts', '0')}",
)
def get_stream_frequency():
    channel_name = request.args.get('channel_name')
    if not channel_name:
        return jsonify({"error": "channel_name is required"}), 400
    tz_name = request.args.get('timezone', 'UTC')
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = ZoneInfo('UTC')
    mode = request.args.get('mode', 'span')
    if mode not in ('span', 'start'):
        mode = 'span'
    # 60 = hourly (24 buckets), 30 = half-hourly (48 buckets)
    resolution = request.args.get('resolution', '60')
    step_minutes = 30 if resolution == '30' else 60
    num_slots = (24 * 60) // step_minutes  # 24 or 48
    exclude_shorts = request.args.get('exclude_shorts', '0').lower() in ('1', 'true', 'yes')
    min_duration = timedelta(minutes=10)
    channel = Channel.query.filter_by(channel_name=channel_name).first()
    if channel is None:
        return jsonify({"error": "channel not found"}), 404
    latest_video = (
        Video.query
        .filter(Video.channel_id == channel.channel_id)
        .order_by(Video.end_time.desc())
        .first()
    )
    frequency = {str(s): 0 for s in range(num_slots)}
    day_frequency = {str(d): 0 for d in range(7)}  # 0=Sun .. 6=Sat
    def slot_of(dt):
        """Bucket index for a local datetime at the chosen resolution."""
        return (dt.hour * 60 + dt.minute) // step_minutes
    if latest_video is None:
        return {
            "channel_name": channel_name,
            "timezone": tz_name,
            "mode": mode,
            "resolution": step_minutes,
            "frequency": frequency,
            "day_frequency": day_frequency,
        }
    latest_end_utc = latest_video.end_time.astimezone(ZoneInfo('UTC'))
    window_start = latest_end_utc - timedelta(days=365)
    query = (
        Video.query
        .filter(
            Video.channel_id == channel.channel_id,
            Video.end_time >= window_start,
            Video.end_time.isnot(None),
            Video.duration.isnot(None),
        )
    )
    if exclude_shorts:
        query = query.filter(Video.duration >= min_duration)
    videos = query.all()
    step = timedelta(minutes=step_minutes)
    for video in videos:
        end_time = video.end_time
        duration = video.duration
        if end_time is None or duration is None:
            continue
        end_utc = end_time.astimezone(ZoneInfo('UTC'))
        start_utc = end_utc - duration
        start_local = start_utc.astimezone(tz)
        end_local = end_utc.astimezone(tz)
        # ── START mode ──────────────────────────────────────
        if mode == 'start':
            frequency[str(slot_of(start_local))] += 1
            dow = (start_local.weekday() + 1) % 7
            day_frequency[str(dow)] += 1
            continue
        # ── SPAN mode ───────────────────────────────────────
        if end_local <= start_local:
            frequency[str(slot_of(end_local))] += 1
            dow = (end_local.weekday() + 1) % 7
            day_frequency[str(dow)] += 1
            continue
        # floor start to the resolution boundary
        floored_minute = (start_local.minute // step_minutes) * step_minutes
        cursor = start_local.replace(minute=floored_minute, second=0, microsecond=0)
        while cursor <= end_local:
            frequency[str(slot_of(cursor))] += 1
            cursor += step
        start_date = start_local.date()
        end_date = end_local.date()
        current_date = start_date
        seen_days: set[int] = set()
        while current_date <= end_date:
            dow = (current_date.weekday() + 1) % 7
            seen_days.add(dow)
            current_date += timedelta(days=1)
        for dow in seen_days:
            day_frequency[str(dow)] += 1
    return {
        "channel_name": channel_name,
        "timezone": tz_name,
        "mode": mode,
        "resolution": step_minutes,
        "frequency": frequency,
        "day_frequency": day_frequency,
    }


@api_bp.route('/api/get_stream_calendar', methods=['GET'])
@cached_json(
    lambda: f"stream_calendar_{request.args.get('channel_name')}_"
            f"{request.args.get('timezone', 'UTC')}_"
            f"{request.args.get('year', 'latest')}_"
            f"{request.args.get('exclude_shorts', '0')}",
)
def get_stream_calendar():
    channel_name = request.args.get('channel_name')
    if not channel_name:
        return jsonify({"error": "channel_name is required"}), 400
    tz_name = request.args.get('timezone', 'UTC')
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = ZoneInfo('UTC')
    exclude_shorts = request.args.get('exclude_shorts', '0').lower() in ('1', 'true', 'yes')
    min_duration = timedelta(minutes=10)
    channel = Channel.query.filter_by(channel_name=channel_name).first()
    if channel is None:
        return jsonify({"error": "channel not found"}), 404
    query = Video.query.filter(
        Video.channel_id == channel.channel_id,
        Video.end_time.isnot(None),
        Video.duration.isnot(None),
    )
    if exclude_shorts:
        query = query.filter(Video.duration >= min_duration)
    videos = query.all()
    # Convert every start time to the requested local timezone
    dates_count: dict[str, int] = {}
    all_years: set[int] = set()
    for video in videos:
        if video.end_time is None or video.duration is None:
            continue
        end_utc = video.end_time.astimezone(ZoneInfo('UTC'))
        start_local = (end_utc - video.duration).astimezone(tz)
        date_str = start_local.strftime('%Y-%m-%d')
        all_years.add(start_local.year)
        dates_count[date_str] = dates_count.get(date_str, 0) + 1
    available_years = sorted(all_years)
    if not available_years:
        return {
            "channel_name": channel_name,
            "timezone": tz_name,
            "year": None,
            "available_years": [],
            "calendar": {},
        }
    # Resolve requested year
    requested_year = request.args.get('year', 'latest')
    if requested_year != 'latest':
        try:
            year = int(requested_year)
            if year not in all_years:
                year = available_years[-1]
        except ValueError:
            year = available_years[-1]
    else:
        year = available_years[-1]
    # Filter to requested year only
    calendar = {
        date_str: count
        for date_str, count in dates_count.items()
        if date_str.startswith(str(year))
    }
    return {
        "channel_name": channel_name,
        "timezone": tz_name,
        "year": year,
        "available_years": available_years,
        "calendar": calendar,
    }


@api_bp.route('/api/get_video_highlights', methods=['GET'])
@cached_json(lambda: f"video_highlights_{(request.args.get('channel_name') or '').replace(' ', '_')}_{request.args.get('month')}")
def get_video_highlights():
    channel_name = request.args.get('channel_name')
    month_str = request.args.get('month')
    if not channel_name or not month_str:
        return jsonify({"success": False, "error": "Missing required parameters: 'channel_name' and 'month' are required."}), 400
    try:
        month_start_sql = parse_month(month_str)
    except ValueError:
        return jsonify({"success": False, "error": "Invalid month format. Please use 'YYYY-MM'."}), 400
    try:
        video_start_time = Video.end_time - Video.duration
        relative_seconds = extract('epoch', func.to_timestamp(VideoHighlight.start_seconds) - video_start_time).label('relative_seconds')
        results = (
            db.session.query(
                VideoHighlight.video_id,
                Video.title,
                VideoHighlight.topic_tag,
                VideoHighlight.generated_summary,
                relative_seconds,
            )
            .join(Video, VideoHighlight.video_id == Video.video_id)
            .join(Channel, Video.channel_id == Channel.channel_id)
            .filter(
                Channel.channel_name == channel_name,
                func.date_trunc('month', Video.end_time) == month_start_sql,
            )
            .order_by(Video.end_time.desc(), VideoHighlight.start_seconds.asc())
            .all()
        )
        if not results:
            return {"success": True, "data": []}
        videos_dict = {}
        for row in results:
            if row.relative_seconds < 0:
                continue
            if row.video_id not in videos_dict:
                videos_dict[row.video_id] = {"video_id": row.video_id, "video_title": row.title, "timestamps": []}
            videos_dict[row.video_id]["timestamps"].append({
                "topic": row.topic_tag,
                "summary": row.generated_summary,
                "youtube_url": f"https://www.youtube.com/watch?v={row.video_id}&t={int(row.relative_seconds)}s",
            })
        return {"success": True, "data": list(videos_dict.values())}
    except Exception as e:
        print(f"An error occurred in get_video_highlights: {e}")
        return jsonify({"success": False, "error": "An internal server error occurred."}), 500


@api_bp.route('/api/search_highlights', methods=['GET'])
def search_highlights():
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
        distance = VideoHighlight.summary_embedding.cosine_distance(query_vector).label('distance')
        query = (
            db.session.query(
                VideoHighlight.generated_summary,
                VideoHighlight.topic_tag,
                VideoHighlight.video_id,
                Video.title,
                Video.end_time,
                Video.duration,
                VideoHighlight.start_seconds,
                distance,
            )
            .join(Video, VideoHighlight.video_id == Video.video_id)
        )
        if filters["channel_name"]:
            query = (
                query.join(Channel, Video.channel_id == Channel.channel_id)
                     .filter(Channel.channel_name.ilike(f"%{filters['channel_name']}%"))
            )
        if filters["from_date"]:
            query = query.filter(Video.end_time >= filters["from_date"])
        if filters["to_date"]:
            to_date_exclusive = datetime.strptime(filters["to_date"], "%Y-%m-%d") + timedelta(days=1)
            query = query.filter(Video.end_time < to_date_exclusive)
        results = query.order_by(distance.asc()).limit(10).all()
        data = []
        for row in results:
            video_start_time = row.end_time - row.duration
            highlight_time = datetime.fromtimestamp(row.start_seconds, tz=timezone.utc)
            relative_seconds = (highlight_time - video_start_time).total_seconds()
            if relative_seconds < 0:
                continue
            data.append({
                "summary": row.generated_summary,
                "topic": row.topic_tag,
                "video_id": row.video_id,
                "video_title": row.title,
                "date": row.end_time.strftime('%Y-%m-%d'),
                "youtube_url": f"https://www.youtube.com/watch?v={row.video_id}&t={int(relative_seconds)}s",
                "similarity_score": 1 - row.distance,
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
            "image": f"https:{image.get('src')}" if image.get("src") else None,
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

@api_bp.route('/api/ccv/<video_id>')
def get_ccv(video_id):
    """Fetch concurrent viewer count for a YouTube live stream."""
    try:
        # Validate video ID format
        if not re.match(r'^[a-zA-Z0-9_-]{11}$', video_id):
            return jsonify({'error': 'Invalid video ID', 'ccv': None}), 400
        
        # Fetch the YouTube watch page
        url = f'https://www.youtube.com/watch?v={video_id}'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        html = response.text
        
        # Patterns ordered by priority - originalViewCount is the live CCV
        patterns = [
            # originalViewCount is the actual live concurrent viewer count
            r'"originalViewCount":\s*"(\d+)"',
            # Fallback patterns for "X watching now" text
            r'"watching now"[^}]*"simpleText":\s*"([\d,]+)',
            r'([\d,]+)\s*watching now',
            r'"viewCountText":\s*\{[^}]*"runs":\s*\[\s*\{[^}]*"text":\s*"([\d,]+)"[^}]*\}[^]]*"text":\s*"[^"]*watching',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, html)
            if match:
                count_str = match.group(1).replace(',', '')
                try:
                    ccv = int(count_str)
                    return jsonify({'ccv': ccv, 'video_id': video_id})
                except ValueError:
                    continue
        
        # If no live viewer count found, the stream might not be live
        return jsonify({'ccv': None, 'video_id': video_id, 'message': 'Stream may not be live'})
        
    except requests.RequestException as e:
        return jsonify({'error': str(e), 'ccv': None}), 500
    except Exception as e:
        return jsonify({'error': str(e), 'ccv': None}), 500
