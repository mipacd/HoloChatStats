
from flask import Blueprint, render_template, redirect, request, session, Response
from flask_babel import _
from utils import get_locale

routes_bp = Blueprint('routes', __name__)


@routes_bp.route('/')
def index():
    user_agent = request.headers.get('User-Agent', '').lower()
    is_mobile = any(x in user_agent for x in ['mobile', 'android', 'iphone'])
    if is_mobile:
        return render_template("index_pwa.html", _=_, get_locale=get_locale)
    else:
        return render_template("index.html", _=_, get_locale=get_locale)

# Serve service worker from root
@routes_bp.route('/service-worker.js')
def service_worker():
    return routes_bp.send_static_file('service-worker.js'), 200, {
        'Content-Type': 'application/javascript',
        'Service-Worker-Allowed': '/'
    }


@routes_bp.route('/set_language/<language>')
def set_language(language):
    if language in ['en', 'ja', 'ko']:
        session['language'] = language
        print("Language stored in session:", session['language'])
        return redirect(request.referrer or "/")
    return "Invalid language", 400


@routes_bp.route('/streaming_hours')
def streaming_hours_view():
    return render_template('streaming_hours.html', _=_, get_locale=get_locale)


@routes_bp.route('/streaming_hours_avg')
def streaming_hours_avg_view():
    return render_template('streaming_hours_avg.html', _=_, get_locale=get_locale)


@routes_bp.route('/streaming_hours_max')
def streaming_hours_max_view():
    return render_template('streaming_hours_max.html', _=_, get_locale=get_locale)


@routes_bp.route('/streaming_hours_diff')
def streaming_hours_diff_view():
    return render_template('streaming_hours_diff.html', _=_, get_locale=get_locale)


@routes_bp.route('/chat_makeup')
def chat_makeup_view():
    return render_template('chat_makeup.html', _=_, get_locale=get_locale)


@routes_bp.route('/common_users')
def common_users_view():
    return render_template('common_users.html', _=_, get_locale=get_locale)


@routes_bp.route('/membership_counts')
def membership_counts_view():
    user_agent = request.headers.get('User-Agent', '').lower()
    is_mobile = any(x in user_agent for x in ['mobile', 'android', 'iphone'])
    if is_mobile:
        return render_template('membership_counts_pwa.html', _=_, get_locale=get_locale)
    else:
        return render_template('membership_counts.html', _=_, get_locale=get_locale)


@routes_bp.route('/membership_percentages')
def membership_percentages_view():
    return render_template('membership_percentages.html', _=_, get_locale=get_locale)


@routes_bp.route('/membership_change')
def membership_expirations_view():
    return render_template('membership_change.html', _=_, get_locale=get_locale)


@routes_bp.route('/chat_leaderboards')
def chat_leaderboard_view():
    return render_template('chat_leaderboards.html', _=_, get_locale=get_locale)


@routes_bp.route('/user_change')
def user_changes_view():
    return render_template('user_change.html', _=_, get_locale=get_locale)


@routes_bp.route('/monthly_streaming_hours')
def monthly_streaming_hours_view():
    return render_template('monthly_streaming_hours.html', _=_, get_locale=get_locale)


@routes_bp.route('/exclusive_chat')
def exclusive_chat_users_view():
    return render_template('exclusive_chat.html', _=_, get_locale=get_locale)


@routes_bp.route('/message_types')
def message_types_view():
    return render_template('message_types.html', _=_, get_locale=get_locale)


@routes_bp.route('/funniest_timestamps')
def funniest_timestamps_view():
    return render_template('funniest_timestamps.html', _=_, get_locale=get_locale)


@routes_bp.route('/common_members')
def common_members_view():
    return render_template('common_members.html', _=_, get_locale=get_locale)


@routes_bp.route('/channel_clustering')
def channel_clustering_view():
    return render_template('channel_clustering.html', _=_, get_locale=get_locale)


@routes_bp.route('/jp_user_percents')
def jp_user_percents_view():
    return render_template('jp_user_percents.html', _=_, get_locale=get_locale)


@routes_bp.route('/user_info')
def user_info_view():
    return render_template('user_info.html', _=_, get_locale=get_locale)


@routes_bp.route('/engagement')
def engagement_redirect():
    return render_template('engagement.html', _=_, get_locale=get_locale)


@routes_bp.route('/site_metrics')
def site_metrics_view():
    return render_template('site_metrics.html', _=_, get_locale=get_locale)


@routes_bp.route('/recommendation_engine')
def recommendation_engine_view():
    return render_template('recommendation_engine.html', _=_, get_locale=get_locale)


@routes_bp.route('/highlights')
def highlights_view():
    return render_template('highlights.html', _=_, get_locale=get_locale)


@routes_bp.route('/highlight_search')
def highlight_search_view():
    return render_template('highlight_search.html', _=_, get_locale=get_locale)


@routes_bp.route('/heatmap')
def heatmap_view():
    return render_template('heatmap.html', _=_, get_locale=get_locale)


@routes_bp.route("/eri")
def eri_chat():
    return render_template("eri.html", _=_, get_locale=get_locale)


# v1 redirects
@routes_bp.route('/stream-time')
def stream_time_redirect():
    return redirect("/streaming_hours")


@routes_bp.route('/nonjp-holojp')
def nonjp_holojp_redirect():
    return redirect("/message_types")


@routes_bp.route('/jp-holoiden')
def holoiden_redirect():
    return redirect("/message_types")


@routes_bp.route('/chat-makeup')
def chat_makeup_redirect():
    return redirect("/chat_makeup")


@routes_bp.route('/langsum')
def langsum_redirect():
    return redirect("/message_types")


@routes_bp.route('/en-livetl')
def livetl_redirect():
    return redirect("https://old.holochatstats.info/en-livetl")


@routes_bp.route('/en-tl-stream')
def en_tl_stream_redirect():
    return redirect("https://old.holochatstats.info/en-tl-stream")


@routes_bp.route('/common-chat')
def common_chat_redirect():
    return redirect("/common_users")


@routes_bp.route('/excl-chat')
def excl_chat_redirect():
    return redirect("/exclusive_chat")


@routes_bp.route('/members')
def members_redirect():
    return redirect("/membership_counts")


@routes_bp.route('/member-percent')
def member_percent_redirect():
    return redirect("/membership_percentages")


@routes_bp.route('/stream-time-series')
def stream_time_series_redirect():
    return redirect("/monthly_streaming_hours")


@routes_bp.route('/coverage')
def coverage_redirect():
    return redirect("https://old.holochatstats.info/coverage")


@routes_bp.route('/robots.txt')
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