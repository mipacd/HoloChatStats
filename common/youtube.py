import base64
import http.cookiejar
import os
import re
import json
import time
import sys
import logging
import copy
import hashlib
import requests
from yt_dlp import YoutubeDL
from common.config import secret

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
_AUTH = None
log = logging.getLogger("youtube")
_YID_ASSIGNMENT_RE = re.compile(
    r'(?:window\s*\[\s*["\']ytInitialData["\']\s*\]'
    r'|["\']?ytInitialData["\']?)\s*[:=]\s*')


class YoutubePayloadError(RuntimeError):
    """A successful YouTube HTTP response did not contain expected data."""


def _payload_diagnostic(response, stage):
    """Describe an unusable response without logging cookies or page text."""
    content = response.content or b""
    content_type = response.headers.get("Content-Type", "unknown")
    sample = content[:4096].lower()
    kind = "empty" if not content.strip() else "non-JSON"
    if b"consent.youtube" in sample or b"before you continue" in sample:
        kind = "YouTube consent page"
    elif b"accounts.google" in sample or b"sign in" in sample:
        kind = "YouTube sign-in page"
    elif b"unusual traffic" in sample or b"automated queries" in sample:
        kind = "YouTube bot-check page"
    return (f"{stage} returned {kind} payload "
            f"(HTTP {response.status_code}, content-type={content_type!r}, "
            f"bytes={len(content)})")

def _auth():
    """Materialize the Secrets Manager cookie only in Lambda's private /tmp."""
    global _AUTH
    if _AUTH is not None:
        return _AUTH
    data = secret(os.environ["YT_SECRET_ID"])
    user_agent = data.get("user_agent") or USER_AGENT
    cookie_path = None
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})
    encoded = data.get("cookies_b64")
    if encoded:
        raw = base64.b64decode(encoded, validate=True)
        text = raw.decode("utf-8-sig").replace("\r\n", "\n")
        if not text.startswith(("# HTTP Cookie File\n",
                                "# Netscape HTTP Cookie File\n")):
            raise RuntimeError("YouTube cookie secret is not Netscape format")
        cookie_path = "/tmp/youtube-cookies.txt"
        with open(cookie_path, "w", encoding="utf-8", newline="\n") as out:
            out.write(text)
        os.chmod(cookie_path, 0o600)
        jar = http.cookiejar.MozillaCookieJar(cookie_path)
        jar.load(ignore_discard=True, ignore_expires=True)
        session.cookies.update(jar)
        youtube_cookies = [c for c in jar
                           if c.domain.endswith(("youtube.com", "google.com"))]
        auth_names = {"SID", "HSID", "SSID", "APISID", "SAPISID",
                      "LOGIN_INFO", "__Secure-1PSID", "__Secure-3PSID",
                      "__Secure-1PAPISID", "__Secure-3PAPISID"}
        auth_cookies = [c for c in youtube_cookies if c.name in auth_names]
        # Names/counts are safe operational metadata; values are never logged.
        log.info("YouTube cookie jar loaded (cookies=%s youtube_google=%s "
                 "auth_markers=%s custom_user_agent=%s)",
                 len(jar), len(youtube_cookies), len(auth_cookies),
                 bool(data.get("user_agent")))
    _AUTH = {"cookiefile": cookie_path, "user_agent": user_agent,
             "session": session}
    return _AUTH

def _ydl_options():
    auth = _auth()
    opts = {"quiet": True, "noprogress": True,
            "user_agent": auth["user_agent"]}
    if auth["cookiefile"]:
        opts["cookiefile"] = auth["cookiefile"]
        # Current yt-dlp guidance for logged-in YouTube extraction avoids the
        # problematic default logged-in client while retaining an embedded
        # fallback that can handle age-gated metadata.
        opts["extractor_args"] = {
            "youtube": {"player_client": ["default", "web_embedded"]}
        }
    return opts


def _extract_video_info(url, attempts=6):
    """Run yt-dlp metadata extraction with retries for malformed/transient
    YouTube responses. Authentication and availability errors are returned
    immediately because retrying cannot repair the cookie/account state."""
    transient = ("unterminated string", "jsondecodeerror", "503",
                 "service unavailable", "timed out", "timeout",
                 "connection reset", "remote end closed",
                 "page needs to be reloaded")
    for attempt in range(attempts):
        try:
            with YoutubeDL(_ydl_options()) as ydl:
                return ydl.extract_info(url, download=False)
        except Exception as exc:
            if not any(marker in str(exc).lower() for marker in transient):
                raise
            if attempt + 1 >= attempts:
                raise
            time.sleep(min(2 ** attempt, 16))

def _fetch_html(url):
    """
    Retrieves HTML content from a given URL using a custom User-Agent header. Makes a GET request
    with a 20-second timeout and raises an exception if the request fails. Returns the raw HTML
    text for further processing and extraction of embedded data.
    
    Args:
        url (str): The URL to fetch HTML from
    
    Returns:
        str: The HTML content as text
    
    Raises:
        requests.exceptions.HTTPError: If HTTP request fails
        requests.exceptions.Timeout: If request exceeds 20 seconds
        requests.exceptions.RequestException: For other network errors
    """
    r = _auth()["session"].get(url, timeout=20)
    r.raise_for_status()
    return r.text

def _render_runs(runs):
    """
    Convert a list of YouTube message 'runs' into a flat string.
    Text runs are passed through; emoji runs are rendered as their
    shortcode (custom emotes) or the unicode character (standard emoji).
    """
    parts = []
    for run in runs or []:
        if "text" in run:
            parts.append(run["text"])
        elif "emoji" in run:
            e = run["emoji"]
            if e.get("isCustomEmoji"):
                shortcuts = e.get("shortcuts") or []
                if shortcuts:
                    parts.append(shortcuts[0])
                else:
                    label = (
                        e.get("image", {})
                         .get("accessibility", {})
                         .get("accessibilityData", {})
                         .get("label", "emoji")
                    )
                    parts.append(f":{label}:")
            else:
                # Standard unicode emoji – emojiId is the character itself
                parts.append(e.get("emojiId", ""))
    return "".join(parts)

def _extract_params(html):
    """
    Parses HTML to extract YouTube API parameters including the API key, client version, and initial
    data object. Uses regex patterns to find embedded JavaScript values. Returns default version if
    not found. Essential for making authenticated API requests to YouTube's backend services.
    
    Args:
        html (str): Raw HTML content from YouTube page
    
    Returns:
        tuple: (api_key (str or None), version (str), yid (dict or None))
    
    Raises:
        json.JSONDecodeError: If ytInitialData JSON is malformed
    """
    key_m = re.search(r'INNERTUBE_API_KEY["\']\s*:\s*"([^"]+)"', html)
    ver_m = re.search(r'INNERTUBE_CONTEXT_CLIENT_VERSION["\']\s*:\s*"([^"]+)"', html)
    api_key = key_m.group(1) if key_m else None
    version = ver_m.group(1) if ver_m else "2.20201021.03.00"
    yid = None
    decode_error = None
    # A watch page can mention ytInitialData before the real assignment.  Try
    # every candidate instead of letting one incidental/truncated match hide a
    # later valid object.
    for yid_m in _YID_ASSIGNMENT_RE.finditer(html):
        start = html.find("{", yid_m.end())
        if start >= 0:
            try:
                # raw_decode understands nesting and escaped braces inside
                # strings; the old non-greedy regex cut valid JSON short.
                candidate, _end = json.JSONDecoder().raw_decode(html[start:])
                if isinstance(candidate, dict):
                    yid = candidate
                    break
            except json.JSONDecodeError as exc:
                decode_error = exc
    if yid is None and decode_error is not None:
        raise decode_error
    return api_key, version, yid

def _extract_innertube_context(html, version):
    """Extract the complete client context YouTube issued with the page.

    Replay requests increasingly require fields such as visitorData in
    addition to the client version. Keep the minimal context only as a
    compatibility fallback for unusual watch-page variants.
    """
    merged = {}
    for marker in re.finditer(r'ytcfg\.set\s*\(\s*', html):
        start = html.find("{", marker.end())
        if start < 0:
            continue
        try:
            value, _end = json.JSONDecoder().raw_decode(html[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            merged.update(value)
    context = merged.get("INNERTUBE_CONTEXT")
    if isinstance(context, dict) and isinstance(context.get("client"), dict):
        return context
    return {"client": {"clientName": "WEB", "clientVersion": version}}


def _fetch_params(url, attempts=8):
    """Fetch and decode watch-page parameters, retrying truncated HTML."""
    for attempt in range(attempts):
        try:
            html = _fetch_html(url)
            api_key, version, yid = _extract_params(html)
            return api_key, version, yid, _extract_innertube_context(html, version)
        except json.JSONDecodeError:
            if attempt + 1 >= attempts:
                raise
            time.sleep(2 ** attempt)

_CONTINUATION_KINDS = (
    "liveChatReplayContinuationData", "reloadContinuationData",
    "timedContinuationData", "invalidationContinuationData",
)

def _continuation_from_entries(entries):
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        for kind in _CONTINUATION_KINDS:
            data = entry.get(kind)
            if isinstance(data, dict) and data.get("continuation"):
                return data["continuation"]
    return None

def _find_live_chat_renderer(value):
    if isinstance(value, dict):
        renderer = value.get("liveChatRenderer")
        if isinstance(renderer, dict):
            return renderer
        for child in value.values():
            found = _find_live_chat_renderer(child)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_live_chat_renderer(child)
            if found is not None:
                return found
    return None

def _find_continuation(ytInitialData):
    """
    Recursively searches through the YouTube initial data structure to find a continuation token. This
    token is required for fetching chat replay data. Walks through dictionaries and lists looking for
    the 'continuation' key. Returns the first continuation token found or None if absent.
    
    Args:
        ytInitialData (dict): YouTube's initial data object
    
    Returns:
        str or None: Continuation token if found, None otherwise
    
    Raises:
        None
    """
    renderer = _find_live_chat_renderer(ytInitialData)
    if renderer is not None:
        # Do not fall through to an unrelated comments/recommendations token
        # when a live-chat renderer is present.
        return _continuation_from_entries(renderer.get("continuations"))

    def walk(d):
        # Check if current element is a dictionary and search for continuation key
        if isinstance(d, dict):
            # Return continuation value if key exists in current dictionary
            if "continuation" in d:
                return d["continuation"]
            # Recursively search all dictionary values for continuation token
            for v in d.values():
                res = walk(v)
                # Return result if continuation token was found in nested structure
                if res:
                    return res
        # Check if current element is a list and search each item
        elif isinstance(d, list):
            # Iterate through list items searching for continuation token recursively
            for i in d:
                res = walk(i)
                # Return result if continuation token was found in list item
                if res:
                    return res
        return None
    return walk(ytInitialData)

def _fetch_chat(api_key, version, continuation, context=None,
                player_offset_ms=None, click_tracking=None):
    """
    Makes a POST request to YouTube's API endpoint to retrieve live chat replay data. Uses the
    continuation token to paginate through chat messages. Includes proper headers and context for authentication.
    Returns JSON response containing chat actions and potentially the next continuation token for pagination.
    
    Args:
        api_key (str): YouTube API key
        version (str): Client version string
        continuation (str): Continuation token for pagination
    
    Returns:
        dict: JSON response containing chat data
    
    Raises:
        requests.exceptions.HTTPError: If API request fails
        requests.exceptions.Timeout: If request exceeds 60 seconds
        requests.exceptions.RequestException: For other network errors
        json.JSONDecodeError: If response is not valid JSON
    """
    url = f"https://www.youtube.com/youtubei/v1/live_chat/get_live_chat_replay?key={api_key}"
    request_context = copy.deepcopy(context) if context else {
        "client": {"clientName": "WEB", "clientVersion": version}}
    if click_tracking:
        request_context["clickTracking"] = {
            "clickTrackingParams": click_tracking}
    data = {"context": request_context, "continuation": continuation}
    if player_offset_ms is not None:
        data["currentPlayerState"] = {
            "playerOffsetMs": str(max(int(player_offset_ms) - 5000, 0))}
    headers = {
        "Content-Type": "application/json",
        "Origin": "https://www.youtube.com",
        "X-Origin": "https://www.youtube.com",
        "Referer": ("https://www.youtube.com/live_chat_replay?continuation="
                    + continuation),
        "X-Youtube-Client-Name": ("1" if request_context.get(
            "client", {}).get("clientName") == "WEB" else str(
                request_context.get("client", {}).get("clientName", "1"))),
        "X-Youtube-Client-Version": version,
        "X-Goog-AuthUser": "0",
    }
    visitor_data = request_context.get("client", {}).get("visitorData")
    if visitor_data:
        headers["X-Goog-Visitor-Id"] = visitor_data
    session = _auth()["session"]
    cookie_values = {}
    try:
        cookies = iter(session.cookies)
    except TypeError:
        cookies = iter(())
    for cookie in cookies:
        if cookie.name in ("SAPISID", "__Secure-1PAPISID",
                           "__Secure-3PAPISID", "LOGIN_INFO"):
            cookie_values[cookie.name] = cookie.value
    # Match YouTube's current cookie authentication. SAPISIDHASH falls back
    # to the 3P cookie when SAPISID is absent; the dedicated 1P/3P hashes are
    # also sent when those cookies exist. Cookie values never enter logs.
    auth_cookies = (
        ("SAPISIDHASH", cookie_values.get("SAPISID")
         or cookie_values.get("__Secure-3PAPISID")),
        ("SAPISID1PHASH", cookie_values.get("__Secure-1PAPISID")),
        ("SAPISID3PHASH", cookie_values.get("__Secure-3PAPISID")),
    )
    if any(value for _scheme, value in auth_cookies):
        timestamp = round(time.time())
        authorization = []
        for scheme, value in auth_cookies:
            if not value:
                continue
            digest = hashlib.sha1(
                f"{timestamp} {value} https://www.youtube.com".encode()
            ).hexdigest()
            authorization.append(f"{scheme} {timestamp}_{digest}")
        headers["Authorization"] = " ".join(authorization)
    if cookie_values.get("LOGIN_INFO"):
        headers["X-Youtube-Bootstrap-Logged-In"] = "true"
    retryable = (
        requests.exceptions.Timeout,
        requests.exceptions.ConnectionError,
        requests.exceptions.ChunkedEncodingError,
        json.JSONDecodeError,
        YoutubePayloadError,
        requests.exceptions.HTTPError,
    )
    # A replay can contain thousands of pages.  YouTube occasionally returns a
    # truncated JSON body or a short burst of 429/5xx responses for one page;
    # losing the entire Lambda invocation for that is both slow and likely to
    # hit the same continuation again.  Keep retries local to the page.
    for attempt in range(8):
        try:
            r = session.post(
                url, headers=headers, json=data,
                timeout=60)
            r.raise_for_status()
            # Decode with the stdlib so every malformed/truncated response has
            # the same exception type across requests releases.
            try:
                decoded = json.loads(r.content)
            except json.JSONDecodeError as exc:
                raise YoutubePayloadError(
                    _payload_diagnostic(r, "InnerTube replay API")) from exc
            if not isinstance(decoded, dict):
                raise YoutubePayloadError(
                    "InnerTube replay API returned a JSON value that is not "
                    "an object")
            return decoded
        except retryable as exc:
            response = getattr(exc, "response", None)
            status = getattr(response, "status_code", None)
            if (isinstance(exc, requests.exceptions.HTTPError)
                    and status != 429 and (status is None or status < 500)):
                raise
            if attempt == 7:
                raise
            retry_after = (response.headers.get("Retry-After")
                           if response is not None else None)
            try:
                delay = float(retry_after) if retry_after is not None else 2 ** attempt
            except (TypeError, ValueError):
                delay = 2 ** attempt
            time.sleep(max(0.25, min(delay, 30)))

def _parse_messages(actions, video_start_ts):
    """
    Extracts and formats chat messages from YouTube API response actions. Processes regular messages,
    paid messages, new membership notifications, and gift membership recipient notifications. Extracts author information, badges,
    message text, and timestamps. Filters out invalid messages and calculates absolute timestamps based on
    video start time. Returns structured message data.
    
    Args:
        actions (list or None): List of chat actions from API response
        video_start_ts (float): Unix timestamp of video start
    
    Returns:
        list: List of dictionaries containing parsed message data
    
    Raises:
        ValueError: If timestamp conversion fails
        KeyError: If expected data structure is missing (caught internally)
    """
    msgs = []
    # Iterate through all chat actions or empty list if actions is None
    for a in actions or []:
        # Skip actions that aren't replay chat items
        if "replayChatItemAction" not in a:
            continue

        replay_action = a["replayChatItemAction"]
        item = replay_action.get("actions", [{}])[0]
        chat = item.get("addChatItemAction", {}).get("item", {})
        
        # Extract video offset from the replay action itself
        video_offset_time_msec = replay_action.get("videoOffsetTimeMsec")
        if video_offset_time_msec is None:
            # Try alternative location
            video_offset_time_msec = item.get("addChatItemAction", {}).get("videoOffsetTimeMsec")
        
        # Skip if we still don't have a valid offset
        if video_offset_time_msec is None:
            continue
            
        offset_ms = int(float(video_offset_time_msec))
        # Skip messages with negative offsets as they are invalid
        if offset_ms < 0:
            continue

        # Check for regular text messages, paid messages, memberships, and gift memberships
        for t in ("liveChatTextMessageRenderer", 
                  "liveChatPaidMessageRenderer",
                  "liveChatMembershipItemRenderer",
                  "liveChatSponsorshipsGiftRedemptionAnnouncementRenderer"):
            # Skip if current message type not found in chat item
            if t not in chat:
                continue

            r = chat[t]

            author = {
                "id": r.get("authorExternalChannelId", None),
                "name": r.get("authorName", {}).get("simpleText", "").strip(),
                "badges": [],
            }

            # Extract all author badges like membership status or moderator badges
            for badge in r.get("authorBadges", []) or []:
                badge_label = badge.get("liveChatAuthorBadgeRenderer", {}).get("tooltip", "")
                # Add badge label if it exists and is not empty
                if badge_label:
                    author["badges"].append(badge_label)

            # Skip messages with no author name as they are likely invalid
            if not author["name"]:
                continue

            msg = ""
            msg_type = "chat"
            msg_data = {
                "author": author,
                "timestamp": 0  # Will be set later
            }
            
            # Handle regular chat messages with runs
            if t in ("liveChatTextMessageRenderer", "liveChatPaidMessageRenderer"):
                msg_runs = r.get("message", {}).get("runs", [])
                msg = _render_runs(msg_runs).strip()
                msg_data["message"] = msg
                if t == "liveChatPaidMessageRenderer":
                    msg_type = "paid_message"
                msg_data["message_type"] = msg_type
            
            # Handle new membership notifications
            elif t == "liveChatMembershipItemRenderer":
                msg_type = "new_member"
                msg_data["message"] = ""  # Empty string for new member messages
                msg_data["message_type"] = msg_type
            
            # Handle gift membership redemption notifications (recipients only)
            elif t == "liveChatSponsorshipsGiftRedemptionAnnouncementRenderer":
                msg_type = "gift_member"
                msg_data["message"] = ""  # Empty string for gift messages
                msg_data["message_type"] = msg_type
                
                # Extract gifter username from the message
                msg_runs = r.get("message", {}).get("runs", [])
                full_text = _render_runs(msg_runs)
                gifter_match = re.search(r'by\s+(\S+)', full_text)
                gifter = gifter_match.group(1) if gifter_match else None
                # Look for the gifter's name in the runs (usually after "by" text)
                if msg_runs:
                    full_text = "".join([x.get("text", "") for x in msg_runs])
                    # Try to extract username after "by " pattern, preserving @ if present
                    gifter_match = re.search(r'by\s+(\S+)', full_text)
                    if gifter_match:
                        gifter = gifter_match.group(1)
                
                # Add gifter field if found
                if gifter:
                    msg_data["gifter"] = gifter
            
            # Skip if it's a regular/paid message with no content
            if t in ("liveChatTextMessageRenderer", "liveChatPaidMessageRenderer") and not msg:
                continue

            timestamp = video_start_ts + (offset_ms / 1000.0)
            msg_data["timestamp"] = timestamp

            msgs.append(msg_data)
    return msgs


def _live_chat_continuation(obj):
    if not isinstance(obj, dict):
        return None
    return (obj.get("continuationContents", {})
            .get("liveChatContinuation"))

def _extract_unfiltered_cont(obj):
    """Return YouTube's 'Live chat replay' selector rather than Top chat."""
    return _extract_unfiltered_cont_data(obj)[0]

def _extract_unfiltered_cont_data(obj):
    """Return the unfiltered replay token and its click-tracking value."""
    live = _live_chat_continuation(obj) or {}
    items = (live.get("header", {}).get("liveChatHeaderRenderer", {})
             .get("viewSelector", {}).get("sortFilterSubMenuRenderer", {})
             .get("subMenuItems", []))
    # yt-dlp uses the second item; prefer it but tolerate reordered variants.
    ordered = (items[1:] + items[:1]) if len(items) > 1 else items
    for item in ordered:
        data = (item.get("continuation", {}).get("reloadContinuationData", {})
                if isinstance(item, dict) else {})
        if data.get("continuation"):
            return (data["continuation"],
                    data.get("clickTrackingParams")
                    or data.get("trackingParams"))
    return None, None

def _extract_next_cont(obj):
    """
    Recursively searches through a nested data structure to find the next continuation token for
    pagination. Similar to _find_continuation but used for extracting tokens from API responses. Handles both
    dictionary and list structures. Returns the first continuation token found or None if absent.
    
    Args:
        obj (dict, list, or any): Data structure to search
    
    Returns:
        str or None: Continuation token if found, None otherwise
    
    Raises:
        None
    """
    live = _live_chat_continuation(obj)
    if isinstance(live, dict):
        # Restrict selection to the live-chat continuation list. The response
        # can also contain unrelated continuations for menus and banners.
        return _continuation_from_entries(live.get("continuations"))

    # Compatibility fallback for saved fixtures and older response shapes.
    # Check if object is dictionary and search for continuation key
    if isinstance(obj, dict):
        # Iterate through all key-value pairs in the dictionary
        for k, v in obj.items():
            # Return value if key matches continuation
            if k == "continuation":
                return v
            res = _extract_next_cont(v)
            # Return result if continuation found in nested structure
            if res:
                return res
    # Check if object is list and search each element
    elif isinstance(obj, list):
        # Recursively search each item in the list for continuation
        for i in obj:
            res = _extract_next_cont(i)
            # Return result if continuation found in list element
            if res:
                return res
    return None

def _extract_next_cont_data(obj):
    """Return the replay token plus its optional click-tracking parameter."""
    live = _live_chat_continuation(obj)
    if isinstance(live, dict):
        for entry in live.get("continuations") or []:
            if not isinstance(entry, dict):
                continue
            for kind in _CONTINUATION_KINDS:
                data = entry.get(kind)
                if isinstance(data, dict) and data.get("continuation"):
                    return data["continuation"], data.get("clickTrackingParams")
        return None, None
    return _extract_next_cont(obj), None

def _fetch_initial_chat(continuation, attempts=8):
    """Load the replay bootstrap page before using the InnerTube POST API."""
    url = "https://www.youtube.com/live_chat_replay"
    for attempt in range(attempts):
        try:
            response = _auth()["session"].get(
                url, params={"continuation": continuation}, timeout=60)
            response.raise_for_status()
            _key, _version, initial = _extract_params(response.text)
            if initial:
                return initial
            try:
                decoded = json.loads(response.content)
            except json.JSONDecodeError as exc:
                raise YoutubePayloadError(
                    _payload_diagnostic(response,
                                        "live-chat replay bootstrap")) from exc
            if not isinstance(decoded, dict):
                raise YoutubePayloadError(
                    "live-chat replay bootstrap returned a JSON value that "
                    "is not an object")
            return decoded
        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError,
                requests.exceptions.HTTPError,
                json.JSONDecodeError,
                YoutubePayloadError) as exc:
            response = getattr(exc, "response", None)
            status = getattr(response, "status_code", None)
            if (isinstance(exc, requests.exceptions.HTTPError)
                    and status != 429 and (status is None or status < 500)):
                raise
            if attempt + 1 >= attempts:
                raise
            time.sleep(min(2 ** attempt, 30))

def iter_youtube_chat(video_id):
    """
    Generator function that yields chat messages from a YouTube video replay. Fetches video metadata,
    extracts API parameters, and iteratively retrieves chat messages using continuation tokens. Handles pagination
    and deduplication. Yields individual messages as dictionaries containing author info, text, and timestamp.
    
    Args:
        video_id (str): YouTube video ID
    
    Yields:
        dict: Individual chat message with author, message, and timestamp fields
    
    Raises:
        RuntimeError: If ytInitialData or continuation token not found
        requests.exceptions.RequestException: For network-related errors
        yt_dlp.utils.DownloadError: If video info extraction fails
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    info = _extract_video_info(url)
    duration = info.get("duration", 0)
    video_start_ts = info.get("release_timestamp") or info.get("timestamp") or 0

    api_key, version, yid, context = _fetch_params(url)
    # Check if initial data was found, raise error if missing
    if not yid:
        raise RuntimeError("ytInitialData not found — possibly need cookies")

    continuation = _find_continuation(yid)
    # Check if continuation token exists, raise error if not found
    if not continuation:
        raise RuntimeError("No continuation found")

    # YouTube's first replay request is an HTML bootstrap endpoint. It yields
    # the unfiltered live-chat token used by subsequent JSON API requests.
    first = _fetch_initial_chat(continuation)
    continuation = (_extract_unfiltered_cont(first)
                    or _extract_next_cont(first))
    seen = set()
    # Continue fetching chat pages while continuation tokens are available
    while continuation:
        # Check for duplicate continuation tokens to prevent infinite loops
        if continuation in seen:
            break
        seen.add(continuation)

        data = _fetch_chat(api_key, version, continuation, context=context)
        actions = data.get("actions") or data.get("continuationContents", {}).get(
            "liveChatContinuation", {}
        ).get("actions")

        # Yield each parsed message from the current batch of actions
        for msg in _parse_messages(actions, video_start_ts):
            yield msg

        continuation = _extract_next_cont(data)
        time.sleep(0.08)


def main():
    """
    Main function for command-line execution. Accepts video ID and output filename as arguments, retrieves
    all chat messages from the specified YouTube video, and writes them to a JSON file. Each message
    includes author details, message text, and timestamp. Provides error handling and user feedback during execution.
    
    Args:
        None (reads from sys.argv)
    
    Returns:
        None
    
    Raises:
        SystemExit: If incorrect number of arguments provided
        IOError: If output file cannot be written
        Exception: Any exception from iter_youtube_chat
    """
    # Check if correct number of command-line arguments were provided
    if len(sys.argv) != 3:
        print("Usage: python script.py <video_id> <output_file>")
        print("Example: python script.py dQw4w9WgXcQ chat_log.json")
        sys.exit(1)
    
    video_id = sys.argv[1]
    output_file = sys.argv[2]
    
    print(f"Fetching chat messages for video: {video_id}")
    
    messages = []
    
    try:
        # Iterate through all chat messages and collect them in a list
        for msg in iter_youtube_chat(video_id):
            messages.append(msg)
            # Print progress indicator every 100 messages to show activity
            if len(messages) % 100 == 0:
                print(f"Collected {len(messages)} messages...")
        
        print(f"\nTotal messages collected: {len(messages)}")
        
        # Write all collected messages to output file in JSON format
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
        
        print(f"Chat log saved to: {output_file}")
    
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
        sys.exit(1)
    except IOError as e:
        print(f"File error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

class ChatReplay:
    """Resumable chat-replay reader.
    State that must survive across Lambda invocations: api_key, client version,
    continuation token, video_start_ts. api_key/version are cheap to re-derive
    and can expire, so we refresh them on every cold resume and only persist
    the continuation token + start ts.
    """
    def __init__(self, video_id, continuation=None, video_start_ts=None,
                 duration=None, player_offset_s=None):
        self.video_id = video_id
        url = f"https://www.youtube.com/watch?v={video_id}"
        # Discovery already records the video's end time and duration.  Use
        # that metadata even for the first replay page so chat downloads do
        # not unnecessarily enter yt-dlp's media-player JS challenge path.
        # Keep yt-dlp as a fallback for old/manual jobs without metadata.
        if video_start_ts is None:
            info = _extract_video_info(url)
            self.duration = info.get("duration") or duration or 0
            self.video_start_ts = (info.get("release_timestamp")
                                   or info.get("timestamp") or 0)
        else:
            self.duration = duration or 0
            self.video_start_ts = video_start_ts
        params = _fetch_params(url)
        self.api_key, self.version, yid = params[:3]
        self.context = (params[3] if len(params) > 3 else {
            "client": {"clientName": "WEB", "clientVersion": self.version}})
        if not yid:
            raise RuntimeError("ytInitialData not found — possibly need cookies")
        self._needs_bootstrap = continuation is None
        self.continuation = continuation or _find_continuation(yid)
        if not self.continuation:
            raise RuntimeError("No continuation found")
        self.player_offset_ms = (int(float(player_offset_s) * 1000)
                                 if player_offset_s is not None else 0)
        self.click_tracking = None
        self._seen = set()
    def pages(self):
        """Yield (messages, continuation_after_this_page). Caller decides when
        to stop; `continuation` is the token to persist to resume here."""
        if self._needs_bootstrap:
            data = _fetch_initial_chat(self.continuation)
            unfiltered, tracking = _extract_unfiltered_cont_data(data)
            if unfiltered:
                self.continuation = unfiltered
                self.click_tracking = tracking
            else:
                live = _live_chat_continuation(data) or {}
                nxt, self.click_tracking = _extract_next_cont_data(data)
                messages = _parse_messages(live.get("actions"),
                                           self.video_start_ts)
                if messages:
                    self.player_offset_ms = max(
                        self.player_offset_ms,
                        int((messages[-1]["timestamp"]
                             - self.video_start_ts) * 1000))
                yield messages, nxt
                self.continuation = nxt
            self._needs_bootstrap = False
        while self.continuation:
            if self.continuation in self._seen:
                return
            self._seen.add(self.continuation)
            data = _fetch_chat(
                self.api_key, self.version, self.continuation,
                context=self.context, player_offset_ms=self.player_offset_ms,
                click_tracking=self.click_tracking)
            actions = data.get("actions") or data.get("continuationContents", {}) \
                .get("liveChatContinuation", {}).get("actions")
            nxt, self.click_tracking = _extract_next_cont_data(data)
            messages = _parse_messages(actions, self.video_start_ts)
            if messages:
                self.player_offset_ms = max(
                    self.player_offset_ms,
                    int((messages[-1]["timestamp"]
                         - self.video_start_ts) * 1000))
            yield messages, nxt
            self.continuation = nxt
            time.sleep(0.08)


# Execute main function only when script is run directly
if __name__ == "__main__":
    main()
