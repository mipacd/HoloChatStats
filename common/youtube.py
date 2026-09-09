import base64
import http.cookiejar
import os
import re
import json
import time
import sys
import logging
import requests
from yt_dlp import YoutubeDL
from common.config import secret

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
_AUTH = None
log = logging.getLogger("youtube")

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
                 "connection reset", "remote end closed")
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
    for yid_m in re.finditer(r'ytInitialData["\']?\s*[:=]\s*', html):
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


def _fetch_params(url, attempts=8):
    """Fetch and decode watch-page parameters, retrying truncated HTML."""
    for attempt in range(attempts):
        try:
            return _extract_params(_fetch_html(url))
        except json.JSONDecodeError:
            if attempt + 1 >= attempts:
                raise
            time.sleep(2 ** attempt)

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

def _fetch_chat(api_key, version, continuation):
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
    data = {
        "context": {"client": {"clientName": "WEB", "clientVersion": version}},
        "continuation": continuation,
    }
    retryable = (
        requests.exceptions.Timeout,
        requests.exceptions.ConnectionError,
        requests.exceptions.ChunkedEncodingError,
        json.JSONDecodeError,
        requests.exceptions.HTTPError,
    )
    # A replay can contain thousands of pages.  YouTube occasionally returns a
    # truncated JSON body or a short burst of 429/5xx responses for one page;
    # losing the entire Lambda invocation for that is both slow and likely to
    # hit the same continuation again.  Keep retries local to the page.
    for attempt in range(8):
        try:
            r = _auth()["session"].post(
                url, headers={"Content-Type": "application/json"}, json=data,
                timeout=60)
            r.raise_for_status()
            # Decode with the stdlib so every malformed/truncated response has
            # the same exception type across requests releases.
            decoded = json.loads(r.content)
            if not isinstance(decoded, dict):
                raise json.JSONDecodeError("YouTube response is not an object",
                                           r.text, 0)
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

    api_key, version, yid = _fetch_params(url)
    # Check if initial data was found, raise error if missing
    if not yid:
        raise RuntimeError("ytInitialData not found — possibly need cookies")

    continuation = _find_continuation(yid)
    # Check if continuation token exists, raise error if not found
    if not continuation:
        raise RuntimeError("No continuation found")

    seen = set()
    # Continue fetching chat pages while continuation tokens are available
    while continuation:
        # Check for duplicate continuation tokens to prevent infinite loops
        if continuation in seen:
            break
        seen.add(continuation)

        data = _fetch_chat(api_key, version, continuation)
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
    def __init__(self, video_id, continuation=None, video_start_ts=None):
        self.video_id = video_id
        url = f"https://www.youtube.com/watch?v={video_id}"
        if video_start_ts is None or continuation is None:
            info = _extract_video_info(url)
            self.duration = info.get("duration") or 0
            self.video_start_ts = (info.get("release_timestamp")
                                   or info.get("timestamp") or 0)
        else:
            self.duration = None
            self.video_start_ts = video_start_ts
        self.api_key, self.version, yid = _fetch_params(url)
        if not yid:
            raise RuntimeError("ytInitialData not found — possibly need cookies")
        self.continuation = continuation or _find_continuation(yid)
        if not self.continuation:
            raise RuntimeError("No continuation found")
        self._seen = set()
    def pages(self):
        """Yield (messages, continuation_after_this_page). Caller decides when
        to stop; `continuation` is the token to persist to resume here."""
        while self.continuation:
            if self.continuation in self._seen:
                return
            self._seen.add(self.continuation)
            data = _fetch_chat(self.api_key, self.version, self.continuation)
            actions = data.get("actions") or data.get("continuationContents", {}) \
                .get("liveChatContinuation", {}).get("actions")
            nxt = _extract_next_cont(data)
            yield _parse_messages(actions, self.video_start_ts), nxt
            self.continuation = nxt
            time.sleep(0.08)


# Execute main function only when script is run directly
if __name__ == "__main__":
    main()
