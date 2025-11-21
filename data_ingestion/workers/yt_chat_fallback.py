import re
import json
import time
import sys
import requests
from yt_dlp import YoutubeDL

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"

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
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    return r.text

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
    yid_m = re.search(r'ytInitialData["\']?\s*[:=]\s*(\{.*?\})[;\n]', html, flags=re.DOTALL)
    api_key = key_m.group(1) if key_m else None
    version = ver_m.group(1) if ver_m else "2.20201021.03.00"
    yid = json.loads(yid_m.group(1)) if yid_m else None
    return api_key, version, yid

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
    headers = {"User-Agent": USER_AGENT, "Content-Type": "application/json"}
    r = requests.post(url, headers=headers, json=data, timeout=60)
    r.raise_for_status()
    return r.json()

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

        item = a["replayChatItemAction"].get("actions", [{}])[0]
        chat = item.get("addChatItemAction", {}).get("item", {})

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
                msg = "".join([x.get("text", "") for x in msg_runs]).strip()
                msg_data["message"] = msg
                # Determine if this is a paid message
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
                gifter = None
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

            offset_ms = int(float(r.get("videoOffsetTimeMsec", 0)))
            # Skip messages with negative offsets as they are invalid
            if offset_ms < 0:
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
    with YoutubeDL() as ydl:
        info = ydl.extract_info(url, download=False)
        duration = info.get("duration", 0)
        video_start_ts = info.get("release_timestamp") or info.get("timestamp") or 0

    html = _fetch_html(url)
    api_key, version, yid = _extract_params(html)
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


# Execute main function only when script is run directly
if __name__ == "__main__":
    main()