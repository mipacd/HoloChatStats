import re
import json
import time
import requests
from yt_dlp import YoutubeDL

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"

def _fetch_html(url):
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    return r.text

def _extract_params(html):
    key_m = re.search(r'INNERTUBE_API_KEY["\']\s*:\s*"([^"]+)"', html)
    ver_m = re.search(r'INNERTUBE_CONTEXT_CLIENT_VERSION["\']\s*:\s*"([^"]+)"', html)
    yid_m = re.search(r'ytInitialData["\']?\s*[:=]\s*(\{.*?\})[;\n]', html, flags=re.DOTALL)
    api_key = key_m.group(1) if key_m else None
    version = ver_m.group(1) if ver_m else "2.20201021.03.00"
    yid = json.loads(yid_m.group(1)) if yid_m else None
    return api_key, version, yid

def _find_continuation(ytInitialData):
    def walk(d):
        if isinstance(d, dict):
            if "continuation" in d:
                return d["continuation"]
            for v in d.values():
                res = walk(v)
                if res:
                    return res
        elif isinstance(d, list):
            for i in d:
                res = walk(i)
                if res:
                    return res
        return None
    return walk(ytInitialData)

def _fetch_chat(api_key, version, continuation):
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
    msgs = []
    for a in actions or []:
        if "replayChatItemAction" not in a:
            continue

        item = a["replayChatItemAction"].get("actions", [{}])[0]
        chat = item.get("addChatItemAction", {}).get("item", {})

        for t in ("liveChatTextMessageRenderer", "liveChatPaidMessageRenderer"):
            if t not in chat:
                continue

            r = chat[t]

            # --- Author Info ---
            author = {
                "id": r.get("authorExternalChannelId", None),
                "name": r.get("authorName", {}).get("simpleText", "").strip(),
                "badges": [],
            }

            # Extract badges (e.g. membership, mod, owner)
            for badge in r.get("authorBadges", []) or []:
                badge_label = badge.get("liveChatAuthorBadgeRenderer", {}).get("tooltip", "")
                if badge_label:
                    author["badges"].append(badge_label)

            if not author["name"]:
                continue

            # --- Message Text ---
            msg_runs = r.get("message", {}).get("runs", [])
            msg = "".join([x.get("text", "") for x in msg_runs]).strip()
            if not msg:
                continue

            # --- Timestamp (offset + video start time) ---
            offset_ms = int(float(r.get("videoOffsetTimeMsec", 0)))
            if offset_ms < 0:
                continue

            # Convert to UNIX timestamp using video_start_ts
            timestamp = video_start_ts + (offset_ms / 1000.0)

            msgs.append({
                "author": author,
                "message": msg,
                "timestamp": timestamp
            })
    return msgs


def _extract_next_cont(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "continuation":
                return v
            res = _extract_next_cont(v)
            if res:
                return res
    elif isinstance(obj, list):
        for i in obj:
            res = _extract_next_cont(i)
            if res:
                return res
    return None

def iter_youtube_chat(video_id):
    url = f"https://www.youtube.com/watch?v={video_id}"
    with YoutubeDL() as ydl:
        info = ydl.extract_info(url, download=False)
        duration = info.get("duration", 0)
        # start_time is in epoch seconds
        video_start_ts = info.get("release_timestamp") or info.get("timestamp") or 0

    html = _fetch_html(url)
    api_key, version, yid = _extract_params(html)
    if not yid:
        raise RuntimeError("ytInitialData not found — possibly need cookies")

    continuation = _find_continuation(yid)
    if not continuation:
        raise RuntimeError("No continuation found")

    seen = set()
    while continuation:
        if continuation in seen:
            break
        seen.add(continuation)

        data = _fetch_chat(api_key, version, continuation)
        actions = data.get("actions") or data.get("continuationContents", {}).get(
            "liveChatContinuation", {}
        ).get("actions")

        # 👇 Pass the video_start_ts here
        for msg in _parse_messages(actions, video_start_ts):
            yield msg

        continuation = _extract_next_cont(data)
        time.sleep(0.08)

