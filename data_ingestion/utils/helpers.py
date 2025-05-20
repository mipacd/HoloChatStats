import os
import sys
import yt_dlp
from utils.logging_utils import get_logger
from config.settings import get_config

# Create necessary directories for cache
def setup_cache_dir():
    logger = get_logger()
    """Create cache directories if they do not exist.

    The cache directories are created inside the configured cache directory.
    The cache directory is divided into two subdirectories: "videos" and "chat_logs".
    """
    cache_dir = get_config("Settings", "CacheDir")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    if not os.path.exists(cache_dir + "/videos"):
        os.makedirs(os.path.join(cache_dir, "videos"))
    if not os.path.exists(cache_dir + "/chat_logs"):
        os.makedirs(os.path.join(cache_dir, "chat_logs"))
    logger.info("Cache directories configured.")

def deduplicate_batch(batch, key_indices):
    """Deduplicates a batch of rows based on the unique key indices."""
    seen = set()
    deduplicated_batch = []

    for row in batch:
        # Extract the unique key (e.g., a tuple of the key indices)
        key = tuple(row[index] for index in key_indices)
        if key not in seen:
            deduplicated_batch.append(row)
            seen.add(key)

    return deduplicated_batch

def is_video_past(video_id):
    url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "logger": None,
        "outtmpl": os.devnull,
        "verbose": False,
    }

    sys.stderr = open(os.devnull, "w")

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            video_data = ydl.extract_info(url, download=False)
            if video_data.get("is_live"):
                return False # stream is live
            return True # stream has ended
    except yt_dlp.utils.DownloadError as e:
        if "This live event will begin in" in str(e):
            return False # stream is upcoming
    finally:
        sys.stderr.close()
        sys.stderr = sys.__stderr__

    return False # other status

# Returns list of streams to ignore (such as membership streams made public)
def get_ignore_list():
    ignore_list = [
        "uC6emG9xSJI", # Fauna Start
        "bqGsIflAilo",
        "9LKLH0PvZNg",
        "N-FazGH3pMc",
        "-KqJgikUz-0",
        "J-5k266uDQ0",
        "b_BOq-zYXn8",
        "b4DYJ7-1IQA",
        "oOEt0yiFOj8",
        "00uSzJUIk-M",
        "Sh7aHxDuhm0",
        "SKDcj-T5aGA",
        "43uVLZsPkBs",
        "ZZewIRQ7lQI",
        "8DR9J1D3GH0",
        "0aATokqOzZE",
        "dw6qeolYvJs",
        "c7m9H7FBa0s",
        "sm7GjAaZdzY",
        "yKrnnCuGx38",
        "kg-pshICjP0",
        "COBL8itTgUk",
        "FEW47e5ONHk",
        "C1HQmunNVUY",
        "5W94amtdoDY",
        "MepLJBsgsQ4",
        "BItZTYknlA8",
        "h3yP51Lc8Ds",
        "4I4V5_xJ-qI",
        "FuhM-356UOY",
        "t6LYeCm-Ghs",
        "Ts_SHcaVDjg",
        "NqyBkCZFPaw",
        "oT5WJfMM9cc",
        "dFlii9331nw",
        "-x1XoANfQAA",
        "VFPXCxI51Hg",
        "oVmg5ctd4i0",
        "dNkrcmA_HzU",
        "fBYJWytRLLo",
        "4W186W1Er8U",
        "7m8R6WxgjUE",
        "WbbwerxYD6o",
        "Nb3hP1PpfNE",
        "h9sQ4LCH-Ck",
        "VvjiSEIg0Ig",
        "tOjds7TSje0",
        "S7ZsBiLth0k",
        "g_tJQxpTAsw",
        "3bzdHNiqx74",
        "PPbahgj17mA",
        "jAv6NXW0LkA",
        "FnYyjBlfMDc",
        "hXk6FE1Ce3o",
        "5AwCqnee9Ek",
        "Aj5pmKHIIXc",
        "AI9yq42OBZU",
        "0zr549A0Sa8",
        "bqHAOYTaD28",
        "TyLN_CNwGG0",
        "oMA5yAGepEk",
        "UqOvKVc2PoU",
        "MPYXTh36IAM",
        "Kxh769Q6HzI",
        "2t9ELE4QQ3w",
        "gsJK4M954Bg",
        "0YB_jET_U9A",
        "ODll2JO1qKs",
        "4RhvYh5Etrk",
        "3KCB1aETLpY",
        "kshWK9WEMYc",
        "J3ERu05HUBc",
        "oaZXf0yDBQc" # Fauna End
    ]
    return ignore_list
