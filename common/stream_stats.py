"""Privacy-preserving, per-video aggregates computed while raw chat exists."""
import json
import math
import re
from collections import Counter, defaultdict

from common.chat_parser import categorize_message, parse_membership_rank
from common.feature_analysis import has_humor


SCHEMA_VERSION = 1
WORD_LIMIT = 200
WORD_MIN_COUNT = 5
HISTOGRAM_SECONDS = 60
HUMOR_SECONDS = 30
KNOWN_CATEGORIES = frozenset(("emoji", "jp", "kr", "ru", "number", "es_en_id"))
_WORD_RE = re.compile(
    r"[A-Za-z\u00c0-\u024f\u0370-\u03ff\u0400-\u04ff]+"
    r"(?:['\u2019-][A-Za-z\u00c0-\u024f\u0370-\u03ff\u0400-\u04ff]+)*"
)
_EMOTE_RE = re.compile(r":[^:\s]+:")
_STOPWORDS = frozenset("""
 a an the and or but if of in to for on at by from with as into about over under
 after before between through am is are was were be been being have has had
 having do does did doing done will would could should may might must shall can
 i me my mine myself we us our ours ourselves you your yours yourself yourselves
 he him his himself she her hers herself it its itself they them their theirs
 themselves this that these those what which who whom whose where when why how
 all any both each few more most some such no not nor only own same so than too
 very just also now here there then up down out off again im ive id ill youre
 youve youll youd hes shes theyre theyve theyll thats theres whats heres isnt
 arent wasnt werent havent hasnt dont doesnt didnt wont wouldnt couldnt shouldnt
 yeah yea yep yup nope okay oh ah uh um hmm huh wow lol lmao lmfao rofl haha
 hahaha hehe hi hey hello yes bye get got go going goes gone went come coming
 came make making made see saw seen know knew says said say like liked think
 thought want wanted look looking really still much many good great nice
""".split())


def _words(text):
    text = _EMOTE_RE.sub(" ", text or "")
    for match in _WORD_RE.finditer(text):
        word = match.group().lower().replace("\u2019", "'")
        if 3 <= len(word) <= 25 and word not in _STOPWORDS:
            if word.replace("'", "") not in _STOPWORDS:
                yield word


def _funny_moments(buckets, duration_seconds):
    if duration_seconds < 600 or not buckets:
        return []
    count = max(1, math.ceil(duration_seconds / HUMOR_SECONDS))
    raw = [buckets.get(i, 0) for i in range(count)]
    smooth = [raw[i] + (raw[i - 1] if i else 0)
              + (raw[i + 1] if i + 1 < count else 0)
              for i in range(count)]
    wanted = min(5, max(1, int(duration_seconds // 1800)))
    gap = 300 // HUMOR_SECONDS
    blocked = set()
    found = []
    for _ in range(wanted):
        candidates = ((value, -idx, idx) for idx, value in enumerate(smooth)
                      if idx not in blocked and value > 0)
        best = max(candidates, default=None)
        if best is None:
            break
        value, _neg, idx = best
        offset = idx * HUMOR_SECONDS
        found.append({"offset_seconds": offset,
                      "start_seconds": max(0, offset - 10),
                      "count": value})
        blocked.update(range(max(0, idx - gap), min(count, idx + gap + 1)))
    return sorted(found, key=lambda item: item["offset_seconds"])


class StreamStatsAccumulator:
    """Incrementally aggregates messages; no original text leaves ``add``."""

    def __init__(self, duration_seconds=0, stream_start_ts=None):
        self.duration = max(0, int(duration_seconds or 0))
        self.start = float(stream_start_ts) if stream_start_ts is not None else None
        bins = math.ceil(self.duration / HISTOGRAM_SECONDS) if self.duration else 0
        self.histogram = [0] * bins
        self.users = set()
        self.ranks = {}
        self.categories = Counter()
        self.words = Counter()
        self.humor = defaultdict(int)
        self.message_count = 0
        self.first_timestamp = None

    def add(self, message):
        author = message.get("author") or {}
        user_id = author.get("id")
        if not user_id:
            return
        self.users.add(user_id)
        timestamp = float(message.get("timestamp") or 0)
        if timestamp and (self.first_timestamp is None
                          or timestamp < self.first_timestamp):
            self.first_timestamp = timestamp
        badges = author.get("badges") or []
        message_type = message.get("message_type", "chat")
        if message_type in ("new_member", "gift_member"):
            rank = (-2 if message_type == "gift_member" and not badges
                    else parse_membership_rank(badges[0].lower() if badges else ""))
            self.ranks[user_id] = rank
            return
        text = message.get("message")
        if not isinstance(text, str) or not text.strip():
            return
        category = categorize_message(text)
        if not category:
            return
        self.message_count += 1
        self.categories[category] += 1
        self.ranks[user_id] = parse_membership_rank(
            badges[0].lower() if badges else "")
        self.words.update(_words(text))
        if self.start is not None and timestamp >= self.start:
            offset = timestamp - self.start
            minute = int(offset // HISTOGRAM_SECONDS)
            if minute >= len(self.histogram) and not self.duration:
                self.histogram.extend([0] * (minute + 1 - len(self.histogram)))
            if 0 <= minute < len(self.histogram):
                self.histogram[minute] += 1
            if has_humor(text):
                self.humor[int(offset // HUMOR_SECONDS)] += 1

    def add_legacy(self, message):
        """Consume one flattened legacy row without retaining its identity/text."""
        user_id = message.get("user_id")
        if not isinstance(user_id, str) or not user_id:
            return False
        self.users.add(user_id)
        try:
            timestamp = float(message.get("timestamp") or 0)
        except (TypeError, ValueError):
            timestamp = 0.0
        if abs(timestamp) >= 1_000_000_000_000:
            timestamp /= 1_000_000
        if timestamp and (self.first_timestamp is None
                          or timestamp < self.first_timestamp):
            self.first_timestamp = timestamp
        try:
            rank = int(message.get("membership_rank", -1))
        except (TypeError, ValueError):
            rank = -1
        message_type = message.get("message_type", "chat")
        if message_type in ("new_member", "gift_member"):
            self.ranks[user_id] = (-2 if message_type == "gift_member"
                                   and rank < 0 else rank)
            return True
        text = message.get("message")
        if not isinstance(text, str) or not text.strip():
            return True
        category = message.get("message_category")
        if category not in KNOWN_CATEGORIES:
            category = categorize_message(text)
        if not category:
            return True
        self.message_count += 1
        self.categories[category] += 1
        self.ranks[user_id] = rank
        self.words.update(_words(text))
        if self.start is not None and timestamp >= self.start:
            offset = timestamp - self.start
            minute = int(offset // HISTOGRAM_SECONDS)
            if 0 <= minute < len(self.histogram):
                self.histogram[minute] += 1
            if has_humor(text):
                self.humor[int(offset // HUMOR_SECONDS)] += 1
        return True

    def finish(self):
        rank_counts = Counter(self.ranks.values())
        members = sum(count for rank, count in rank_counts.items() if rank >= 0)
        member_percentage = (round(100 * members / len(self.users), 3)
                             if self.users else 0.0)
        words = sorted(
            ((word, count) for word, count in self.words.items()
             if count >= WORD_MIN_COUNT), key=lambda item: (-item[1], item[0]))
        return {
            "schema_version": SCHEMA_VERSION,
            "message_count": self.message_count,
            "unique_chatters": len(self.users),
            "member_chatters": members,
            "member_percentage": member_percentage,
            "category_counts": dict(sorted(self.categories.items())),
            "membership_rank_counts": {
                str(rank): count for rank, count in sorted(rank_counts.items())
            },
            "histogram_bin_seconds": HISTOGRAM_SECONDS,
            "histogram_counts": self.histogram,
            "funny_moments": _funny_moments(self.humor, self.duration),
            "word_counts": [[word, count] for word, count in words[:WORD_LIMIT]],
            "first_message_at": self.first_timestamp,
        }


def upsert_stream_stats(cur, video_id, aggregate, *, preserve_ready=False):
    """Write one complete aggregate. Only aggregate fields cross this boundary."""
    conflict = ("DO UPDATE SET "
                "schema_version=EXCLUDED.schema_version, status='ready', "
                "message_count=EXCLUDED.message_count, "
                "unique_chatters=EXCLUDED.unique_chatters, "
                "member_chatters=EXCLUDED.member_chatters, "
                "member_percentage=EXCLUDED.member_percentage, "
                "category_counts=EXCLUDED.category_counts, "
                "membership_rank_counts=EXCLUDED.membership_rank_counts, "
                "histogram_bin_seconds=EXCLUDED.histogram_bin_seconds, "
                "histogram_counts=EXCLUDED.histogram_counts, "
                "funny_moments=EXCLUDED.funny_moments, "
                "word_counts=EXCLUDED.word_counts, "
                "first_message_at=EXCLUDED.first_message_at, "
                "last_error=NULL, computed_at=NOW(), updated_at=NOW()")
    if preserve_ready:
        conflict += " WHERE video_stream_stats.status <> 'ready'"
    cur.execute(f"""INSERT INTO video_stream_stats (
                     video_id, schema_version, status, message_count,
                     unique_chatters, member_chatters, member_percentage,
                     category_counts, membership_rank_counts,
                     histogram_bin_seconds, histogram_counts, funny_moments,
                     word_counts, first_message_at, last_error,
                     computed_at, updated_at)
                   VALUES (%s,%s,'ready',%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s,
                           %s::jsonb,%s::jsonb,%s::jsonb,to_timestamp(%s),
                           NULL,NOW(),NOW())
                   ON CONFLICT (video_id) {conflict}""",
                (video_id, aggregate["schema_version"],
                 aggregate["message_count"], aggregate["unique_chatters"],
                 aggregate["member_chatters"],
                 aggregate["member_percentage"],
                 json.dumps(aggregate["category_counts"]),
                 json.dumps(aggregate["membership_rank_counts"]),
                 aggregate["histogram_bin_seconds"],
                 json.dumps(aggregate["histogram_counts"]),
                 json.dumps(aggregate["funny_moments"]),
                 json.dumps(aggregate["word_counts"]),
                 aggregate["first_message_at"]))
    return cur.rowcount > 0
