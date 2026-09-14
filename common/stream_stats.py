"""Privacy-preserving, per-video aggregates computed while raw chat exists."""
import json
import math
import re
from collections import Counter, defaultdict

from common.chat_parser import categorize_message, parse_membership_rank
from common.feature_analysis import has_humor


SCHEMA_VERSION = 2
WORD_LIMIT = 200
WORD_MIN_COUNT = 5
HISTOGRAM_SECONDS = 60
HUMOR_SECONDS = 30
TIMING_OUTLIER_MINIMUM = 10
# Legacy archives can include pre-roll or post-roll that YouTube later trimmed
# from the public VOD. Keep a timeline when a substantial majority still maps
# to playable offsets; gross mismatches remain unsafe for timestamp links.
TIMING_OUTLIER_FRACTION = 0.20
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


def timing_outlier_limit(message_count):
    """Maximum discarded timing samples allowed for a usable old archive."""
    return max(TIMING_OUTLIER_MINIMUM,
               math.ceil(max(0, int(message_count or 0))
                         * TIMING_OUTLIER_FRACTION))


def materially_invalid_timing(aggregate, duration_seconds):
    """Whether timeline loss is large enough to preserve the old aggregate."""
    messages = int(aggregate.get("message_count") or 0)
    histogram = aggregate.get("histogram_counts") or []
    outliers = int(aggregate.get("out_of_range_messages") or 0)
    if duration_seconds and messages and not sum(histogram):
        return True
    return outliers > timing_outlier_limit(messages)


class StreamStatsAccumulator:
    """Incrementally aggregates messages; no original text leaves ``add``."""

    def __init__(self, duration_seconds=0, stream_start_ts=None, *,
                 legacy_rebase=False, timing_source=None,
                 checkpoint_last_offset=None):
        self.duration = max(0, int(duration_seconds or 0))
        self.start = float(stream_start_ts) if stream_start_ts is not None else None
        self.legacy_rebase = bool(legacy_rebase)
        self._legacy_anchor_checked = False
        self.timing_source = timing_source or (
            "metadata_derived" if self.start is not None else "unavailable")
        bins = math.ceil(self.duration / HISTOGRAM_SECONDS) if self.duration else 0
        self.histogram = [0] * bins
        self.users = set()
        self.ranks = {}
        self.categories = Counter()
        self.words = Counter()
        self.humor = defaultdict(int)
        self.message_count = 0
        self.first_timestamp = None
        self.first_offset = None
        self.last_offset = None
        self.out_of_range = 0
        try:
            checkpoint = float(checkpoint_last_offset)
            self.checkpoint_last_offset = checkpoint if checkpoint > 0 else None
        except (TypeError, ValueError):
            self.checkpoint_last_offset = None
        # Old Lambda rows lack replay offsets. Aggregate their timing by second
        # until the final retained timestamp reveals the original anchor.
        self._pending_timing = Counter()
        self._pending_humor = Counter()
        self._checkpoint_last_timestamp = None

    def _offset(self, timestamp, explicit=None):
        if explicit is not None:
            try:
                value = float(explicit)
            except (TypeError, ValueError):
                value = -1
            if math.isfinite(value) and value >= 0:
                self.timing_source = "direct_replay_offset"
                return value
        if self.start is None or not timestamp:
            return None
        return timestamp - self.start

    def _prepare_timestamp_anchor(self, timestamp, explicit=None):
        """Select a stable fallback anchor for raw rows without replay offsets."""
        if explicit is not None or not self.legacy_rebase \
                or self._legacy_anchor_checked or not timestamp:
            return
        self._legacy_anchor_checked = True
        if self.start is None or timestamp < self.start - 60:
            self.start = timestamp
            self.timing_source = "first_message_fallback"

    def _record_timing(self, offset, text=None, *, count=1, humor_count=None):
        if offset is None or not math.isfinite(offset):
            self.out_of_range += count
            return
        # Minor metadata rounding is normal. Keep a message within one minute of
        # either boundary instead of turning an otherwise healthy stream into a
        # repair candidate.
        if -60 <= offset < 0:
            offset = 0.0
        elif self.duration and self.duration <= offset <= self.duration + 60:
            offset = max(0.0, self.duration - 0.001)
        minute = int(offset // HISTOGRAM_SECONDS)
        if minute >= len(self.histogram) and not self.duration:
            self.histogram.extend([0] * (minute + 1 - len(self.histogram)))
        if offset < 0 or (self.duration and offset >= self.duration):
            self.out_of_range += count
            return
        if 0 <= minute < len(self.histogram):
            self.histogram[minute] += count
        else:
            self.out_of_range += count
            return
        if self.first_offset is None or offset < self.first_offset:
            self.first_offset = offset
        if self.last_offset is None or offset > self.last_offset:
            self.last_offset = offset
        reactions = (int(humor_count) if humor_count is not None
                     else (1 if has_humor(text or "") else 0))
        if reactions:
            self.humor[int(offset // HUMOR_SECONDS)] += reactions

    def _defer_checkpoint_timing(self, timestamp, text):
        if not timestamp or self.checkpoint_last_offset is None:
            return False
        second = int(timestamp)
        self._pending_timing[second] += 1
        if has_humor(text):
            self._pending_humor[second] += 1
        return True

    def _finalize_checkpoint_timing(self):
        if not self._pending_timing:
            return
        final_timestamp = self._checkpoint_last_timestamp
        if final_timestamp is None:
            final_timestamp = max(self._pending_timing)
        self.start = final_timestamp - self.checkpoint_last_offset
        if self.timing_source != "direct_replay_offset":
            self.timing_source = "recovered_checkpoint"
        for timestamp, count in self._pending_timing.items():
            self._record_timing(
                timestamp - self.start, count=count,
                humor_count=self._pending_humor.get(timestamp, 0))
        self._pending_timing.clear()
        self._pending_humor.clear()

    def add(self, message):
        try:
            timestamp = float(message.get("timestamp") or 0)
        except (AttributeError, TypeError, ValueError):
            timestamp = 0.0
        if timestamp and (self._checkpoint_last_timestamp is None
                          or timestamp > self._checkpoint_last_timestamp):
            self._checkpoint_last_timestamp = timestamp
        author = message.get("author") or {}
        user_id = author.get("id")
        if not user_id:
            return
        self.users.add(user_id)
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
        explicit_offset = message.get("offset_seconds")
        self._prepare_timestamp_anchor(timestamp, explicit_offset)
        if explicit_offset is not None or not self._defer_checkpoint_timing(
                timestamp, text):
            self._record_timing(self._offset(timestamp, explicit_offset), text)

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
        self._prepare_timestamp_anchor(timestamp)
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
        self._record_timing(self._offset(timestamp), text)
        return True

    def finish(self):
        self._finalize_checkpoint_timing()
        rank_counts = Counter(self.ranks.values())
        members = sum(count for rank, count in rank_counts.items() if rank >= 0)
        member_percentage = (round(100 * members / len(self.users), 3)
                             if self.users else 0.0)
        words = sorted(
            ((word, count) for word, count in self.words.items()
             if count >= WORD_MIN_COUNT), key=lambda item: (-item[1], item[0]))
        last_offset = self.last_offset
        quiet_tail = (max(0.0, self.duration - last_offset)
                      if self.duration and last_offset is not None else None)
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
            "timing_source": self.timing_source,
            "first_offset_seconds": self.first_offset,
            "last_offset_seconds": last_offset,
            "out_of_range_messages": self.out_of_range,
            "quiet_tail_seconds": quiet_tail,
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
                "timing_source=EXCLUDED.timing_source, "
                "first_offset_seconds=EXCLUDED.first_offset_seconds, "
                "last_offset_seconds=EXCLUDED.last_offset_seconds, "
                "out_of_range_messages=EXCLUDED.out_of_range_messages, "
                "quiet_tail_seconds=EXCLUDED.quiet_tail_seconds, "
                "last_error=NULL, computed_at=NOW(), updated_at=NOW()")
    if preserve_ready:
        conflict += " WHERE video_stream_stats.status <> 'ready'"
    cur.execute(f"""INSERT INTO video_stream_stats (
                     video_id, schema_version, status, message_count,
                     unique_chatters, member_chatters, member_percentage,
                     category_counts, membership_rank_counts,
                     histogram_bin_seconds, histogram_counts, funny_moments,
                     word_counts, first_message_at, timing_source,
                     first_offset_seconds, last_offset_seconds,
                     out_of_range_messages, quiet_tail_seconds, last_error,
                     computed_at, updated_at)
                   VALUES (%s,%s,'ready',%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s,
                           %s::jsonb,%s::jsonb,%s::jsonb,to_timestamp(%s),%s,
                           %s,%s,%s,%s,
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
                 aggregate["first_message_at"], aggregate["timing_source"],
                 aggregate["first_offset_seconds"],
                 aggregate["last_offset_seconds"],
                 aggregate["out_of_range_messages"],
                 aggregate["quiet_tail_seconds"]))
    return cur.rowcount > 0
