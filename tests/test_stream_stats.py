import unittest

from common.stream_stats import StreamStatsAccumulator


def msg(uid, timestamp, text, badges=None, kind="chat"):
    return {"author": {"id": uid, "badges": badges or []},
            "timestamp": timestamp, "message": text, "message_type": kind}


class StreamStatsTests(unittest.TestCase):
    def test_aggregates_without_retaining_messages_or_users(self):
        agg = StreamStatsAccumulator(720, 1000)
        for i in range(5):
            agg.add(msg(f"u{i}", 1001 + i, "Rareword commonword lol"))
        result = agg.finish()
        self.assertEqual(result["message_count"], 5)
        self.assertEqual(result["unique_chatters"], 5)
        self.assertEqual(result["member_percentage"], 0)
        self.assertEqual(result["histogram_counts"][0], 5)
        self.assertEqual(result["word_counts"], [["commonword", 5], ["rareword", 5]])
        self.assertNotIn("users", result)
        self.assertNotIn("messages", result)

    def test_word_threshold_members_and_empty_chat(self):
        agg = StreamStatsAccumulator(120, 1000)
        for i in range(4):
            agg.add(msg(f"u{i}", 1000 + i, "almost"))
        agg.add(msg("member", 1005, "memberword " * 5, ["Member (6 months)"]))
        result = agg.finish()
        self.assertEqual(result["member_chatters"], 1)
        self.assertEqual(result["member_percentage"], 20)
        self.assertEqual(result["membership_rank_counts"]["6"], 1)
        self.assertEqual(result["word_counts"], [["memberword", 5]])
        self.assertEqual(StreamStatsAccumulator().finish()["message_count"], 0)

    def test_funny_moments_are_spaced_and_bounded(self):
        agg = StreamStatsAccumulator(7200, 1000)
        for offset in (30, 31, 60, 700, 701, 1400, 1401):
            agg.add(msg(str(offset), 1000 + offset, "hahaha"))
        moments = agg.finish()["funny_moments"]
        self.assertLessEqual(len(moments), 4)
        self.assertTrue(all(b["offset_seconds"] - a["offset_seconds"] >= 300
                            for a, b in zip(moments, moments[1:])))
        self.assertTrue(all(m["start_seconds"] == max(0, m["offset_seconds"] - 10)
                            for m in moments))

    def test_long_stream_histogram_and_category_counts(self):
        agg = StreamStatsAccumulator(48 * 3600, 1000)
        agg.add(msg("jp", 1000, "Japanese fallback"))
        agg.add(msg("late", 1000 + 48 * 3600 - 1, "final message"))
        result = agg.finish()
        self.assertEqual(len(result["histogram_counts"]), 48 * 60)
        self.assertEqual(result["histogram_counts"][0], 1)
        self.assertEqual(result["histogram_counts"][-1], 1)
        self.assertEqual(sum(result["category_counts"].values()), 2)


if __name__ == "__main__":
    unittest.main()
