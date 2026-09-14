import unittest

from common.stream_stats import (
    StreamStatsAccumulator, materially_invalid_timing, timing_outlier_limit,
)


def msg(uid, timestamp, text, badges=None, kind="chat", offset=None):
    value = {"author": {"id": uid, "badges": badges or []},
             "timestamp": timestamp, "message": text, "message_type": kind}
    if offset is not None:
        value["offset_seconds"] = offset
    return value


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

    def test_direct_replay_offset_wins_over_shifted_metadata(self):
        agg = StreamStatsAccumulator(3600, 10_000, legacy_rebase=True)
        agg.add(msg("viewer", 8_800, "closing message", offset=3590))
        result = agg.finish()
        self.assertEqual(result["timing_source"], "direct_replay_offset")
        self.assertEqual(result["histogram_counts"][-1], 1)
        self.assertEqual(result["last_offset_seconds"], 3590)
        self.assertEqual(result["quiet_tail_seconds"], 10)

    def test_old_lambda_rows_rebase_when_metadata_anchor_drifted(self):
        agg = StreamStatsAccumulator(600, 10_000, legacy_rebase=True)
        agg.add(msg("first", 9_400, "opening message"))
        agg.add(msg("last", 9_990, "closing message"))
        result = agg.finish()
        self.assertEqual(result["timing_source"], "first_message_fallback")
        self.assertEqual(result["histogram_counts"][0], 1)
        self.assertEqual(result["histogram_counts"][-1], 1)
        self.assertEqual(result["out_of_range_messages"], 0)

    def test_small_metadata_rounding_does_not_trigger_rebase(self):
        agg = StreamStatsAccumulator(600, 10_000, legacy_rebase=True)
        agg.add(msg("early", 9_993, "hello message"))
        result = agg.finish()
        self.assertEqual(result["timing_source"], "metadata_derived")
        self.assertEqual(result["histogram_counts"][0], 1)
        self.assertEqual(result["out_of_range_messages"], 0)

    def test_checkpoint_recovers_old_lambda_anchor_from_last_message(self):
        agg = StreamStatsAccumulator(
            3600, 10_000, legacy_rebase=True, checkpoint_last_offset=3590)
        agg.add(msg("first", 8_800, "opening message"))
        agg.add(msg("last", 12_390, "closing message"))
        result = agg.finish()
        self.assertEqual(result["timing_source"], "recovered_checkpoint")
        self.assertEqual(result["histogram_counts"][0], 1)
        self.assertEqual(result["histogram_counts"][-1], 1)
        self.assertEqual(result["quiet_tail_seconds"], 10)

    def test_reported_shift_ranges_rebase_but_healthy_rounding_does_not(self):
        affected = ((3775, 888), (5041, 549), (6065, 278), (5742, 5931))
        for duration, drift in affected:
            with self.subTest(duration=duration, drift=drift):
                metadata_start = 20_000
                archived_start = metadata_start - drift
                agg = StreamStatsAccumulator(
                    duration, metadata_start, legacy_rebase=True)
                agg.add(msg("first", archived_start, "opening message"))
                agg.add(msg("last", archived_start + duration - 5,
                            "goodbye message"))
                result = agg.finish()
                self.assertEqual(result["timing_source"],
                                 "first_message_fallback")
                last_nonzero = max(
                    index for index, count in enumerate(
                        result["histogram_counts"]) if count)
                self.assertGreaterEqual((last_nonzero + 1) * 60,
                                        duration - 5)
        for drift in (1, 7):
            with self.subTest(healthy_drift=drift):
                agg = StreamStatsAccumulator(
                    600, 20_000, legacy_rebase=True)
                agg.add(msg("viewer", 20_000 - drift, "hello message"))
                result = agg.finish()
                self.assertEqual(result["timing_source"], "metadata_derived")
                self.assertEqual(result["histogram_counts"][0], 1)

    def test_legacy_timing_requires_eighty_percent_usable_coverage(self):
        self.assertEqual(timing_outlier_limit(100), 20)
        self.assertEqual(timing_outlier_limit(10_001), 2001)
        mostly_valid = {
            "message_count": 1000,
            "histogram_counts": [800],
            "out_of_range_messages": 200,
        }
        materially_shifted = {
            "message_count": 1000,
            "histogram_counts": [799],
            "out_of_range_messages": 201,
        }
        self.assertFalse(materially_invalid_timing(mostly_valid, 60))
        self.assertTrue(materially_invalid_timing(materially_shifted, 60))
        self.assertTrue(materially_invalid_timing({
            "message_count": 50, "histogram_counts": [0],
            "out_of_range_messages": 0}, 60))
        self.assertFalse(materially_invalid_timing({
            "message_count": 50, "histogram_counts": [],
            "out_of_range_messages": 0}, 0))


if __name__ == "__main__":
    unittest.main()
