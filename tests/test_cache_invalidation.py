import fnmatch
import unittest
from unittest import mock

from common import cache_invalidation


class FakeRedis:
    def __init__(self, keys):
        self.keys = set(keys)
        self.deleted = []
        self.values = {}

    def scan_iter(self, match, count):
        del count
        yield from sorted(k for k in self.keys if fnmatch.fnmatch(k, match))

    def delete(self, *keys):
        removed = 0
        for key in keys:
            if key in self.keys:
                self.keys.remove(key)
                self.deleted.append(key)
                removed += 1
        return removed

    def set(self, key, value):
        self.values[key] = str(value)
        return True

class CacheInvalidationTests(unittest.TestCase):
    def test_only_aggregate_application_keys_are_removed(self):
        aggregate = {
            "num_messages",
            "date_ranges",
            "monthly_streaming_hours_Test_forecast_True",
            "channel_recommendations:user-1:6m",
            "stream_calendar_Test_UTC_latest_0",
            "published_coverage_v2:date_ranges",
            "published_coverage_v2:number_of_chat_logs",
            "published_coverage_v2:num_messages",
        }
        protected = {
            "group_chat_makeup_all_2026-07",
            "recommendation_monthly_data:2026-07",
            "rate:visitor-hash",
            "v2:page_views:2026-09-07",
            "cache_hits:2026-09-07",
        }
        fake = FakeRedis(aggregate | protected)
        with mock.patch.dict("os.environ", {"REDIS_HOST": "redis"},
                             clear=False), mock.patch.object(
                                 cache_invalidation.redis, "Redis",
                                 return_value=fake):
            removed = cache_invalidation.invalidate_finalized_month_caches(
                batch_size=2)
        self.assertEqual(removed, len(aggregate))
        self.assertEqual(set(fake.deleted), aggregate)
        self.assertEqual(fake.keys, protected)

    def test_finalized_month_requests_a_warm_pass(self):
        remove = {
            "group_chat_makeup_all_2026-07",
            "common_users_Alpha_2026-07_Beta_2026-06",
            "common_members_Alpha_2026-06_Beta_2026-07",
            "recommendation_monthly_data:2026-07",
        }
        preserve = {
            "group_chat_makeup_all_2026-06",
            "recommendation_monthly_data:2026-06",
            "rate:visitor-hash",
        }
        fake = FakeRedis(remove | preserve)
        with mock.patch.dict("os.environ", {"REDIS_HOST": "redis"},
                             clear=False), mock.patch.object(
                                 cache_invalidation.redis, "Redis",
                                 return_value=fake):
            cache_invalidation.invalidate_finalized_month_caches(
                finalized_month="2026-07-01")
        self.assertEqual(set(fake.deleted), remove)
        self.assertEqual(fake.keys, preserve)
        self.assertEqual(fake.values["cache_warm:requested_month"],
                         "2026-07-01")

if __name__ == "__main__":
    unittest.main()
