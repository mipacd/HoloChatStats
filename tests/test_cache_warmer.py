import fnmatch
import unittest

from web.cache_warmer import _persist_analytics_caches, build_tasks


class PersistRedis:
    def __init__(self, keys, expiring):
        self.keys = set(keys)
        self.expiring = set(expiring)

    def scan_iter(self, match, count):
        del count
        yield from sorted(k for k in self.keys if fnmatch.fnmatch(k, match))

    def persist(self, key):
        changed = key in self.expiring
        self.expiring.discard(key)
        return int(changed)


class CacheWarmerTaskTests(unittest.TestCase):
    def test_only_analytics_cache_ttls_are_removed(self):
        analytics = {"common_users_Alpha_2026-07_Beta_2026-07",
                     "exclusive_chat_users_Alpha"}
        operational = {"rate:visitor-hash", "v2:page_views:2026-09-08"}
        fake = PersistRedis(analytics | operational, analytics | operational)
        changed = _persist_analytics_caches(fake)
        self.assertEqual(changed, len(analytics))
        self.assertEqual(fake.expiring, operational)

    def test_tasks_are_bounded_and_prioritize_exclusive_chat(self):
        tasks = build_tasks(["Alpha", "Beta"], ["Group A"], "2026-07-01")
        exclusive = [
            (path, query) for path, query in tasks
            if path == "/api/get_exclusive_chat_users"
        ]
        self.assertEqual(exclusive, [
            ("/api/get_exclusive_chat_users", {"channel": "Alpha"}),
            ("/api/get_exclusive_chat_users", {"channel": "Beta"}),
        ])
        self.assertEqual(tasks[:2], exclusive)

        paths = {path for path, _query in tasks}
        self.assertNotIn("/api/get_user_info", paths)
        self.assertNotIn("/api/get_common_users", paths)
        self.assertNotIn("/api/get_common_users_matrix", paths)
        self.assertNotIn("/api/community_graph/find_user", paths)
        self.assertNotIn("/api/community_graph", paths)
        self.assertNotIn("/api/channel_clustering", paths)
        self.assertNotIn("/api/content_clustering", paths)
        self.assertTrue(all("identifier" not in query for _path, query in tasks))

    def test_default_and_named_group_cache_keys_are_covered(self):
        tasks = build_tasks(["Alpha"], ["Group A"], "2026-07-19")
        makeup_queries = [
            query for path, query in tasks
            if path == "/api/get_group_chat_makeup"
        ]
        self.assertIn({"month": "2026-07"}, makeup_queries)
        self.assertIn({"group": "Group A", "month": "2026-07"},
                      makeup_queries)


if __name__ == "__main__":
    unittest.main()
