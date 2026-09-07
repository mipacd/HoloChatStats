import unittest

from web.cache_warmer import build_tasks


class CacheWarmerTaskTests(unittest.TestCase):
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
