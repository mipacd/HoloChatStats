import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class ResumableMonthMergeTests(unittest.TestCase):
    def test_database_batch_is_bounded_and_retry_safe(self):
        sql = (ROOT / "migrations" / "011_resumable_month_merge.sql").read_text(
            encoding="utf-8")
        self.assertIn("CREATE OR REPLACE FUNCTION merge_month_batch", sql)
        self.assertIn("LIMIT batch_size", sql)
        self.assertIn("FOR UPDATE SKIP LOCKED", sql)
        self.assertIn("ON CONFLICT (user_id, channel_id, last_message_at, video_id)",
                      sql)
        self.assertNotIn("SET status      = 'merged'", sql)

    def test_worker_commits_progress_and_self_continues(self):
        merge = (ROOT / "handlers" / "merge.py").read_text(encoding="utf-8")
        self.assertIn('MERGE_BATCH_SIZE = 10000', merge)
        self.assertIn("SELECT merge_month_batch(%s::date, %s)", merge)
        self.assertIn("SET rows_merged=COALESCE(rows_merged, 0) + %s", merge)
        self.assertIn('"phase": "moving"', merge)
        self.assertIn("_request_merge_resume(month)", merge)
        self.assertIn("WHERE status='merging'", merge)

    def test_reaper_recovers_a_throttled_continuation(self):
        reaper = (ROOT / "handlers" / "reap.py").read_text(encoding="utf-8")
        self.assertIn('"month_merge": _resume_month_merge(dry)', reaper)
        self.assertIn("status='merging'", reaper)
        self.assertIn("INTERVAL '3 minutes'", reaper)
        self.assertIn('"source": "reaper"', reaper)
        self.assertIn("LIMIT 1 FOR UPDATE SKIP LOCKED", reaper)

    def test_publication_boundary_closes_only_after_final_checks(self):
        merge = (ROOT / "handlers" / "merge.py").read_text(encoding="utf-8")
        finalize = merge.split("def _finalize_month", 1)[1].split(
            "def _request_merge_resume", 1)[0]
        self.assertIn("FOR UPDATE", finalize)
        self.assertIn("user_data_current", finalize)
        self.assertLess(finalize.index("refresh_membership_data_for_month"),
                        finalize.index("SET status='merged'"))
        self.assertLess(finalize.index("SET status='merged'"),
                        finalize.index("publication_hold:"))
        self.assertIn("FINALIZE_STATEMENT_TIMEOUT_MS", finalize)

    def test_admin_reports_durable_row_progress(self):
        admin = (ROOT / "handlers" / "admin.py").read_text(encoding="utf-8")
        self.assertIn("publish_rows_moved", admin)
        self.assertIn("Rows moved", admin)
        self.assertIn('["merged", "merging"]', admin)


if __name__ == "__main__":
    unittest.main()
