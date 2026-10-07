import os
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock


os.environ.setdefault("RAW_BUCKET", "test-raw")

from handlers import refresh


class FakeCursor:
    def __init__(self, row, marker_cleared=True):
        self.row = row
        self.marker_cleared = marker_cleared
        self.rowcount = 0
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, sql, params=None):
        normalized = " ".join(sql.split())
        self.calls.append((normalized, params))
        if normalized.startswith("DELETE FROM service_config") and \
                "value='pending'" in normalized:
            self.rowcount = 1 if self.marker_cleared else 0
        else:
            self.rowcount = 1

    def fetchone(self):
        return self.row


class FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.autocommit = False

    def cursor(self):
        return self._cursor

    def close(self):
        pass


class LateRepublishTests(unittest.TestCase):
    def row(self, stage):
        started = datetime(2026, 10, 6, 12, 0, tzinfo=timezone.utc)
        return (started, str(stage), started, True)

    def test_each_invocation_commits_exactly_one_expensive_stage(self):
        cursor = FakeCursor(self.row(0))
        with mock.patch.object(refresh, "get_conn",
                               return_value=FakeConnection(cursor)), \
                mock.patch.object(refresh, "emit"):
            result = refresh._resume_republish(["2026-07-01"])

        self.assertEqual(result["stage"], 1)
        self.assertEqual(result["phase"], "monthly activity")
        self.assertFalse(result["complete"])
        self.assertTrue(any(
            sql.startswith("DELETE FROM late_month_overrides")
            for sql, _params in cursor.calls))
        self.assertTrue(any(
            sql.startswith("CALL refresh_late_month_overlay") and
            params == (datetime(2026, 7, 1).date(), 0)
            for sql, params in cursor.calls))
        self.assertTrue(any(
            sql.startswith("UPDATE service_config SET value=%s") and
            params[0] == "1" for sql, params in cursor.calls))

    def test_final_stage_clears_marker_only_after_cache_invalidation(self):
        cursor = FakeCursor(self.row(5), marker_cleared=True)
        with mock.patch.object(refresh, "get_conn",
                               return_value=FakeConnection(cursor)), \
                mock.patch.object(refresh, "invalidate_finalized_month_caches",
                                  return_value=17) as invalidate, \
                mock.patch.object(refresh, "emit"):
            result = refresh._resume_republish(["2026-07-01"])

        invalidate.assert_called_once()
        self.assertTrue(result["complete"])
        self.assertEqual(result["cache_keys_removed"], 17)
        self.assertTrue(any(
            sql.startswith("INSERT INTO late_month_overrides")
            for sql, _params in cursor.calls))
        self.assertTrue(any("late_data_published:" in str(params)
                            for _sql, params in cursor.calls))
        self.assertTrue(any("late_republish_stage:" in str(params)
                            and sql.startswith("DELETE FROM service_config")
                            for sql, params in cursor.calls))

    def test_late_log_arriving_during_refresh_remains_pending(self):
        marker = datetime(2026, 10, 6, 12, 1, tzinfo=timezone.utc)
        started = datetime(2026, 10, 6, 12, 0, tzinfo=timezone.utc)
        cursor = FakeCursor((marker, "5", started, True),
                            marker_cleared=False)
        with mock.patch.object(refresh, "get_conn",
                               return_value=FakeConnection(cursor)), \
                mock.patch.object(refresh, "invalidate_finalized_month_caches",
                                  return_value=3), \
                mock.patch.object(refresh, "emit"):
            result = refresh._resume_republish(["2026-07-01"])

        self.assertFalse(result["complete"])
        self.assertTrue(result["newer_late_data_pending"])
        self.assertFalse(any("late_data_published:" in str(params)
                             for _sql, params in cursor.calls))

    def test_reaper_and_admin_expose_durable_progress(self):
        root = os.path.dirname(os.path.dirname(__file__))
        with open(os.path.join(root, "handlers", "reap.py"), encoding="utf-8") as f:
            reaper = f.read()
        with open(os.path.join(root, "handlers", "admin.py"), encoding="utf-8") as f:
            admin = f.read()
        self.assertIn('"late_republish": _resume_late_republish(dry)', reaper)
        self.assertIn("late_republish_stage:%", reaper)
        self.assertIn('"republish_stage": r[12]', admin)
        self.assertIn("Re-publishing ${Math.min", admin)

    def test_month_overlays_replace_only_activated_months(self):
        root = os.path.dirname(os.path.dirname(__file__))
        migration = os.path.join(
            root, "migrations", "014_late_month_overlays.sql")
        with open(migration, encoding="utf-8") as f:
            sql = f.read()
        self.assertIn("PROCEDURE refresh_late_month_overlay", sql)
        self.assertIn("CREATE OR REPLACE VIEW mv_user_activity_live", sql)
        self.assertIn("JOIN late_month_overrides", sql)
        self.assertIn("last_message_at >= target_month", sql)
        self.assertNotIn("REFRESH MATERIALIZED VIEW", sql)

        models = (Path(root) / "web" / "models.py").read_text(encoding="utf-8")
        self.assertIn('__tablename__ = "mv_user_activity_live"', models)
        self.assertIn('__tablename__ = "chat_language_stats_live"', models)


if __name__ == "__main__":
    unittest.main()
