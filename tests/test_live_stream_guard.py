import pathlib
import unittest
from datetime import datetime, timedelta, timezone

from handlers.scan import _stream_disposition


ROOT = pathlib.Path(__file__).resolve().parents[1]
NOW = datetime(2026, 9, 18, 12, 0, tzinfo=timezone.utc)
CFG = {
    "future_stream_max_days": "30",
    "upcoming_start_grace_hours": "12",
    "max_live_stream_hours": "48",
}


def micros(dt):
    return int(dt.timestamp() * 1_000_000)


class LiveStreamGuardTests(unittest.TestCase):
    def test_past_stream_is_processed(self):
        self.assertEqual(_stream_disposition({"status": "past"}, CFG, NOW),
                         ("process", None))

    def test_missing_and_implausible_upcoming_starts_are_skipped(self):
        action, _ = _stream_disposition({"status": "upcoming"}, CFG, NOW)
        self.assertEqual(action, "skip")
        action, _ = _stream_disposition({
            "status": "upcoming",
            "start_time": micros(NOW + timedelta(days=31)),
        }, CFG, NOW)
        self.assertEqual(action, "skip")
        action, _ = _stream_disposition({
            "status": "upcoming",
            "start_time": micros(NOW - timedelta(hours=13)),
        }, CFG, NOW)
        self.assertEqual(action, "skip")

    def test_near_upcoming_and_recent_live_streams_cool_down(self):
        for status, start in (
            ("upcoming", NOW + timedelta(hours=2)),
            ("live", NOW - timedelta(hours=2)),
        ):
            action, _ = _stream_disposition({
                "status": status, "start_time": micros(start)}, CFG, NOW)
            self.assertEqual(action, "cooldown")

    def test_missing_or_persistently_live_stream_is_skipped(self):
        action, _ = _stream_disposition({"status": "live"}, CFG, NOW)
        self.assertEqual(action, "skip")
        action, reason = _stream_disposition({
            "status": "live",
            "start_time": micros(NOW - timedelta(hours=49)),
        }, CFG, NOW)
        self.assertEqual(action, "skip")
        self.assertIn("48 hours", reason)

    def test_dispatcher_honors_durable_cooldown(self):
        dispatch = (ROOT / "common" / "dispatch.py").read_text()
        self.assertIn("j.next_attempt_at <= NOW()", dispatch)
        migration = (ROOT / "migrations" /
                     "012_live_stream_cooldown.sql").read_text()
        self.assertIn("live_retry_cooldown_minutes", migration)
        self.assertIn("live_cooldown_max_attempts", migration)
        self.assertIn("persistently unavailable live/upcoming chat", (
            ROOT / "handlers" / "download.py").read_text())
        scan = (ROOT / "handlers" / "scan.py").read_text()
        self.assertIn("status IN ('pending', 'failed')", scan)
        self.assertIn("override_failed=True", scan)

    def test_advisory_lock_does_not_close_shared_database_session(self):
        download = (ROOT / "handlers" / "download.py").read_text()
        lock_block = download.split("lock_conn = get_conn()", 1)[1].split(
            "return {\"ok\": True}", 1)[0]
        self.assertNotIn("lock_conn.close()\n            sqs.send_message",
                         lock_block)
        defer = download.split("def _defer_future_month", 1)[1].split(
            "def _process", 1)[0]
        self.assertNotIn("conn.close()", defer)
        self.assertIn("pg_advisory_unlock", lock_block)

    def test_reaper_recovers_pending_rows_whose_sqs_message_was_lost(self):
        reaper = (ROOT / "handlers" / "reap.py").read_text()
        self.assertIn('"pending_dispatches": _reap_pending_dispatches', reaper)
        recovery = reaper.split("def _reap_pending_dispatches", 1)[1].split(
            "def _reap_ingests", 1)[0]
        self.assertIn("status='pending' AND dispatched_at IS NOT NULL", recovery)
        self.assertIn("next_attempt_at <= NOW()", recovery)
        self.assertIn("SET dispatched_at=NULL", recovery)
        migration = (ROOT / "migrations" /
                     "013_pending_dispatch_recovery.sql").read_text()
        self.assertIn("stale_dispatched_minutes", migration)

    def test_status_exposes_scan_gate_and_delayed_queue_messages(self):
        status = (ROOT / "handlers" / "status.py").read_text()
        self.assertIn('(\"scan\", os.environ[\"SCAN_QUEUE_URL\"])', status)
        self.assertIn("ApproximateNumberOfMessagesDelayed", status)
        self.assertIn('out["dispatch_blocker"]', status)

    def test_final_scan_delivery_opens_the_dispatch_gate(self):
        dispatch = (ROOT / "common" / "dispatch.py").read_text()
        scan = (ROOT / "handlers" / "scan.py").read_text()
        self.assertIn("def run(cfg, scan_messages_to_ignore=0)", dispatch)
        self.assertIn("in_flight - ignore_in_flight", dispatch)
        self.assertIn("dispatch.run(cfg, scan_messages_to_ignore=1)", scan)
        self.assertIn("post-scan dispatch deferred to reaper", scan)


if __name__ == "__main__":
    unittest.main()
