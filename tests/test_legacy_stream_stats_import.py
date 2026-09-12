import gzip
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("RAW_BUCKET", "test-raw")

from common.stream_stats import StreamStatsAccumulator
from handlers import legacy_stats_import as worker
from scripts.import_legacy_stream_stats import (
    MAX_BATCH_BYTES, build_batches, video_id_from_path,
)


def legacy(uid, timestamp, text="archiveword", category="es_en_id", rank=-1,
           **extra):
    return {"user_id": uid, "username": "discard me", "timestamp": timestamp,
            "message": text, "message_category": category,
            "membership_rank": rank, **extra}


class FakeS3:
    def __init__(self, body):
        self.body = body

    def get_object(self, **_kwargs):
        return {"Body": io.BytesIO(self.body)}

    def delete_object(self, **kwargs):
        self.deleted = kwargs


def archive(*lines):
    raw = b"\n".join(line if isinstance(line, bytes)
                      else json.dumps(line).encode() for line in lines)
    return gzip.compress(raw)


class LegacyAccumulatorTests(unittest.TestCase):
    def test_microseconds_seconds_categories_ranks_and_gifts(self):
        start = 1_700_000_000
        acc = StreamStatsAccumulator(900, start)
        for i in range(5):
            timestamp = ((start + i) * 1_000_000 if i < 2
                         else start + i + 0.25)
            self.assertTrue(acc.add_legacy(
                legacy(f"u{i}", timestamp, category="jp",
                       rank=6 if i == 0 else -1)))
        acc.add_legacy(legacy("gift", start + 65, text="", category=None,
                              rank=-2, message_type="gift_member",
                              gifter="discard me too"))
        result = acc.finish()
        self.assertEqual(result["message_count"], 5)
        self.assertEqual(result["unique_chatters"], 6)
        self.assertEqual(result["category_counts"], {"jp": 5})
        self.assertEqual(result["membership_rank_counts"]["6"], 1)
        self.assertEqual(result["membership_rank_counts"]["-2"], 1)
        self.assertEqual(result["histogram_counts"][0], 5)
        self.assertEqual(result["first_message_at"], start)
        self.assertEqual(result["word_counts"], [["archiveword", 5]])
        for forbidden in ("users", "user_id", "username", "messages", "gifter"):
            self.assertNotIn(forbidden, result)

    def test_unknown_category_falls_back_and_missing_timing_stays_empty(self):
        acc = StreamStatsAccumulator(0, None)
        acc.add_legacy(legacy("u", 1_700_000_000, text="12345",
                              category="old-unknown"))
        result = acc.finish()
        self.assertEqual(result["category_counts"], {"number": 1})
        self.assertEqual(result["histogram_counts"], [])
        self.assertEqual(result["funny_moments"], [])

    def test_all_archived_categories_and_legacy_humor_timing(self):
        start = 1_700_000_000
        acc = StreamStatsAccumulator(7200, start)
        for index, category in enumerate(
                ("emoji", "jp", "kr", "ru", "number", "es_en_id")):
            acc.add_legacy(legacy(str(index), start + 30 + index,
                                  text="hahaha", category=category))
        result = acc.finish()
        self.assertEqual(set(result["category_counts"]),
                         {"emoji", "jp", "kr", "ru", "number", "es_en_id"})
        self.assertEqual(sum(result["histogram_counts"]), 6)
        self.assertTrue(result["funny_moments"])


class LegacyWorkerTests(unittest.TestCase):
    def aggregate(self, payload):
        with mock.patch.object(worker, "client", return_value=FakeS3(payload)):
            return worker._aggregate("legacy-stream-stats-import/abcdefghijk.jsonl.gz",
                                     600, 1_700_000_000)

    def test_malformed_rows_are_skipped_but_valid_data_is_kept(self):
        payload = archive(b'{"broken":', legacy("u", 1_700_000_001))
        aggregate, counts = self.aggregate(payload)
        self.assertEqual(counts, {"records": 1, "malformed": 1, "empty": False})
        self.assertEqual(aggregate["message_count"], 1)

    def test_empty_archive_is_a_valid_zero_message_aggregate(self):
        aggregate, counts = self.aggregate(archive())
        self.assertTrue(counts["empty"])
        self.assertEqual(aggregate["message_count"], 0)

    def test_corrupt_or_nonempty_all_invalid_archive_fails(self):
        with self.assertRaises(gzip.BadGzipFile):
            self.aggregate(b"not gzip")
        with self.assertRaisesRegex(ValueError, "no valid legacy records"):
            self.aggregate(archive({}))

    def test_batch_defers_before_starting_work_near_timeout(self):
        class Context:
            def get_remaining_time_in_millis(self):
                return 59_999
        result = worker.import_batch([
            {"video_id": "abcdefghijk",
             "key": "legacy-stream-stats-import/abcdefghijk.jsonl.gz"}
        ], Context())
        self.assertEqual(result["results"], [])
        self.assertEqual(len(result["deferred"]), 1)

    def test_terminal_failure_deletes_staged_object_without_logging_data(self):
        store = FakeS3(b"")
        key = "legacy-stream-stats-import/abcdefghijk.jsonl.gz"
        with mock.patch.object(worker, "_timing_and_state",
                               return_value=(600, 1_700_000_000, False)), \
                mock.patch.object(worker, "_aggregate",
                                  side_effect=ValueError("invalid archive")), \
                mock.patch.object(worker, "client", return_value=store):
            result = worker._process("abcdefghijk", key)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(store.deleted, {"Bucket": worker.BUCKET, "Key": key})

    def test_status_skips_ready_and_reports_missing_without_aws_calls(self):
        class Cursor:
            def __enter__(self): return self
            def __exit__(self, *_args): return None
            def execute(self, *_args): pass
            def fetchall(self): return [("abcdefghijk", True),
                                        ("lmnopqrstuv", False)]
        class Connection:
            def cursor(self): return Cursor()
            def rollback(self): pass
            def close(self): pass
        with mock.patch.object(worker, "get_conn", return_value=Connection()), \
                mock.patch.object(worker, "client") as aws_client:
            result = worker.status(
                ["abcdefghijk", "lmnopqrstuv", "wxyzABCDEFG"])
        self.assertEqual(result["ready"], ["abcdefghijk"])
        self.assertEqual(result["eligible"], ["lmnopqrstuv"])
        self.assertEqual(result["missing"], ["wxyzABCDEFG"])
        aws_client.assert_not_called()


class LegacyUploaderTests(unittest.TestCase):
    def test_filename_validation_and_bounded_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = []
            for index, size in enumerate((10, MAX_BATCH_BYTES, 10)):
                video_id = f"v{index:010d}"
                path = root / f"{video_id}.jsonl.gz"
                with path.open("wb") as output:
                    output.truncate(size)
                files.append((video_id, path))
                self.assertEqual(video_id_from_path(path), video_id)
            self.assertEqual(len(list(build_batches(files))), 3)
            self.assertIsNone(video_id_from_path(root / "notvideo.jsonl.gz"))

    def test_deployment_has_isolated_worker_and_one_day_cleanup(self):
        root = Path(__file__).resolve().parents[1]
        config = (root / "infra/deploylib/config.py").read_text(encoding="utf-8")
        storage = (root / "infra/deploylib/storage.py").read_text(encoding="utf-8")
        handler = (root / "handlers/legacy_stats_import.py").read_text(
            encoding="utf-8")
        self.assertIn('"legacy-stats-import"', config)
        self.assertIn("LEGACY_IMPORT_RETENTION_DAYS = 1", config)
        self.assertIn("LEGACY_IMPORT_PREFIX", storage)
        self.assertIn('client("s3").delete_object', handler)
        self.assertNotIn("INGEST_QUEUE_URL", handler)


if __name__ == "__main__":
    unittest.main()
