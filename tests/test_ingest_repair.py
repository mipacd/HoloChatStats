import gzip
import os
import unittest
from unittest import mock

os.environ.setdefault("RAW_BUCKET", "test-raw-bucket")

from handlers import ingest


class IngestRepairTests(unittest.TestCase):
    def test_malformed_json_part_has_safe_diagnostic(self):
        body = gzip.compress(b'{"message":"unterminated}\n')
        stream = mock.Mock()
        stream.read.return_value = body
        s3 = mock.Mock()
        s3.get_object.return_value = {"Body": stream}

        with self.assertRaises(ingest.RawPartCorrupt) as raised:
            list(ingest._iter_messages(s3, "channel", "video", 1))

        error = str(raised.exception)
        self.assertIn("channel/video/part-00000.jsonl.gz", error)
        self.assertIn("line 1", error)
        self.assertNotIn("unterminated}\n", error)


if __name__ == "__main__":
    unittest.main()
