import gzip
import os
import unittest
from unittest import mock

os.environ.setdefault("RAW_BUCKET", "test-raw-bucket")

from handlers import ingest


class IngestRepairTests(unittest.TestCase):
    def test_malformed_json_line_is_skipped_without_losing_valid_lines(self):
        body = gzip.compress(
            b'{"author":{"id":"one"},"message":"valid"}\n'
            b'{"message":"unterminated}\n'
            b'{"author":{"id":"two"},"message":"also valid"}\n')
        stream = mock.Mock()
        stream.read.return_value = body
        s3 = mock.Mock()
        s3.get_object.return_value = {"Body": stream}

        with mock.patch.object(ingest, "emit") as emit:
            messages = list(ingest._iter_messages(s3, "channel", "video", 1))
        self.assertEqual([item["author"]["id"] for item in messages],
                         ["one", "two"])
        emit.assert_called_once()

    def test_invalid_gzip_still_rejects_the_part(self):
        stream = mock.Mock()
        stream.read.return_value = b"not gzip"
        s3 = mock.Mock()
        s3.get_object.return_value = {"Body": stream}
        with self.assertRaises(ingest.RawPartCorrupt):
            list(ingest._iter_messages(s3, "channel", "video", 1))


if __name__ == "__main__":
    unittest.main()
