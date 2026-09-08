import json
import unittest
from unittest import mock

from common import youtube


class YoutubeResponseTests(unittest.TestCase):
    def test_initial_data_uses_balanced_json_decoder(self):
        payload = {"nested": {"value": "a } brace"}, "items": [1, 2]}
        html = (
            'INNERTUBE_API_KEY":"key"; '
            'INNERTUBE_CONTEXT_CLIENT_VERSION":"1.2"; '
            f'var ytInitialData = {json.dumps(payload)}; trailing();'
        )
        key, version, decoded = youtube._extract_params(html)
        self.assertEqual((key, version), ("key", "1.2"))
        self.assertEqual(decoded, payload)

    @mock.patch.object(youtube.time, "sleep")
    @mock.patch.object(youtube, "_fetch_html")
    def test_truncated_watch_page_is_retried(self, fetch, _sleep):
        fetch.side_effect = [
            'var ytInitialData = {"nested":"unterminated};',
            'var ytInitialData = {"ok":true};',
        ]
        _key, _version, decoded = youtube._fetch_params("https://example.test")
        self.assertEqual(decoded, {"ok": True})
        self.assertEqual(fetch.call_count, 2)


if __name__ == "__main__":
    unittest.main()
