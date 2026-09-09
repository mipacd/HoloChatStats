import json
import unittest
from unittest import mock

import requests

from common import youtube


class YoutubeResponseTests(unittest.TestCase):
    @staticmethod
    def _response(status, body, retry_after=None):
        response = mock.Mock()
        response.status_code = status
        response.content = body.encode("utf-8")
        response.text = body
        response.headers = ({"Retry-After": retry_after}
                            if retry_after is not None else {})
        if status >= 400:
            response.raise_for_status.side_effect = requests.HTTPError(
                f"{status} response", response=response)
        return response

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

    def test_initial_data_skips_an_earlier_malformed_candidate(self):
        html = (
            'ytInitialData = {"broken":"unterminated}; '
            'var ytInitialData = {"valid":true};'
        )
        _key, _version, decoded = youtube._extract_params(html)
        self.assertEqual(decoded, {"valid": True})

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

    @mock.patch.object(youtube.time, "sleep")
    @mock.patch.object(youtube, "_auth")
    def test_chat_page_retries_503_and_honors_retry_after(self, auth, sleep):
        session = mock.Mock()
        session.post.side_effect = [
            self._response(503, "unavailable", "0.5"),
            self._response(200, '{"actions": []}'),
        ]
        auth.return_value = {"session": session}
        result = youtube._fetch_chat("key", "version", "continuation")
        self.assertEqual(result, {"actions": []})
        self.assertEqual(session.post.call_count, 2)
        sleep.assert_called_once_with(0.5)

    @mock.patch.object(youtube.time, "sleep")
    @mock.patch.object(youtube, "_auth")
    def test_chat_page_retries_malformed_json(self, auth, sleep):
        session = mock.Mock()
        session.post.side_effect = [
            self._response(200, '{"actions": [}'),
            self._response(200, '{"actions": []}'),
        ]
        auth.return_value = {"session": session}
        result = youtube._fetch_chat("key", "version", "continuation")
        self.assertEqual(result, {"actions": []})
        self.assertEqual(session.post.call_count, 2)
        sleep.assert_called_once()

    @mock.patch.object(youtube.time, "sleep")
    @mock.patch.object(youtube, "_auth")
    def test_chat_page_does_not_retry_non_transient_http_error(self, auth, sleep):
        session = mock.Mock()
        session.post.return_value = self._response(403, "forbidden")
        auth.return_value = {"session": session}
        with self.assertRaises(requests.HTTPError):
            youtube._fetch_chat("key", "version", "continuation")
        self.assertEqual(session.post.call_count, 1)
        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
