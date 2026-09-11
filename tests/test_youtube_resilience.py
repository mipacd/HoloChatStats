import json
import unittest
from unittest import mock
from types import SimpleNamespace

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
        _key, _version, decoded, context = youtube._fetch_params(
            "https://example.test")
        self.assertEqual(decoded, {"ok": True})
        self.assertEqual(context["client"]["clientName"], "WEB")
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

    def test_live_chat_continuation_wins_over_unrelated_page_token(self):
        initial = {
            "continuation": "wrong-comments-token",
            "contents": {"twoColumnWatchNextResults": {
                "conversationBar": {"liveChatRenderer": {
                    "continuations": [{"reloadContinuationData": {
                        "continuation": "right-chat-token"}}]
                }}}},
        }
        self.assertEqual(youtube._find_continuation(initial),
                         "right-chat-token")

    def test_ytcfg_context_keeps_visitor_data(self):
        html = ('ytcfg.set({"INNERTUBE_CONTEXT":{"client":'
                '{"clientName":"WEB","clientVersion":"1.2",'
                '"visitorData":"visitor-token"}}});')
        context = youtube._extract_innertube_context(html, "fallback")
        self.assertEqual(context["client"]["visitorData"], "visitor-token")

    @mock.patch.object(youtube, "_auth")
    def test_chat_post_uses_full_context_and_player_state(self, auth):
        session = mock.Mock()
        session.cookies = []
        session.post.return_value = self._response(200, '{"actions": []}')
        auth.return_value = {"session": session}
        context = {"client": {"clientName": "WEB", "clientVersion": "1.2",
                              "visitorData": "visitor-token"}}
        youtube._fetch_chat("key", "1.2", "chat-token", context=context,
                            player_offset_ms=9000)
        kwargs = session.post.call_args.kwargs
        self.assertEqual(kwargs["json"]["context"], context)
        self.assertEqual(kwargs["json"]["currentPlayerState"]["playerOffsetMs"],
                         "4000")
        self.assertEqual(kwargs["headers"]["X-Goog-Visitor-Id"],
                         "visitor-token")

    @mock.patch.object(youtube.time, "time", return_value=1234.4)
    @mock.patch.object(youtube, "_auth")
    def test_chat_post_authenticates_cookie_session(self, auth, _time):
        session = mock.Mock()
        session.cookies = [SimpleNamespace(
            name="__Secure-3PAPISID", value="secret-cookie")]
        session.post.return_value = self._response(200, '{"actions": []}')
        auth.return_value = {"session": session}
        youtube._fetch_chat("key", "1.2", "chat-token")
        header = session.post.call_args.kwargs["headers"]["Authorization"]
        self.assertTrue(header.startswith("SAPISIDHASH 1234_"))
        self.assertIn("SAPISID3PHASH 1234_", header)

    def test_unfiltered_replay_keeps_selector_tracking(self):
        response = {"continuationContents": {"liveChatContinuation": {
            "header": {"liveChatHeaderRenderer": {"viewSelector": {
                "sortFilterSubMenuRenderer": {"subMenuItems": [
                    {"continuation": {"reloadContinuationData": {
                        "continuation": "top"}}},
                    {"continuation": {"reloadContinuationData": {
                        "continuation": "live",
                        "trackingParams": "tracked"}}},
                ]}}}}}}}
        self.assertEqual(
            youtube._extract_unfiltered_cont_data(response),
            ("live", "tracked"))

    @mock.patch.object(youtube.time, "sleep")
    @mock.patch.object(youtube, "_fetch_chat")
    @mock.patch.object(youtube, "_fetch_initial_chat")
    @mock.patch.object(youtube, "_fetch_params")
    def test_replay_bootstraps_unfiltered_chat_before_posting(
            self, params, initial_chat, fetch_chat, _sleep):
        context = {"client": {"clientName": "WEB",
                              "clientVersion": "1.2"}}
        params.return_value = (
            "key", "1.2", {"continuation": "watch-token"}, context)
        initial_chat.return_value = {"continuationContents": {
            "liveChatContinuation": {"header": {"liveChatHeaderRenderer": {
                "viewSelector": {"sortFilterSubMenuRenderer": {
                    "subMenuItems": [
                        {"continuation": {"reloadContinuationData": {
                            "continuation": "top-chat-token"}}},
                        {"continuation": {"reloadContinuationData": {
                            "continuation": "live-chat-token"}}},
                    ]
                }}
            }}}}}
        fetch_chat.return_value = {"continuationContents": {
            "liveChatContinuation": {"actions": [], "continuations": []}}}
        replay = youtube.ChatReplay(
            "video-id", video_start_ts=1234.0, duration=20)
        self.assertEqual(list(replay.pages()), [([], None)])
        initial_chat.assert_called_once_with("watch-token")
        self.assertEqual(fetch_chat.call_args.args[:3],
                         ("key", "1.2", "live-chat-token"))
        self.assertEqual(fetch_chat.call_args.kwargs["context"], context)

    @mock.patch.object(youtube.time, "sleep")
    @mock.patch.object(youtube, "_ydl_options", return_value={"quiet": True})
    @mock.patch.object(youtube, "YoutubeDL")
    def test_yt_dlp_metadata_retries_malformed_response(
            self, ydl_cls, _options, sleep):
        first = mock.MagicMock()
        first.__enter__.return_value.extract_info.side_effect = ValueError(
            "Unterminated string starting at line 1")
        second = mock.MagicMock()
        second.__enter__.return_value.extract_info.return_value = {
            "duration": 10,
        }
        ydl_cls.side_effect = [first, second]
        result = youtube._extract_video_info("https://example.test")
        self.assertEqual(result, {"duration": 10})
        self.assertEqual(ydl_cls.call_count, 2)
        sleep.assert_called_once()

    @mock.patch.object(
        youtube, "_fetch_params",
        return_value=("key", "version", {"continuation": "initial-token"}))
    @mock.patch.object(youtube, "_extract_video_info")
    def test_replay_uses_discovery_metadata_without_yt_dlp(
            self, extract_info, _fetch_params):
        replay = youtube.ChatReplay(
            "video-id", video_start_ts=1234.0, duration=5678)
        self.assertEqual(replay.video_start_ts, 1234.0)
        self.assertEqual(replay.duration, 5678)
        self.assertEqual(replay.continuation, "initial-token")
        extract_info.assert_not_called()


if __name__ == "__main__":
    unittest.main()
