import base64
import time
import unittest

from scripts.validate_youtube_cookies import inspect


def encoded(*rows):
    text = "# Netscape HTTP Cookie File\n" + "\n".join(rows) + "\n"
    return base64.b64encode(text.encode()).decode()


class CookieValidationTests(unittest.TestCase):
    def test_finds_unexpired_account_cookie_without_exposing_value(self):
        future = int(time.time()) + 3600
        result = inspect(encoded(
            f".youtube.com\tTRUE\t/\tTRUE\t{future}\tLOGIN_INFO\tsecret-value"
        ))
        self.assertEqual(result, {"cookies": 1, "relevant": 1, "auth": 1})
        self.assertNotIn("secret-value", str(result))

    def test_ignores_expired_account_cookie(self):
        result = inspect(encoded(
            ".youtube.com\tTRUE\t/\tTRUE\t1\tLOGIN_INFO\tsecret-value"
        ))
        self.assertEqual(result, {"cookies": 1, "relevant": 0, "auth": 0})

    def test_accepts_http_only_netscape_rows(self):
        result = inspect(encoded(
            "#HttpOnly_.google.com\tTRUE\t/\tTRUE\t0\tSID\tsecret-value"
        ))
        self.assertEqual(result, {"cookies": 1, "relevant": 1, "auth": 1})


if __name__ == "__main__":
    unittest.main()
