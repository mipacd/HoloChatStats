"""Validate the CI YouTube cookie secret without printing secret values."""
import base64
import binascii
import os
import sys
import time


AUTH_NAMES = {
    "SID", "HSID", "SSID", "APISID", "SAPISID", "LOGIN_INFO",
    "__Secure-1PSID", "__Secure-3PSID",
    "__Secure-1PAPISID", "__Secure-3PAPISID",
}


def inspect(encoded):
    try:
        raw = base64.b64decode(encoded, validate=True)
        lines = raw.decode("utf-8-sig").replace("\r\n", "\n").splitlines()
    except (binascii.Error, UnicodeDecodeError) as exc:
        raise ValueError(f"not valid UTF-8 base64: {exc}") from exc
    if not lines or lines[0] not in (
            "# HTTP Cookie File", "# Netscape HTTP Cookie File"):
        raise ValueError("not a Netscape cookies.txt file")
    now = int(time.time())
    cookies = []
    for line in lines[1:]:
        if line.startswith("#HttpOnly_"):
            line = line[len("#HttpOnly_"):]
        elif not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 7:
            continue
        domain, _subdomains, _path, _secure, expires, name, _value = fields
        try:
            unexpired = int(expires) == 0 or int(expires) > now
        except ValueError:
            unexpired = False
        cookies.append((domain.lower().lstrip("."), name, unexpired))
    relevant = [c for c in cookies
                if c[0].endswith(("youtube.com", "google.com")) and c[2]]
    auth = [c for c in relevant if c[1] in AUTH_NAMES]
    return {"cookies": len(cookies), "relevant": len(relevant),
            "auth": len(auth)}


def main():
    encoded = os.environ.get("YOUTUBE_COOKIES_B64", "")
    if not encoded:
        raise SystemExit("YOUTUBE_COOKIES_B64 is empty")
    try:
        summary = inspect(encoded)
    except ValueError as exc:
        raise SystemExit(f"Invalid YOUTUBE_COOKIES_B64: {exc}") from exc
    if not summary["relevant"]:
        raise SystemExit("Cookie file has no unexpired youtube.com/google.com cookies")
    if not summary["auth"]:
        raise SystemExit(
            "Cookie file has no recognized unexpired Google account session cookies; "
            "export it while signed into an age-verified YouTube account")
    print("YouTube cookie secret validated: "
          f"{summary['cookies']} entries, {summary['relevant']} active YouTube/Google, "
          f"{summary['auth']} account-session markers (values not displayed).")


if __name__ == "__main__":
    main()
