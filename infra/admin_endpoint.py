"""
Works out which URL the emulator actually serves the admin Lambda on, by
probing every known API Gateway URL shape and checking the response body.
Needed because emulators disagree on edge routing, and several candidate shapes
collide with path-style S3 (which answers NoSuchBucket instead of 404).
"""
import json
import urllib.error
import urllib.request
from urllib.parse import urlsplit, urlunsplit
TITLE = "HoloChatStats Admin Page"
S3_HINTS = ("NoSuchBucket", "<Error>", "AccessDenied")
TIMEOUT = 8
def edge(endpoint):
    return (endpoint or "http://localhost:4566").rstrip("/")
def localize(url, endpoint):
    """Rewrite a reported endpoint's host/port onto the edge we can reach.
    Emulators often report an internal hostname (floci) or a wildcard DNS name
    that doesn't resolve from the host."""
    if not url:
        return None
    want = urlsplit(edge(endpoint))
    have = urlsplit(url if "//" in url else "http://" + url)
    return urlunsplit((want.scheme or "http", want.netloc, have.path or "/", "", ""))
def _get(url, timeout=TIMEOUT):
    req = urllib.request.Request(url, headers={"Accept": "text/html,*/*"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, dict(r.headers), r.read(8192).decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read(4096).decode("utf-8", "replace")
    except Exception as e:
        return None, {}, f"{type(e).__name__}: {e}"
def probe(base):
    """Return (ok, note). `base` must end with '/'."""
    status, headers, body = _get(base)
    ct = (headers.get("Content-Type") or headers.get("content-type") or "").lower()
    if status and TITLE in body:
        if "text/html" not in ct:
            return True, (f"serves the page but Content-Type is {ct or 'absent'!r} "
                          f"-- redeploy handlers/admin.py (_resp)")
        return True, "serves the admin page (text/html)"
    status2, _, body2 = _get(base + "api/status")
    if status2 == 200 and '"control"' in body2:
        return True, "serves api/status (page route may need /index.html)"
    if status is None:
        return False, body[:90]
    if any(h in body for h in S3_HINTS):
        return False, f"HTTP {status} intercepted by S3 (path-style bucket)"
    return False, f"HTTP {status} {' '.join(body.split())[:70]}"
def candidates(endpoint, rest=None, http=None):
    """
    rest = {"id": ..., "stages": [...]}   v1 REST API
    http = {"id": ..., "stages": [...], "reported": ApiEndpoint}  v2 HTTP API
    Ordered most- to least-likely; shapes that collide with S3 come last so a
    working answer is found before a misleading NoSuchBucket.
    """
    e = edge(endpoint)
    port = (urlsplit(e).netloc.rsplit(":", 1) + ["4566"])[1]
    out = []
    if rest and rest.get("id"):
        rid = rest["id"]
        for st in (rest.get("stages") or []) + ["admin", "$default", "default", "prod"]:
            out.append(f"{e}/restapis/{rid}/{st}/_user_request_/")
            out.append(f"{e}/restapis/{rid}/{st}/")
        out.append(f"http://{rid}.execute-api.localhost:{port}/admin/")
    if http and http.get("id"):
        hid = http["id"]
        loc = localize(http.get("reported"), endpoint)
        if loc:
            out.append(loc.rstrip("/") + "/")
        for st in (http.get("stages") or []) + ["$default"]:
            out.append(f"{e}/restapis/{hid}/{st}/_user_request_/")
        out.append(f"{e}/_aws/execute-api/{hid}/")
        out.append(f"{e}/{hid}/")            # S3-colliding shapes last
        out.append(f"http://{hid}.execute-api.localhost:{port}/")
    seen, uniq = set(), []
    for u in out:
        u = u if u.endswith("/") else u + "/"
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    return uniq
def discover(endpoint, rest=None, http=None, verbose=True):
    """Probe candidates; return (working_url_or_None, report rows)."""
    report = []
    winner = None
    for url in candidates(endpoint, rest, http):
        ok, note = probe(url)
        report.append((ok, url, note))
        if verbose:
            print(f"  {'OK  ' if ok else '--  '}{url}\n      {note}")
        if ok and winner is None:
            winner = url
            if verbose:
                continue          # keep probing only to fill the report
            break
    return winner, report