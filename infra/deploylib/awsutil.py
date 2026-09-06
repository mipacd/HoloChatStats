"""Thin boto3 plumbing shared by every provisioning step."""
import socket
import time
import urllib.request
import boto3
import botocore.exceptions
# short attribute -> real boto3 service name
ALIASES = {
    "lam": "lambda",
    "sm": "secretsmanager",
    "apigw": "apigatewayv2",
    "apigw_v1": "apigateway",
    "ec": "elasticache",
    "cw": "cloudwatch",
}
SERVICES = {
    "s3", "sqs", "lambda", "events", "iam", "ssm", "secretsmanager",
    "apigateway", "apigatewayv2", "rds", "sts", "ec2", "ecs", "ecr",
    "elasticache", "elbv2", "logs", "cloudwatch",
}
RETRYABLE = {"RequestTimeout", "Throttling", "ThrottlingException",
             "TooManyRequestsException", "ServiceUnavailable",
             "InternalError", "InternalFailure"}
CONNECTION_ERRORS = (botocore.exceptions.ConnectionClosedError,
                     botocore.exceptions.EndpointConnectionError,
                     botocore.exceptions.ReadTimeoutError,
                     botocore.exceptions.ConnectTimeoutError)
class Clients:
    """Lazy, endpoint-aware boto3 clients: clients.s3, clients.lam, clients.ecs ..."""
    def __init__(self, region, endpoint=None):
        self._session = boto3.session.Session(region_name=region)
        self._kwargs = {"endpoint_url": endpoint} if endpoint else {}
        self._cache = {}
    def __getattr__(self, name):
        svc = ALIASES.get(name, name)
        if svc not in SERVICES:
            raise AttributeError(name)
        if svc not in self._cache:
            self._cache[svc] = self._session.client(svc, **self._kwargs)
        return self._cache[svc]
def err_code(e):
    return getattr(e, "response", {}).get("Error", {}).get("Code", "")
def matches(e, *needles):
    """True if an error's code or message contains any needle (emulators are
    inconsistent about which one carries the signal)."""
    blob = f"{err_code(e)} {e}".lower()
    return any(n.lower() in blob for n in needles)
def retry(method, *args, _attempts=6, **kwargs):
    """Retry transient connection/throttle failures.  floci's JVM backend closes
    connections under load; real AWS throttles."""
    last = None
    for attempt in range(_attempts):
        try:
            return method(*args, **kwargs)
        except CONNECTION_ERRORS as e:
            last = e
        except botocore.exceptions.ClientError as e:
            if err_code(e) not in RETRYABLE:
                raise
            last = e
        if attempt < _attempts - 1:
            time.sleep(min(2 ** attempt, 16))
    raise last
def ignore(method, *args, only=(), **kwargs):
    """Call method, swallowing ClientError (optionally only matching `only`)."""
    try:
        return method(*args, **kwargs)
    except botocore.exceptions.ClientError as e:
        if only and not matches(e, *only):
            raise
        return None
def wait_for(probe, timeout=180, interval=2, report=None):
    """Poll probe() until truthy.  Returns the value, or None on timeout."""
    deadline, i = time.time() + timeout, 0
    while time.time() < deadline:
        value = probe()
        if value:
            return value
        if report and i % 5 == 0:
            report()
        i += 1
        time.sleep(interval)
    return None
def scan_local_ports(start, end, host="127.0.0.1", timeout=0.05):
    """Closed localhost ports refuse instantly, so 1000 ports takes < 1 s."""
    found = []
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            if s.connect_ex((host, port)) == 0:
                found.append(port)
    return found
def http_ok(url, timeout=5):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "deploy/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return 200 <= r.status < 400
    except Exception:
        return False