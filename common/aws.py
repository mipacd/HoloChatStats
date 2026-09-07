import os
import boto3
from botocore.config import Config
# Our own variable name first: Lambda reserves/validates some AWS_* keys, and
# some emulators inject their own AWS_ENDPOINT_URL that would shadow ours.
_ENDPOINT = (os.environ.get("INGEST_ENDPOINT_URL")
             or os.environ.get("AWS_ENDPOINT_URL")
             or None)
# Fail fast instead of hanging: if the endpoint is wrong we want an error in
# seconds, not a 150-second retry storm that reads as a Lambda timeout.
_CFG = Config(
    retries={"max_attempts": 3, "mode": "standard"},
    connect_timeout=3,
    read_timeout=10,
)
_S3_CFG = Config(
    retries={"max_attempts": 5, "mode": "standard"},
    connect_timeout=3,
    # Floci serves S3 from the same modest host as RDS and Lambda. Ten seconds
    # is too aggressive while a compressed chat part is being read or written.
    read_timeout=int(os.environ.get("S3_READ_TIMEOUT_SECONDS", "120")),
    tcp_keepalive=True,
)
_cache = {}
def endpoint():
    return _ENDPOINT
def client(service: str):
    if service not in _cache:
        kwargs = {"config": _S3_CFG if service == "s3" else _CFG}
        if _ENDPOINT:
            kwargs["endpoint_url"] = _ENDPOINT
        _cache[service] = boto3.client(service, **kwargs)
    return _cache[service]
