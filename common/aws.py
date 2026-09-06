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
_cache = {}
def endpoint():
    return _ENDPOINT
def client(service: str):
    if service not in _cache:
        kwargs = {"config": _CFG}
        if _ENDPOINT:
            kwargs["endpoint_url"] = _ENDPOINT
        _cache[service] = boto3.client(service, **kwargs)
    return _cache[service]