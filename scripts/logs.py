#!/usr/bin/env python3
"""Tail-ish log reader that tolerates partial CloudWatch Logs implementations.
    python scripts/logs.py migrate --since 10m
    python scripts/logs.py download --follow
"""
import argparse
import time
from datetime import datetime, timezone
import boto3
from botocore.config import Config
def parse_since(s):
    unit, mult = s[-1], {"s": 1, "m": 60, "h": 3600, "d": 86400}
    return int(s[:-1]) * mult[unit] * 1000
def main():
    p = argparse.ArgumentParser()
    p.add_argument("function", help="short name, e.g. migrate / download / scan")
    p.add_argument("--app", default="chat-ingest")
    p.add_argument("--endpoint", default="http://localhost:4566")
    p.add_argument("--region", default="us-east-1")
    p.add_argument("--since", default="15m")
    p.add_argument("--filter", default=None, help="filter pattern, e.g. ERROR")
    p.add_argument("--follow", action="store_true")
    args = p.parse_args()
    logs = boto3.client("logs", endpoint_url=args.endpoint, region_name=args.region,
                        config=Config(read_timeout=30))
    group = f"/aws/lambda/{args.app}-{args.function}"
    start = int(time.time() * 1000) - parse_since(args.since)
    seen = set()
    while True:
        kwargs = {"logGroupName": group, "startTime": start}
        if args.filter:
            kwargs["filterPattern"] = args.filter
        try:
            token = None
            while True:
                if token:
                    kwargs["nextToken"] = token
                resp = logs.filter_log_events(**kwargs)
                for e in resp.get("events", []):
                    key = (e.get("eventId") or
                           f"{e['timestamp']}:{hash(e['message'])}")
                    if key in seen:
                        continue
                    seen.add(key)
                    ts = datetime.fromtimestamp(e["timestamp"] / 1000, timezone.utc)
                    print(f"{ts:%H:%M:%S} {e['message'].rstrip()}")
                    start = max(start, e["timestamp"] + 1)
                token = resp.get("nextToken")
                if not token:
                    break
        except logs.exceptions.ResourceNotFoundException:
            print(f"log group {group} does not exist yet "
                  f"(function has never been invoked, or floci is not "
                  f"capturing lambda logs -- fall back to `docker logs`)")
            if not args.follow:
                return
        if not args.follow:
            return
        time.sleep(2)
if __name__ == "__main__":
    main()