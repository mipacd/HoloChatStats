#!/usr/bin/env python3
"""Audit and repair shifted stream-stat aggregates from retained raw parts."""
import argparse
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.config import Config


class Repairer:
    def __init__(self, args):
        self.args = args
        self.function = f"{args.app}-legacy-stats-import"
        self.client = boto3.client(
            "lambda", endpoint_url=args.endpoint_url, region_name=args.region,
            aws_access_key_id="test", aws_secret_access_key="test",
            config=Config(connect_timeout=5, read_timeout=920,
                          retries={"max_attempts": 0}, tcp_keepalive=True))
        self.report = Path(args.report)
        self.report.parent.mkdir(parents=True, exist_ok=True)
        self.counts = Counter()

    def invoke(self, payload):
        last_error = None
        for attempt in range(self.args.retries):
            try:
                response = self.client.invoke(
                    FunctionName=self.function, InvocationType="RequestResponse",
                    Payload=json.dumps(payload, separators=(",", ":")).encode())
                raw = response["Payload"].read().decode("utf-8")
                body = json.loads(raw)
                if response.get("FunctionError"):
                    raise RuntimeError(body.get("errorMessage", raw))
                return body
            except Exception as exc:
                last_error = exc
                # Retrying a function that exhausted its full 15-minute budget
                # only repeats the same expensive work. The caller records it
                # and moves on so a local archive can repair it later.
                if "Task timed out after" in str(exc):
                    break
                if attempt + 1 < self.args.retries:
                    time.sleep(2 ** attempt)
        raise last_error

    def reconcile(self, video_id):
        """Check whether an ambiguous client failure hid a successful commit."""
        response = self.invoke({
            "action": "status", "video_ids": [video_id],
            "include_misaligned": True,
        })
        if video_id in response.get("ready", []):
            return {"video_id": video_id,
                    "status": "repaired-after-timeout"}
        return None

    def record(self, result):
        entry = {"recorded_at": datetime.now(timezone.utc).isoformat(), **result}
        with self.report.open("a", encoding="utf-8") as output:
            output.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self.counts[result.get("status", "failed")] += 1

    def candidates(self):
        cursor = None
        yielded = 0
        while True:
            response = self.invoke(
                {"action": "timing_audit", "cursor": cursor, "limit": 100})
            items = response.get("items", [])
            for item in items:
                if self.args.limit is not None and yielded >= self.args.limit:
                    return
                yielded += 1
                yield item
            cursor = response.get("next_cursor")
            if not cursor:
                return


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint-url", required=True)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--app", default="chat-ingest")
    parser.add_argument("--report",
                        default=".cache/stream-stats-timing-repair.jsonl")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.retries < 1:
        parser.error("--retries must be at least 1")
    return args


def main(argv=None):
    args = parse_args(argv)
    repairer = Repairer(args)
    found = 0
    for item in repairer.candidates():
        found += 1
        video_id = item["video_id"]
        if args.dry_run:
            print(f"suspect {video_id}: first={item.get('first_message_at')} "
                  f"start={item.get('calculated_start')} "
                  f"source={item.get('timing_source')} "
                  f"histogram={item.get('histogram_message_count')}/"
                  f"{item.get('message_count')} "
                  f"out_of_range={item.get('out_of_range_messages')}")
            continue
        try:
            response = repairer.invoke(
                {"action": "repair_retained", "video_ids": [video_id]})
            results = response.get("results", [])
            result = results[0] if results else {
                "video_id": video_id, "status": "failed",
                "error": "repair Lambda omitted the requested video"}
        except Exception as exc:
            try:
                result = repairer.reconcile(video_id)
            except Exception:
                result = None
            if result is None:
                result = {"video_id": video_id, "status": "failed",
                          "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
        repairer.record(result)
        print(f"{video_id}: {result.get('status')}", flush=True)
    if args.dry_run:
        print(f"dry run: {found} suspect aggregate(s)")
        return 0
    print("final: " + ", ".join(
        f"{key}={value}" for key, value in sorted(repairer.counts.items())))
    return 1 if (repairer.counts["failed"]
                 or repairer.counts["validation-failed"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
