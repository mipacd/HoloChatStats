#!/usr/bin/env python3
"""Resumable one-time uploader for aggregate-only legacy stream statistics."""
import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


VIDEO_ID_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
PREFIX = "legacy-stream-stats-import/"
MAX_STATUS_IDS = 500
MAX_BATCH_FILES = 32
MAX_BATCH_BYTES = 64 * 1024 * 1024


def video_id_from_path(path):
    suffix = ".jsonl.gz"
    name = path.name
    video_id = name[:-len(suffix)] if name.endswith(suffix) else ""
    if len(video_id) != 11 or any(char not in VIDEO_ID_CHARS for char in video_id):
        return None
    return video_id


def build_batches(items, max_files=MAX_BATCH_FILES,
                  max_bytes=MAX_BATCH_BYTES):
    batch, size = [], 0
    for item in items:
        item_size = item[1].stat().st_size
        if batch and (len(batch) >= max_files or size + item_size > max_bytes):
            yield batch
            batch, size = [], 0
        batch.append(item)
        size += item_size
    if batch:
        yield batch


class Importer:
    def __init__(self, args):
        self.args = args
        common = dict(endpoint_url=args.endpoint_url,
                      region_name=args.region,
                      aws_access_key_id="test", aws_secret_access_key="test")
        self.s3 = boto3.client(
            "s3", **common,
            config=Config(connect_timeout=5, read_timeout=300,
                          retries={"max_attempts": 3, "mode": "standard"},
                          s3={"addressing_style": "path"}))
        self.lam = boto3.client(
            "lambda", **common,
            config=Config(connect_timeout=5, read_timeout=920,
                          retries={"max_attempts": 0}, tcp_keepalive=True))
        self.function = f"{args.app}-legacy-stats-import"
        self.bucket = args.bucket or f"{args.app}-raw-chat"
        self.counts = Counter()
        self.report_path = Path(args.report)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)

    def invoke(self, payload):
        response = self.lam.invoke(
            FunctionName=self.function, InvocationType="RequestResponse",
            Payload=json.dumps(payload, separators=(",", ":")).encode())
        raw = response["Payload"].read().decode("utf-8")
        try:
            body = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("import Lambda returned non-JSON output") from exc
        if response.get("FunctionError"):
            message = body.get("errorMessage") if isinstance(body, dict) else raw
            raise RuntimeError(f"import Lambda failed: {message}")
        return body

    def status(self, ids):
        last_error = None
        for attempt in range(1, self.args.retries + 1):
            try:
                return self.invoke({"action": "status", "video_ids": ids})
            except Exception as exc:
                last_error = exc
                if attempt < self.args.retries:
                    time.sleep(2 ** (attempt - 1))
        raise last_error

    def record(self, video_id, status, **details):
        entry = {"recorded_at": datetime.now(timezone.utc).isoformat(),
                 "video_id": video_id, "status": status, **details}
        with self.report_path.open("a", encoding="utf-8") as report:
            report.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self.counts[status] += 1

    def exists(self, key):
        try:
            self.s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def upload(self, path, key):
        last_error = None
        for attempt in range(1, self.args.retries + 1):
            try:
                self.s3.upload_file(str(path), self.bucket, key,
                                    ExtraArgs={"ContentType": "application/gzip"})
                return
            except Exception as exc:
                last_error = exc
                if attempt < self.args.retries:
                    time.sleep(2 ** (attempt - 1))
        raise last_error

    def cleanup(self, key):
        try:
            self.s3.delete_object(Bucket=self.bucket, Key=key)
        except Exception as exc:
            print(f"warning: cleanup failed for {key}: {type(exc).__name__}",
                  file=sys.stderr)

    def preflight(self, files):
        eligible = []
        for offset in range(0, len(files), MAX_STATUS_IDS):
            chunk = files[offset:offset + MAX_STATUS_IDS]
            by_id = {video_id: path for video_id, path in chunk}
            result = self.status(list(by_id))
            accounted = set(result.get("ready", [])) | set(
                result.get("eligible", [])) | set(result.get("missing", []))
            if accounted != set(by_id):
                raise RuntimeError("status response did not account for every video ID")
            for video_id in result.get("ready", []):
                self.record(video_id, "skipped-ready")
            for video_id in result.get("missing", []):
                self.record(video_id, "missing-video")
            eligible.extend((video_id, by_id[video_id])
                            for video_id in result.get("eligible", []))
            print(f"preflight {min(offset + len(chunk), len(files))}/{len(files)}",
                  flush=True)
        return eligible

    def staging_count(self):
        count, token = 0, None
        while True:
            kwargs = {"Bucket": self.bucket, "Prefix": PREFIX}
            if token:
                kwargs["ContinuationToken"] = token
            response = self.s3.list_objects_v2(**kwargs)
            count += len(response.get("Contents", []))
            if not response.get("IsTruncated"):
                return count
            token = response["NextContinuationToken"]

    def process_batch(self, batch):
        pending = {video_id: {"path": path, "failures": 0}
                   for video_id, path in batch}
        while pending:
            request_items = []
            for video_id, state in list(pending.items()):
                key = f"{PREFIX}{video_id}.jsonl.gz"
                try:
                    if not self.exists(key):
                        self.upload(state["path"], key)
                    request_items.append({"video_id": video_id, "key": key})
                except Exception as exc:
                    self.record(video_id, "failed", stage="upload",
                                error=f"{type(exc).__name__}: {str(exc)[:300]}")
                    pending.pop(video_id)
            if not request_items:
                continue
            try:
                response = self.invoke({"action": "import", "items": request_items})
            except Exception as exc:
                # The response can be lost after the database commit. Querying
                # status before retry prevents a newer ready aggregate being
                # needlessly retransferred or reported as failed.
                try:
                    current = self.status([item["video_id"] for item in request_items])
                    ready = set(current.get("ready", []))
                except Exception:
                    ready = set()
                for video_id in ready:
                    self.record(video_id, "processed", recovered_after_timeout=True)
                    self.cleanup(f"{PREFIX}{video_id}.jsonl.gz")
                    pending.pop(video_id, None)
                for item in request_items:
                    video_id = item["video_id"]
                    state = pending.get(video_id)
                    if state:
                        state["failures"] += 1
                    if state and state["failures"] >= self.args.retries:
                        self.record(video_id, "failed", stage="invoke",
                                    error=f"{type(exc).__name__}: {str(exc)[:300]}")
                        self.cleanup(item["key"])
                        pending.pop(video_id, None)
                if pending:
                    time.sleep(2 ** min(3, max(s["failures"] for s in pending.values())))
                continue
            terminal = {result.get("video_id"): result
                        for result in response.get("results", [])}
            deferred = {item.get("video_id")
                        for item in response.get("deferred", [])}
            for video_id, result in terminal.items():
                if video_id not in pending:
                    continue
                status_name = result.get("status", "failed")
                if status_name == "failed":
                    pending[video_id]["failures"] += 1
                    if pending[video_id]["failures"] < self.args.retries:
                        continue
                details = {key: value for key, value in result.items()
                           if key not in ("video_id", "status")}
                self.record(video_id, status_name, **details)
                self.cleanup(f"{PREFIX}{video_id}.jsonl.gz")
                pending.pop(video_id, None)
            # A malformed response must not create an infinite local loop.
            accounted = set(terminal) | deferred
            for item in request_items:
                video_id = item["video_id"]
                if (video_id in pending and video_id not in accounted
                        and video_id not in deferred):
                    pending[video_id]["failures"] += 1
                    if pending[video_id]["failures"] >= self.args.retries:
                        self.record(video_id, "failed", stage="response",
                                    error="Lambda omitted the item from its response")
                        self.cleanup(item["key"])
                        pending.pop(video_id, None)
            done = sum(self.counts.values())
            print(f"completed {done}: " + ", ".join(
                f"{key}={value}" for key, value in sorted(self.counts.items())),
                  flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--endpoint-url", required=True,
                        help="Floci edge URL, e.g. http://192.168.1.20:4566")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--app", default="chat-ingest")
    parser.add_argument("--bucket")
    parser.add_argument("--report",
                        default=".cache/legacy-stream-stats-import-report.jsonl")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.retries < 1:
        parser.error("--retries must be at least 1")
    return args


def main(argv=None):
    args = parse_args(argv)
    if not args.directory.is_dir():
        raise SystemExit(f"not a directory: {args.directory}")
    valid, invalid = [], []
    for path in sorted(args.directory.glob("*.jsonl.gz")):
        video_id = video_id_from_path(path)
        (valid if video_id else invalid).append((video_id, path))
    if args.limit is not None:
        valid = valid[:max(0, args.limit)]
    print(f"found {len(valid)} valid archive filenames; {len(invalid)} invalid")
    if args.dry_run:
        total = sum(path.stat().st_size for _, path in valid)
        print(f"dry run: {total / (1024 ** 3):.2f} GiB in "
              f"{sum(1 for _ in build_batches(valid))} bounded batches")
        return 0
    importer = Importer(args)
    for _, path in invalid:
        importer.record(path.name, "invalid-filename")
    eligible = importer.preflight(valid)
    print(f"uploading {len(eligible)} eligible archives", flush=True)
    for batch in build_batches(eligible):
        importer.process_batch(batch)
    staged = importer.staging_count()
    print(f"production staging objects remaining: {staged}")
    if staged:
        print("warning: remaining objects are protected by the one-day "
              "import-prefix lifecycle rule", file=sys.stderr)
    print("final: " + ", ".join(
        f"{key}={value}" for key, value in sorted(importer.counts.items())))
    return 1 if importer.counts["failed"] or staged else 0


if __name__ == "__main__":
    raise SystemExit(main())
