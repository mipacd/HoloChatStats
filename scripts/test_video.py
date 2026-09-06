#!/usr/bin/env python3
"""
End-to-end test of the download+ingest pipeline for a single video.
    python scripts/test_video.py UCxxxxxxxx dQw4w9WgXcQ --force
"""
import argparse
import json
import sys
import time
import boto3
from botocore.config import Config
TERMINAL = ("done", "failed", "skipped")
def invoke(lam, app, payload):
    resp = lam.invoke(FunctionName=f"{app}-migrate",
                      InvocationType="RequestResponse",
                      Payload=json.dumps(payload).encode())
    body = json.loads(resp["Payload"].read() or "{}")
    if resp.get("FunctionError"):
        sys.exit(f"\nlambda error: {json.dumps(body, indent=2)}")
    return body
def fmt(status):
    stage = status.get("status", "?")
    pct = status.get("download_pct")
    bits = [f"status={stage}"]
    if pct is not None:
        bits.append(f"download={pct:5.1f}%")
    if status.get("part_count"):
        bits.append(f"parts={status['part_count']}")
    if status.get("attempts"):
        bits.append(f"attempts={status['attempts']}")
    if status.get("user_data_rows"):
        bits.append(f"db_rows={status['user_data_rows']}")
    if status.get("last_error"):
        bits.append(f"err={str(status['last_error'])[:60]!r}")
    return "  ".join(bits)
def main():
    p = argparse.ArgumentParser()
    p.add_argument("channel_id")
    p.add_argument("video_id")
    p.add_argument("--force", action="store_true",
                   help="reset any previous attempt (DB job + raw S3 parts)")
    p.add_argument("--app", default="chat-ingest")
    p.add_argument("--endpoint", default="http://localhost:4566")
    p.add_argument("--region", default="us-east-1")
    p.add_argument("--interval", type=float, default=5.0)
    p.add_argument("--timeout", type=float, default=3600,
                   help="give up after this many seconds (default 1h)")
    args = p.parse_args()
    lam = boto3.client("lambda", endpoint_url=args.endpoint,
                       region_name=args.region,
                       config=Config(read_timeout=310,
                                     retries={"max_attempts": 0}))
    print(f"enqueueing {args.video_id} (channel {args.channel_id}) ...")
    res = invoke(lam, args.app, {"action": "enqueue_video",
                                 "channel_id": args.channel_id,
                                 "video_id": args.video_id,
                                 "force": args.force})
    if res.get("error"):
        sys.exit(f"cannot enqueue: {res['error']}")
    print(json.dumps({k: res[k] for k in ("title", "duration_s", "enqueued",
                                          "note") if k in res},
                     ensure_ascii=False, indent=2))
    if not res.get("enqueued") and res.get("status") in TERMINAL:
        sys.exit(0)
    t0 = time.time()
    last_line = ""
    while time.time() - t0 < args.timeout:
        status = invoke(lam, args.app, {"action": "job_status",
                                        "video_id": args.video_id})
        line = f"[{int(time.time() - t0):4d}s] {fmt(status)}"
        if line != last_line:
            print(line)
            last_line = line
        if status.get("status") in TERMINAL:
            print("\nfinal state:")
            print(json.dumps(status, indent=2, ensure_ascii=False, default=str))
            sys.exit(0 if status["status"] == "done" else 1)
        time.sleep(args.interval)
    sys.exit(f"timed out after {args.timeout}s -- job may still be running; "
             f"re-run with a longer --timeout or check /status")
if __name__ == "__main__":
    main()