#!/usr/bin/env python3
# scripts/invoke.py -- quoting-proof lambda invoker
#   python scripts/invoke.py migrate --action ping
#   python scripts/invoke.py migrate --action migrate
#   python scripts/invoke.py migrate --action enqueue_pending --arg limit=100
#   python scripts/invoke.py discover --arg force=true
import argparse, json, sys
import boto3
from botocore.config import Config
def main():
    p = argparse.ArgumentParser()
    p.add_argument("function")
    p.add_argument("--action", default=None)
    p.add_argument("--arg", action="append", default=[],
                   help="key=value; value parsed as JSON if possible")
    p.add_argument("--app", default="chat-ingest")
    p.add_argument("--endpoint", default="http://localhost:4566")
    p.add_argument("--region", default="us-east-1")
    args = p.parse_args()
    payload = {}
    if args.action:
        payload["action"] = args.action
    for kv in args.arg:
        k, _, v = kv.partition("=")
        try:
            payload[k] = json.loads(v)
        except json.JSONDecodeError:
            payload[k] = v
    lam = boto3.client("lambda", endpoint_url=args.endpoint,
                       region_name=args.region,
                       config=Config(read_timeout=910, retries={"max_attempts": 0}))
    resp = lam.invoke(FunctionName=f"{args.app}-{args.function}",
                      InvocationType="RequestResponse",
                      Payload=json.dumps(payload).encode())
    body = resp["Payload"].read().decode()
    try:
        print(json.dumps(json.loads(body), indent=2, ensure_ascii=False))
    except json.JSONDecodeError:
        print(body)
    sys.exit(1 if resp.get("FunctionError") else 0)
if __name__ == "__main__":
    main()