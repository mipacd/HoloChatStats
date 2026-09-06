#!/usr/bin/env python3
"""Find (or re-find) the HoloChatStats Admin Page URL.
    python scripts/admin_url.py
    python scripts/admin_url.py --open
"""
import argparse
import sys
from pathlib import Path
import boto3
import botocore.exceptions
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "infra"))
from admin_endpoint import discover, probe            # noqa: E402
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint", default="http://localhost:4566")
    p.add_argument("--region", default="us-east-1")
    p.add_argument("--app", default="chat-ingest")
    p.add_argument("--open", action="store_true")
    p.add_argument("--refresh", action="store_true",
                   help="ignore the cached URL in SSM and re-probe everything")
    args = p.parse_args()
    def mk(svc):
        return boto3.client(svc, endpoint_url=args.endpoint, region_name=args.region)
    ssm = mk("ssm")
    if not args.refresh:
        try:
            cached = ssm.get_parameter(Name=f"/{args.app}/admin/url")["Parameter"]["Value"]
            ok, note = probe(cached)
            print(f"  {'OK  ' if ok else '--  '}{cached}\n      {note} (cached)")
            if ok:
                _finish(cached, args)
                return
        except botocore.exceptions.ClientError:
            pass
    rest = http = None
    try:
        v1 = mk("apigateway")
        api = next((a for a in v1.get_rest_apis().get("items", [])
                    if a["name"] == f"{args.app}-admin-rest"), None)
        if api:
            stages = [s["stageName"] for s in
                      v1.get_stages(restApiId=api["id"]).get("item", [])]
            rest = {"id": api["id"], "stages": stages or ["admin"]}
    except botocore.exceptions.ClientError as e:
        print(f"(apigateway v1 unavailable: {e})")
    try:
        v2 = mk("apigatewayv2")
        api = next((a for a in v2.get_apis()["Items"]
                    if a["Name"] == f"{args.app}-admin"), None)
        if api:
            stages = [s["StageName"] for s in
                      v2.get_stages(ApiId=api["ApiId"]).get("Items", [])]
            http = {"id": api["ApiId"], "stages": stages,
                    "reported": api.get("ApiEndpoint")}
    except botocore.exceptions.ClientError:
        pass
    if not rest and not http:
        raise SystemExit("no admin API found -- run infra/deploy.py first")
    url, report = discover(args.endpoint, rest=rest, http=http)
    if not url:
        print("\nnothing served the page. The lambda is reachable directly:")
        print("  python scripts/invoke.py admin --arg rawPath=/api/status")
        print("\nIf the REST API probe returned '{\"message\":\"Not Found\"}', the "
              "stage exists but has no deployment yet -- re-run:")
        print("  python infra/deploy.py --endpoint %s --lambda-endpoint "
              "http://floci:4566 --skip-build --migrate-action none"
              % args.endpoint)
        raise SystemExit(1)
    try:
        ssm.put_parameter(Name=f"/{args.app}/admin/url", Value=url,
                          Type="String", Overwrite=True)
    except botocore.exceptions.ClientError:
        pass
    _finish(url, args)
def _finish(url, args):
    print(f"\nadmin page: {url}")
    if args.open:
        import webbrowser
        webbrowser.open(url)
if __name__ == "__main__":
    main()