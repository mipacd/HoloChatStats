"""Command line surface for infra/deploy.py."""
import argparse
import os
import sys
from . import config as C
def parse_args(doc=None):
    p = argparse.ArgumentParser(
        description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    # ---- endpoints -------------------------------------------------------
    p.add_argument("--endpoint", help="emulator edge URL as seen from this "
                                      "shell, e.g. http://localhost:4566 "
                                      "(omit for real AWS)")
    p.add_argument("--lambda-endpoint", help="emulator edge URL as seen from "
                                             "inside spawned containers, e.g. "
                                             "http://floci:4566")
    p.add_argument("--region", default="us-east-1")
    # ---- config / secrets ------------------------------------------------
    p.add_argument("--channels-file", help="local channels.json to upload")
    p.add_argument("--youtube-api-key", default=os.environ.get("YT_API_KEY"))
    p.add_argument("--llm-secrets-json", default=os.environ.get("LLM_SECRETS_JSON"),
                   help="JSON object for the LLM secret (prefer a CI secret; "
                        "never commit a .env file)")
    p.add_argument("--update-secrets", action="store_true",
                   help="overwrite existing secret values with CLI-supplied ones")
    # ---- database --------------------------------------------------------
    p.add_argument("--db-host", help="Postgres host (skip if using --create-rds)")
    p.add_argument("--db-port", type=int, default=5432)
    p.add_argument("--db-name", default="youtube_data")
    p.add_argument("--db-user", default="ingest")
    p.add_argument("--db-password", default=os.environ.get("DB_PASSWORD"))
    p.add_argument("--create-rds", action="store_true",
                   help="create/find an RDS Postgres instance and point the DB "
                        "secret at it")
    p.add_argument("--db-internal-host",
                   help="hostname the lambdas/ECS use to reach postgres "
                        "(default: whatever describe-db-instances reports)")
    p.add_argument("--db-external-host", default="localhost",
                   help="hostname YOU use for psql/pg_restore")
    # ---- build -----------------------------------------------------------
    p.add_argument("--code-only", action="store_true",
                   help="rebuild + push code, skip infra provisioning")
    p.add_argument("--skip-build", action="store_true")
    p.add_argument("--target-platform",
                   help="wheel platform for the lambda runtime, e.g. "
                        "manylinux2014_x86_64")
    p.add_argument("--python-version", default="3.12")
    p.add_argument("--node-version", default="20")
    p.add_argument("--build-in-docker", action="store_true", default=None,
                   help="run pip/npm inside a linux container (REQUIRED on "
                        "Windows and Apple Silicon; default on when the host "
                        "is not linux)")
    p.add_argument("--no-build-in-docker", dest="build_in_docker",
                   action="store_false")
    # ---- migrations ------------------------------------------------------
    p.add_argument("--migrate-action", default="all",
                   choices=["migrate", "seed_channels", "backfill_jobs",
                            "all", "none"])
    p.add_argument("--schema-mode", choices=["docker", "local", "lambda"],
                   default="docker",
                   help="how to apply migrations/*.sql. 'lambda' is only safe "
                        "on a small database -- the first migration after a "
                        "restore cannot finish inside Lambda's 900 s ceiling")
    p.add_argument("--schema-image")    # defaults to --restore-image
    p.add_argument("--schema-network")  # defaults to --restore-network
    p.add_argument("--skip-analyze", action="store_true")
    p.add_argument("--skip-view-refresh", action="store_true")
    p.add_argument("--backlog-floor",
                   help="YYYY-MM-DD; months before this are never fetched")
    # ---- one-time restore ------------------------------------------------
    p.add_argument("--dump-url", default=os.environ.get("DUMP_URL"),
                   help="HTTP(S) URL of a pg_dump -F c backup of youtube_data")
    p.add_argument("--dump-sha256", default=os.environ.get("DUMP_SHA256"))
    p.add_argument("--dump-header", action="append", default=[],
                   help='extra request header, e.g. "Authorization: Bearer x"')
    p.add_argument("--dump-cache-dir", default=str(C.ROOT / ".cache" / "dumps"))
    p.add_argument("--dump-has-create", action="store_true",
                   help="dump was taken with pg_dump --create")
    p.add_argument("--restore-mode", choices=["docker", "local"],
                   default="docker")
    p.add_argument("--restore-network", default="chat-ingest_default")
    p.add_argument("--restore-image", default=C.POSTGRES_CLIENT_IMAGE)
    p.add_argument("--restore-jobs", type=int, default=4,
                   help="pg_restore -j; 1 means --single-transaction")
    p.add_argument("--force-restore", action="store_true")
    p.add_argument("--skip-restore", action="store_true")
    p.add_argument("--keep-migration-log", action="store_true")
    p.add_argument("--stream-restore", action="store_true",
                   help="stream --dump-url into pg_restore without storing it "
                        "on the deployment host")
    p.add_argument("--source-db-container",
                   default=os.environ.get("SOURCE_DB_CONTAINER"),
                   help="one-time direct source container, e.g. hcs-postgres")
    p.add_argument("--source-db-name",
                   default=os.environ.get("SOURCE_DB_NAME"))
    p.add_argument("--source-db-user",
                   default=os.environ.get("SOURCE_DB_USER"))
    p.add_argument("--source-db-password",
                   default=os.environ.get("SOURCE_DB_PASSWORD"))
    p.add_argument("--source-db-image", default=C.LEGACY_POSTGRES_CLIENT_IMAGE,
                   help="pg_restore client image; must understand the source "
                        "server's archive format")
    # ---- CI entry point --------------------------------------------------
    p.add_argument("--auto", action="store_true",
                   help="full provision + restore on first run; code-only after")
    p.add_argument("--full", action="store_true",
                   help="ignore the bootstrap marker and provision everything")
    # ---- web API (EC2) ---------------------------------------------------
    p.add_argument("--skip-web", action="store_true")
    p.add_argument("--web-port", type=int, default=C.WEB_PORT)
    p.add_argument("--web-workers", type=int, default=2)
    p.add_argument("--rebuild-web-deps", action="store_true",
                   help="force-rebuild the web venv even if it is cached in S3")
    # ---- frontend (ECS) --------------------------------------------------
    p.add_argument("--skip-frontend", action="store_true")
    p.add_argument("--frontend-dir", default=str(C.ROOT / "frontend"))
    p.add_argument("--frontend-image-mode", default="auto",
                   choices=["auto", "ecr", "bootstrap"],
                   help="auto: try ECR, fall back to the public nginx image "
                        "pulling the build from S3 at container start")
    p.add_argument("--frontend-tag",
                   help="explicit image tag (default: unix timestamp)")
    p.add_argument("--frontend-launch-type", default="FARGATE",
                   choices=["FARGATE", "EC2", "fargate", "ec2"])
    p.add_argument("--frontend-count", type=int, default=1)
    p.add_argument("--frontend-cpu", default="256")
    p.add_argument("--frontend-memory", default="512")
    p.add_argument("--frontend-public-ip", action="store_true", default=True,
                   help="assignPublicIp=ENABLED for awsvpc tasks (default)")
    p.add_argument("--frontend-private", dest="frontend_public_ip",
                   action="store_false",
                   help="keep the task private (use with --frontend-alb)")
    p.add_argument("--frontend-alb", action="store_true",
                   help="front the service with an ALB (recommended on real AWS)")
    p.add_argument("--frontend-api-base", default="/api",
                   help="value injected as VITE_API_BASE_URL at build time")
    p.add_argument("--frontend-api-backend",
                   help="override the API backend as host:port (default: "
                        f"/{C.APP}/web/internal_url from SSM)")
    p.add_argument("--frontend-strip-api-prefix", action="store_true",
                   help="strip /api before proxying (use when the EC2 API "
                        "serves its routes at / instead of /api)")
    p.add_argument("--frontend-port", type=int, default=C.FRONTEND_PORT,
                   help="port nginx listens on inside the container")
    p.add_argument("--frontend-host-port", type=int, default=C.FRONTEND_HOST_PORT,
                   help="host port the frontend is published on (use something "
                        "other than 80 if the host port is already taken)")
    p.add_argument("--ecr-push-host",
                   help="override the registry host:port docker pushes to "
                        "(floci: usually localhost:5100)")

    p.add_argument("--skip-llm", action="store_true",
                   help="do not deploy the LLM chat server onto the web instance")
    p.add_argument("--llm-port", type=int, default=C.LLM_PORT)
    p.add_argument("--llm-dir", default=str(C.ROOT / C.LLM_DIR_NAME))
    p.add_argument("--frontend-llm-base", default="/llm",
                   help="value injected as VITE_ERI_API_URL at build time")
    
    args = p.parse_args()
    args.dump_sha256 = args.dump_sha256 or None    # CI passes "" for unset
    args.frontend_launch_type = args.frontend_launch_type.upper()
    if args.build_in_docker is None:
        args.build_in_docker = not sys.platform.startswith("linux")
        if args.build_in_docker:
            print(f"host is {sys.platform}: building in docker "
                  f"(use --no-build-in-docker to override)")
    return args
