"""
Idempotent provisioning for the chat-ingestion stack.
Targets the floci emulator by default; works against real AWS if you drop the
endpoint flags (you would then want scoped IAM policies instead of the wide-open
role created here, and --frontend-image-mode ecr --frontend-alb).
Layout:
    deploylib/config.py    names, sizes, wiring
    deploylib/awsutil.py   lazy boto3 clients, retries, probes
    deploylib/*.py         one provisioning concern per module
    deploylib/stack.py     Stack = composition of those mixins
Usage (typical local loop):
    python infra/deploy.py \
        --endpoint http://localhost:4566 \
        --lambda-endpoint http://floci:4566 \
        --db-host postgres --db-password ingest \
        --channels-file ./channels.json \
        --youtube-api-key $YT_KEY
    # later, code-only redeploys (lambdas + web API + frontend):
    python infra/deploy.py --endpoint http://localhost:4566 \
        --lambda-endpoint http://floci:4566 --code-only
"""
import json
import sys
import dbmigrate
from deploylib import Stack
from deploylib import config as C
from deploylib.cli import parse_args
def apply_schema(stack, args):
    """Schema work in whichever mode the caller asked for."""
    if args.schema_mode == "lambda":
        stack.run_migrate("migrate")
        stack.enforce_backlog_floor()
        return
    target = stack.sql_target()
    dbmigrate.apply(**target)
    if not args.skip_view_refresh:
        dbmigrate.populate_matviews(**target)
def deploy_code(stack, args):
    """--code-only: push new code everywhere, no infra changes."""
    # Hydrate endpoints and rotate explicitly supplied secrets before building
    # Lambda environments. A bootstrap marker must not make secret rotation a
    # no-op or leave queue URLs empty.
    check_db_config(stack, args)
    stack.ensure_secrets()
    stack.ensure_news_seed()
    stack.ensure_raw_lifecycle()
    stack.ensure_queues()
    stack.ensure_elasticache()
    # Reapply function configuration as well as code so concurrency and newly
    # added environment variables take effect on ordinary pushes.
    stack.ensure_functions(stack.ensure_lambda_role())
    if not args.skip_web:
        stack.ensure_web_secret()
        stack.sync_llm_secret()
        stack.upload_web_code()
        stack.ensure_web_instance()
    if not args.skip_frontend and stack.build_frontend():
        stack.ensure_frontend_service()
    apply_schema(stack, args)
    stack.run_migrate("drain_retry_queue")
    stack.ensure_esms()
    admin = stack.get_param(f"/{C.APP}/admin/url")     # resolved on full deploy
    if admin:
        stack.summary.append(("Admin page", admin))
    stack.print_summary()
def check_db_config(stack, args):
    if args.create_rds:
        if not args.db_password:
            sys.exit("--create-rds requires --db-password")
        stack.ensure_rds()
    elif not args.db_host:
        try:
            cur = json.loads(stack.sm.get_secret_value(
                SecretId=C.DB_SECRET_ID)["SecretString"])
            print(f"WARNING: reusing existing DB secret -> {cur['username']}@"
                  f"{cur['host']}:{cur['port']}/{cur['dbname']}")
        except stack.sm.exceptions.ResourceNotFoundException:
            sys.exit("no DB secret exists and neither --db-host nor --create-rds "
                     "was given -- one of them is required on first deploy")
def provision(stack, args):
    stack.ensure_buckets()
    check_db_config(stack, args)
    stack.ensure_secrets()
    stack.sync_llm_secret()
    stack.ensure_queues()
    stack.ensure_elasticache()
    stack.ensure_functions(stack.ensure_lambda_role())
    stack.tune_database()
    # Restore BEFORE the event source mappings exist: no consumer can pick up a
    # restored `pending` job while the database is mid-rewrite.
    stack.maybe_restore()
    # Heavy DDL (primary keys, 15 indexes, 4 matviews) runs against Postgres
    # directly, so there is no 900 s ceiling to hit.
    stack.run_schema_migrations()
    stack.run_migrate("drain_retry_queue")
    # Cheap, but they need S3 + the Lambda env, so they stay in Lambda -- as
    # separate invocations, not bundled into {"action": "all"}.
    if args.migrate_action in ("seed_channels", "all"):
        stack.run_migrate("seed_channels")
    if args.migrate_action in ("backfill_jobs", "all"):
        stack.run_migrate("backfill_jobs")
    stack.enforce_backlog_floor()
    stack.ensure_esms()
    stack.ensure_schedules()
    stack.ensure_status_api()
    http_api = stack.ensure_admin_api()
    rest_api = stack.ensure_admin_rest_api()
    # The frontend nginx template needs the resolved REST API path for its
    # LAN-only /admin reverse proxy.
    stack.resolve_admin_url(rest_api, http_api)
    # API server first: the frontend task needs its internal address.
    if not args.skip_web:
        stack.build_web_bundle()
        stack.ensure_web_instance()
    if not args.skip_frontend and stack.build_frontend():
        stack.ensure_frontend_service()
    stack.mark_bootstrapped()
    # Begin the first complete discovery sweep. Downloads remain gated by the
    # dispatcher until every scan message has drained.
    stack.lam.invoke(FunctionName=f"{C.APP}-discover", InvocationType="Event",
                     Payload=b'{"force":true}')
def main():
    args = parse_args(__doc__)
    stack = Stack(args)
    if args.auto and not (args.full or args.force_restore):
        marker = stack.bootstrap_marker()
        if marker:
            print(f"bootstrap already complete ({marker}); code-only update")
            args.code_only = True
        else:
            print("no bootstrap marker: full provision + one-time restore")
    if not args.skip_build:
        stack.build_zip()
    if args.code_only:
        deploy_code(stack, args)
        return
    provision(stack, args)
    stack.print_summary()
    print("\ndeploy complete.")
    print("the ETL discovery sweep has been started at the configured UTC floor")
if __name__ == "__main__":
    main()
