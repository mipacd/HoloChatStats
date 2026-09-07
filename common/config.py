"""
Static stack definition: names, sizes, wiring.  No AWS calls, no side effects --
import this from anywhere.
"""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
BUILD_DIR = ROOT / "build"
ZIP_PATH = ROOT / "build.zip"
APP = "chat-ingest"
PYTHON_RUNTIME = "python3.12"
RDS_ENGINE_VERSION = "pg16"
POSTGRES_CLIENT_IMAGE = "pgvector/pgvector:pg16"
# A PG18 client is required to read the archive produced by the legacy PG18
# server. dbrestore renders it to SQL and removes PG18-only session settings
# before sending it to the PG16 target.
LEGACY_POSTGRES_CLIENT_IMAGE = "pgvector/pgvector:pg18"
METRIC_NAMESPACE = "ChatIngestion"
DB_SECRET_ID = f"{APP}/db"
YT_SECRET_ID = f"{APP}/youtube"
LLM_SECRET_ID = f"{APP}/llm"
WEB_SECRET_ID = f"{APP}/web"
# Lambda code larger than this is staged through S3 instead of inline.
DIRECT_ZIP_LIMIT = 45_000_000
# floci publishes guest container ports on a random host port in this range.
PORT_SCAN_RANGE = (30000, 31000)
WILDCARD_POLICY = {"Version": "2012-10-17",
                   "Statement": [{"Effect": "Allow", "Action": "*", "Resource": "*"}]}
# ---------------------------------------------------------------- storage ----
BUCKETS = {
    "raw":      f"{APP}-raw-chat",
    "config":   f"{APP}-config",
    "frontend": f"{APP}-frontend",     # only used in bootstrap image mode
}
RAW_RETENTION_DAYS = 90
# ----------------------------------------------------------------- queues ----
QUEUES = {
    "scan-q":           {"visibility": 1000, "dlq": "scan-dlq",     "max_receive": 3},
    "download-q":       {"visibility": 5400, "dlq": "download-dlq", "max_receive": 3},
    "download-retry-q": {"visibility": 5400, "dlq": "download-dlq", "max_receive": 5},
    "ingest-q":         {"visibility": 5400, "dlq": "ingest-dlq",   "max_receive": 3},
    "scan-dlq":         {"visibility": 300},
    "download-dlq":     {"visibility": 300},
    "ingest-dlq":       {"visibility": 300},
}
# handler module, timeout(s), memory(MB), reserved concurrency (None = unmanaged)
FUNCTIONS = {
    "migrate":  {"handler": "handlers.migrate.handler",  "timeout": 900, "memory": 512,  "rc": None},
    "discover": {"handler": "handlers.discover.handler", "timeout": 120, "memory": 256,  "rc": None},
    "scan":     {"handler": "handlers.scan.handler",     "timeout": 900, "memory": 512,  "rc": 3},
    "download": {"handler": "handlers.download.handler", "timeout": 900, "memory": 1024, "rc": 1},
    "ingest":   {"handler": "handlers.ingest.handler",   "timeout": 900, "memory": 1024, "rc": 1},
    "refresh":  {"handler": "handlers.refresh.handler",  "timeout": 900, "memory": 512,  "rc": 1},
    "merge":    {"handler": "handlers.merge.handler",    "timeout": 900, "memory": 512,  "rc": 1},
    "status":   {"handler": "handlers.status.handler",   "timeout": 30,  "memory": 256,  "rc": None},
    "admin":    {"handler": "handlers.admin.handler",    "timeout": 60,  "memory": 512,  "rc": None},
    "reap":     {"handler": "handlers.reap.handler",     "timeout": 120, "memory": 256,  "rc": None},
}
# queue -> (function, batch_size, max_concurrency)   max_concurrency min is 2 on AWS
EVENT_SOURCE_MAPPINGS = {
    "scan-q":           ("scan",     1, 3),
    "download-q":       ("download", 1, 2),  # AWS ESM minimum; function rc=1
    "ingest-q":         ("ingest",   1, 2),  # AWS minimum; function rc=1
}
SCHEDULES = {
    "discover": "rate(2 hours)",
    "reap":     "rate(5 minutes)",
    "refresh":  "cron(30 9 * * ? *)",
    "merge":    "cron(15 10 * * ? *)",
}
# ------------------------------------------------------------- cache / db ----
ELASTICACHE_GROUP_ID = f"{APP}-redis"
# ------------------------------------------------- web API server (EC2) ------
WEB_PORT = 8080
WEB_AMI = "ami-amazonlinux2023"            # floci catalog: systemd guest runtime
WEB_PYTHON = "python3.12"
WEB_INSTANCE_NAME = f"{APP}-web"
WEB_VENV_KEY = "web/web-venv.tar.gz"       # large, rebuilt only on dep changes
WEB_CODE_KEY = "web/web-code.tar.gz"       # small, rebuilt every deploy
# LLM chatbot (FastAPI) -- same instance, same venv, second port
LLM_PORT = 8000
LLM_HEALTH_PATH = "/healthz/llm"
LLM_CODE_KEY = "web/llm-code.tar.gz"
LLM_DIR_NAME = "llm_chat"
# ---------------------------------------------------- frontend (ECS) --------
ECS_CLUSTER = f"{APP}-cluster"
FRONTEND_NAME = f"{APP}-frontend"          # service + task family + TG + ALB name
FRONTEND_CONTAINER = "web"
FRONTEND_PORT = 80
FRONTEND_HOST_PORT = 80
FRONTEND_HEALTH_PATH = "/healthz"
FRONTEND_ECR_REPO = f"{APP}/frontend"
FRONTEND_BASE_IMAGE = "nginx:1.27-alpine"
FRONTEND_SITE_KEY = "frontend/site.tar.gz"
FRONTEND_LOG_GROUP = f"/ecs/{FRONTEND_NAME}"
ECS_TASK_EXEC_POLICY = ("arn:aws:iam::aws:policy/service-role/"
                        "AmazonECSTaskExecutionRolePolicy")
DOCKER_HOST_ALIAS = "host.docker.internal"
