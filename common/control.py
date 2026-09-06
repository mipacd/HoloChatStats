"""Shared pause gate. The admin page flips service_config.paused; every worker
checks it and politely puts its message back with a delay instead of dropping
or DLQ-ing it."""
import os
from common.aws import client
from common.config import settings
def paused() -> bool:
    try:
        return str(settings(force=True).get("paused", "false")).lower() == "true"
    except Exception:
        return False          # never block the pipeline on a config read failure
def requeue(queue_url: str, body: str, delay: int = 120):
    client("sqs").send_message(QueueUrl=queue_url, MessageBody=body,
                               DelaySeconds=min(900, max(1, delay)))
def requeue_all(env_var: str, records, delay: int = 120):
    url = os.environ[env_var]
    for r in records:
        requeue(url, r["body"], delay)