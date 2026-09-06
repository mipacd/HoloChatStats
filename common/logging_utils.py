import json
import logging
import os
_RESERVED = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "taskName", "message", "asctime",
}
class JsonFormatter(logging.Formatter):
    def format(self, record):
        doc = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "function": os.environ.get("AWS_LAMBDA_FUNCTION_NAME", "local"),
        }
        rid = getattr(record, "aws_request_id", None)
        if rid:
            doc["request_id"] = rid
        # Anything passed via logger.info(..., extra={...}) lands at top level
        # so CloudWatch Logs Insights can query on it directly.
        for k, v in record.__dict__.items():
            if k not in _RESERVED and not k.startswith("_"):
                doc[k] = v
        if record.exc_info:
            doc["exception"] = self.formatException(record.exc_info)
        return json.dumps(doc, default=str)
def get_logger(name="ingest"):
    logger = logging.getLogger(name)
    if not getattr(logger, "_configured", False):
        logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))
        # Lambda installs its own root handler; replace its formatter rather
        # than adding a second handler (which would double every line).
        root = logging.getLogger()
        if root.handlers:
            for h in root.handlers:
                h.setFormatter(JsonFormatter())
        else:
            h = logging.StreamHandler()
            h.setFormatter(JsonFormatter())
            logger.addHandler(h)
            logger.propagate = False
        logger._configured = True
    return logger