import json, os
from common.aws import client
from common.db import get_conn
from common.config import settings
def handler(event, context):
    conn, sqs = get_conn(), client("sqs")
    out = {}
    with conn.cursor() as cur:
        cur.execute("SELECT status, count(*) FROM ingest_jobs GROUP BY status")
        out["jobs"] = dict(cur.fetchall())
        cur.execute("""SELECT video_id, channel_id, status, attempts,
                              last_offset_s, video_duration_s, last_error,
                              EXTRACT(EPOCH FROM NOW()-updated_at)::int AS stale_s
                       FROM ingest_jobs
                       WHERE status IN ('downloading','ingesting')
                         AND updated_at < NOW() - INTERVAL '30 minutes'
                       ORDER BY updated_at LIMIT 50""")
        cols = [d[0] for d in cur.description]
        out["stuck"] = [dict(zip(cols, r)) for r in cur.fetchall()]
        cur.execute("""SELECT video_id, channel_id, attempts, last_error, updated_at
                       FROM ingest_jobs WHERE status='failed'
                       ORDER BY updated_at DESC LIMIT 50""")
        cols = [d[0] for d in cur.description]
        out["failed"] = [dict(zip(cols, r), updated_at=str(r[-1])) for r in cur.fetchall()]
        cur.execute("""SELECT channel_id, last_scanned_at, last_seen_end_time, last_error
                       FROM channel_watermarks ORDER BY last_scanned_at NULLS FIRST""")
        out["channels"] = [{"channel_id": r[0], "last_scanned_at": str(r[1]),
                            "last_seen_end_time": str(r[2]), "last_error": r[3]}
                           for r in cur.fetchall()]
        cur.execute("""SELECT observed_month, COUNT(*) AS rows,
                              COUNT(DISTINCT video_id) AS videos
                       FROM user_data_current
                       GROUP BY observed_month ORDER BY observed_month""")
        out["staging"] = [{"month": str(r[0]), "rows": r[1], "videos": r[2]}
                          for r in cur.fetchall()]
        cur.execute("""SELECT observed_month, status, rows_merged, merged_at
                       FROM monthly_merge_state
                       ORDER BY observed_month DESC LIMIT 6""")
        out["merges"] = [{"month": str(r[0]), "status": r[1], "rows": r[2],
                          "merged_at": str(r[3])} for r in cur.fetchall()]
    for name, url in (("download", os.environ["DOWNLOAD_QUEUE_URL"]),
                      ("ingest", os.environ["INGEST_QUEUE_URL"]),
                      ("download_dlq", os.environ["DOWNLOAD_DLQ_URL"])):
        attrs = sqs.get_queue_attributes(
            QueueUrl=url, AttributeNames=["ApproximateNumberOfMessages",
                                          "ApproximateNumberOfMessagesNotVisible"])["Attributes"]
        out.setdefault("queues", {})[name] = {
            "visible": int(attrs["ApproximateNumberOfMessages"]),
            "in_flight": int(attrs["ApproximateNumberOfMessagesNotVisible"])}
    out["config"] = settings(force=True)
    return {"statusCode": 200,
            "headers": {"content-type": "application/json"},
            "body": json.dumps(out, default=str)}