"""
Single source of truth for "should we still be doing work for this channel?".
Consulted by every job producer and consumer, because a channel can be
deactivated at any moment -- while a scan message is on the wire, while a
download is mid-flight, while an ingest message sits in the queue.
"""
import time
from common.db import get_conn
_TTL = 30                 # seconds; a deactivation takes effect within one TTL
_cache = {}
def is_active(channel_id) -> bool:
    """False for inactive channels AND for channels that don't exist at all
    (an orphaned job is not work we want to do)."""
    if not channel_id:
        return False
    now = time.time()
    hit = _cache.get(channel_id)
    if hit and now - hit[1] < _TTL:
        return hit[0]
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("SELECT active FROM channels WHERE channel_id = %s", (channel_id,))
        row = cur.fetchone()
    conn.rollback()
    active = bool(row and row[0])
    _cache[channel_id] = (active, now)
    return active
def cancel_job(video_id) -> bool:
    """Drop a single job row. Any SQS message still referencing it becomes a
    no-op, and a download holding a lease on it aborts at its next heartbeat."""
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("DELETE FROM ingest_jobs WHERE video_id = %s", (video_id,))
        n = cur.rowcount
    conn.commit()
    return n > 0
def cancel_channel_jobs(channel_id,
                        statuses=("pending", "downloading", "downloaded")):
    """Cancel a channel's outstanding work. 'ingesting' is excluded by default:
    those abort themselves via the guard in handlers/ingest.py before writing,
    so there is no half-written month to clean up."""
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("""DELETE FROM ingest_jobs
                       WHERE channel_id = %s AND status = ANY(%s)
                       RETURNING video_id""", (channel_id, list(statuses)))
        rows = [r[0] for r in cur.fetchall()]
    conn.commit()
    return rows