from datetime import datetime, timezone
def backlog_floor(conn, cfg=None):
    """service_config.backlog_floor wins. Only when it is unset do we infer
    from merge state / ingested data -- and the inference is deliberately
    conservative, so it can never silently widen the fetch window."""
    v = None
    if cfg:
        v = (cfg.get("backlog_floor") or "").strip() or None
    if v is None:
        with conn.cursor() as cur:
            cur.execute("SELECT value FROM service_config WHERE key='backlog_floor'")
            row = cur.fetchone()
        conn.rollback()
        v = (row[0].strip() if row and row[0] else None)
    if v:
        d = datetime.fromisoformat(v)
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    with conn.cursor() as cur:
        cur.execute("""SELECT COALESCE(
            (SELECT (MAX(observed_month) + INTERVAL '1 month')::date
               FROM monthly_merge_state WHERE status='merged'),
            (SELECT date_trunc('month', MAX(v.end_time))::date
               FROM videos v JOIN ingest_jobs j USING (video_id)
              WHERE j.status='done'))""")
        row = cur.fetchone()[0]
    conn.rollback()
    return (datetime.combine(row, datetime.min.time(), tzinfo=timezone.utc)
            if row else None)