import os, psycopg2, psycopg2.extras
from common.config import secret
_conn = None
def get_conn():
    """One connection per warm container. Lambda is single-threaded per invoke,
    so a pool buys nothing; reuse across invocations is what matters."""
    global _conn
    if _conn is not None and _conn.closed == 0:
        try:
            with _conn.cursor() as c:
                c.execute("SELECT 1")
            return _conn
        except psycopg2.Error:
            try: _conn.close()
            except Exception: pass
            _conn = None
    creds = secret(os.environ["DB_SECRET_ID"])
    _conn = psycopg2.connect(
        host=creds["host"], port=creds.get("port", 5432),
        dbname=creds["dbname"], user=creds["username"], password=creds["password"],
        connect_timeout=5, application_name=os.environ.get("AWS_LAMBDA_FUNCTION_NAME", "local"),
        options="-c statement_timeout=120000",
    )
    _conn.autocommit = False
    return _conn