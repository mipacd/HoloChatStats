from psycopg2 import pool
from config.settings import get_config
from utils.logging_utils import get_logger
import psycopg2

logger = get_logger()
DB_POOL = None

def init_db_pool():
    global DB_POOL
    if DB_POOL is None:
        db_config = {
            "dbname": get_config("Database", "DBName"),
            "user": get_config("Database", "DBUser"),
            "password": get_config("Database", "DBPass"),
            "host": get_config("Database", "DBHost"),
            "port": get_config("Database", "DBPort"),
            "client_encoding": "UTF8",
            'sslmode': 'disable'
        }
        DB_POOL = pool.SimpleConnectionPool(1, 10, **db_config)



# Connect to database
def get_db_connection():
    global DB_POOL
    if DB_POOL is None:
        init_db_pool()
    conn = DB_POOL.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")  # ✅ Test if connection is alive
        return conn
    except psycopg2.OperationalError:
        logger.warning("Stale DB connection detected. Reconnecting...")
        DB_POOL.putconn(conn, close=True)  # ✅ Close broken connection
        conn = DB_POOL.getconn()  # ✅ Get a fresh connection
        return conn

def release_db_connection(conn):

    if conn is not None:
        try:
            conn.rollback()  # ✅ Rollback any uncommitted transactions
            DB_POOL.putconn(conn)
        except psycopg2.InterfaceError:
            logger.warning("⚠️ Trying to release a closed connection. Removing from pool...")
            DB_POOL.putconn(conn, close=True)  # ✅ Remove bad connection from pool