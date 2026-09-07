"""
Apply migrations/*.sql straight against Postgres, outside Lambda.
Why not the migrate Lambda? Because on a freshly restored database the first
migration is not a few milliseconds of DDL -- it is ADD PRIMARY KEY over every
restored table, ~15 btree builds, and four materialized views over user_data.
That is tens of minutes of work and Lambda's hard ceiling is 900 s. The migrate
Lambda remains correct (and fast) for steady-state deploys, where the pending
DDL is small.
Transport is identical to dbrestore: a throwaway postgres client container on
the compose network, or local psql. No new host dependency.
"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
MIGRATIONS_DIR = ROOT / "migrations"
ADVISORY_LOCK_KEY = 744_211_987          # MUST equal handlers/migrate.py
BOOTSTRAP_SQL = (
    "CREATE TABLE IF NOT EXISTS schema_migrations ("
    "  filename TEXT PRIMARY KEY,"
    "  applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW())"
)
# Index and matview builds ARE the cost here. maintenance_work_mem=64MB (the
# default) turns a 3-minute index build into a 30-minute external merge sort.
PGOPTIONS = " ".join([
    "-c statement_timeout=0",
    "-c idle_in_transaction_session_timeout=0",
    "-c lock_timeout=60s",
    "-c maintenance_work_mem=1GB",
    "-c max_parallel_maintenance_workers=4",
    "-c synchronous_commit=off",
])

TUNE_STATEMENTS = [
    "ALTER SYSTEM SET fsync = off",
    "ALTER SYSTEM SET full_page_writes = off",
    "ALTER SYSTEM SET synchronous_commit = off",
    "ALTER SYSTEM SET max_wal_size = '8GB'",
    "ALTER SYSTEM SET checkpoint_timeout = '15min'",
    "ALTER SYSTEM SET maintenance_work_mem = '1GB'",
    "ALTER SYSTEM SET max_parallel_maintenance_workers = 4",
    "ALTER SYSTEM SET max_parallel_workers_per_gather = 4",
    "SELECT pg_reload_conf()",
]


class _Psql:
    def __init__(self, *, host, port, dbname, user, password,
                 mode="docker", network=None, image="pgvector/pgvector:pg16"):
        if mode not in ("docker", "local"):
            raise ValueError(f"bad mode {mode!r}")
        self.mode, self.network, self.image = mode, network, image
        self.conn = ["-h", host, "-p", str(port), "-U", user, "-d", dbname]
        self.password = password
        self.mount = "/migrations" if mode == "docker" else MIGRATIONS_DIR.as_posix()
    def _cmd(self, extra):
        base = ["-X", "-q", "-v", "ON_ERROR_STOP=1", *self.conn, *extra]
        if self.mode == "docker":
            if not shutil.which("docker"):
                sys.exit("schema-mode=docker but docker is not on PATH")
            return ([
                "docker", "run", "--rm",
                *(["--network", self.network] if self.network else []),
                "-e", f"PGPASSWORD={self.password}",
                "-e", f"PGOPTIONS={PGOPTIONS}",
                "-v", f"{MIGRATIONS_DIR.resolve().as_posix()}:/migrations:ro",
                self.image, "psql", *base,
            ], None)
        if not shutil.which("psql"):
            sys.exit("schema-mode=local but psql is not on PATH "
                     "(install postgresql-client, or use --schema-mode docker)")
        return (["psql", *base],
                {**os.environ, "PGPASSWORD": self.password, "PGOPTIONS": PGOPTIONS})
    def run(self, extra):
        """Streams psql's stdout/stderr, so long index builds show NOTICEs live."""
        cmd, env = self._cmd(extra)
        proc = subprocess.Popen(cmd, env=env)
        elapsed = 0
        while True:
            try:
                proc.wait(timeout=60)
                break
            except subprocess.TimeoutExpired:
                elapsed += 60
                print(f"    schema operation still active ({elapsed}s, "
                      f"pid={proc.pid})", flush=True)
        if proc.returncode != 0:
            sys.exit(f"psql failed (exit {proc.returncode}): {' '.join(extra)}")
    def rows(self, sql):
        cmd, env = self._cmd(["-t", "-A", "-c", sql])
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            sys.exit(f"psql query failed:\n{proc.stderr}")
        return [ln for ln in proc.stdout.splitlines() if ln.strip()]
def analyze(**kw):
    """pg_restore leaves zero statistics and reltuples = -1. Every plan the
    migration makes is a guess until this runs."""
    p = _Psql(**kw)
    print("  ANALYZE (post-restore statistics) ...")
    t0 = time.time()
    p.run(["-c", "ANALYZE"])
    print(f"  ANALYZE done in {time.time() - t0:.0f}s")
def apply(**kw):
    p = _Psql(**kw)
    p.run(["-c", BOOTSTRAP_SQL])
    done = set(p.rows("SELECT filename FROM schema_migrations"))
    pending = [f for f in sorted(MIGRATIONS_DIR.glob("*.sql"))
               if f.name not in done]
    if not pending:
        print("schema migrations: nothing pending")
        return {"applied": [], "already_applied": sorted(done)}
    print(f"schema migrations: {len(pending)} pending "
          f"({', '.join(f.name for f in pending)})")
    for f in pending:
        print(f"  applying {f.name} ...")
        t0 = time.time()
        # One transaction per file, exactly like the Lambda. pg_advisory_xact_lock
        # (not pg_advisory_lock) releases on commit -- no lock can outlive the
        # process, which is the failure the Lambda path keeps hitting.
        p.run([
            "--single-transaction",
            "-c", f"SELECT pg_advisory_xact_lock({ADVISORY_LOCK_KEY})",
            "-f", f"{p.mount}/{f.name}",
            "-c", f"INSERT INTO schema_migrations (filename) VALUES ('{f.name}') "
                  f"ON CONFLICT DO NOTHING",
        ])
        print(f"  {f.name} ok ({time.time() - t0:.0f}s)")
    return {"applied": [f.name for f in pending], "already_applied": sorted(done)}
def populate_matviews(**kw):
    """
    003 creates the matviews WITH NO DATA, so the migration transaction stays
    cheap. Populate them here, outside any transaction. relispopulated makes
    this idempotent and self-healing: it only touches views that were never
    built (or were rebuilt by a re-run of 003).
    """
    p = _Psql(**kw)
    names = p.rows(
        "SELECT quote_ident(n.nspname)||'.'||quote_ident(c.relname) "
        "FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
        "WHERE c.relkind = 'm' AND NOT c.relispopulated ORDER BY 1")
    if not names:
        print("materialized views: all populated")
        return
    for name in names:
        print(f"  populating {name} (non-concurrent, first build) ...")
        t0 = time.time()
        p.run(["-c", f"REFRESH MATERIALIZED VIEW {name}"])
        print(f"  {name} populated ({time.time() - t0:.0f}s)")

def tune(**kw):
    p = _Psql(**kw)
    print("  applying emulator restore tuning (ALTER SYSTEM) ...")
    args = []
    for stmt in TUNE_STATEMENTS:
        args += ["-c", stmt]
    p.run(args)
