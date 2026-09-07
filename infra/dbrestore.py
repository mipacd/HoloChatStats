"""
Fetch a pg_dump custom-format (-F c) backup over HTTP and restore it.
The dump is downloaded by the *runner* (not inside the restore container) so the
checksum is verified once, the download is cacheable across re-runs, and the
container needs no network access to your NAS -- only to Postgres.
"""
import hashlib
import os
import shutil
import subprocess
import sys
import threading
import urllib.request
import re
import tempfile
from pathlib import Path
CHUNK = 1 << 20
RESTORE_RENDERER = "chat-ingest-restore-renderer"
RESTORE_LOADER = "chat-ingest-restore-loader"

# TOC entries we never want pg_restore to execute:
#  - MATERIALIZED VIEW DATA: the legacy MVs get dropped and rebuilt by
#    003_views.sql + populate_matviews(); refreshing them during restore
#    builds every MV twice over 40M rows.
#  - SCHEMA public / EXTENSION plpgsql (+ comments): present in every fresh
#    database; restoring them is the classic benign "already exists" pair
#    that makes --exit-on-error unusable.
_TOC_SKIP = re.compile(
    r"MATERIALIZED VIEW DATA"
    r"|\bSCHEMA\b.*\bpublic\b"
    r"|\bEXTENSION\b.*\bplpgsql\b"
    r"|\bCOMMENT\b.*\bEXTENSION\b.*\bplpgsql\b"
)


def fetch(url, cache_dir, sha256=None, headers=(), timeout=60):
    """Download (or reuse) the dump. Returns (path, hexdigest)."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    name = url.rstrip("/").rsplit("/", 1)[-1] or "database.dump"
    dest = cache / name
    if dest.exists() and sha256:
        have = _digest(dest)
        if have == sha256.lower():
            print(f"  cached dump matches sha256 -> {dest} "
                  f"({dest.stat().st_size / 1e6:.1f} MB)")
            return dest, have
        print(f"  cached dump sha256 mismatch ({have[:12]}...); re-downloading")
    req = urllib.request.Request(url)
    for h in headers:
        k, _, v = h.partition(":")
        req.add_header(k.strip(), v.strip())
    tmp = dest.with_suffix(dest.suffix + ".part")
    h = hashlib.sha256()
    print(f"  downloading {url}")
    with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            buf = r.read(CHUNK)
            if not buf:
                break
            f.write(buf)
            h.update(buf)
            done += len(buf)
            if total:
                pct = 100.0 * done / total
                print(f"\r    {done/1e6:8.1f} / {total/1e6:.1f} MB  {pct:5.1f}%",
                      end="", flush=True)
        print()
    digest = h.hexdigest()
    if sha256 and digest != sha256.lower():
        tmp.unlink(missing_ok=True)
        sys.exit(f"dump sha256 mismatch: expected {sha256.lower()}, got {digest}")
    tmp.replace(dest)
    print(f"  dump ready: {dest} ({dest.stat().st_size / 1e6:.1f} MB) "
          f"sha256={digest[:16]}...")
    return dest, digest
def _digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for buf in iter(lambda: f.read(CHUNK), b""):
            h.update(buf)
    return h.hexdigest()

def stream_restore(url, *, host, port, dbname, user, password, network=None,
                   image="pgvector/pgvector:pg18", sha256=None, headers=(),
                   timeout=60, reset_schema=False):
    """Stream a custom-format archive through pg_restore and into psql.

    This keeps the archive off the destination disk. Integrity is calculated
    while streaming; a mismatch fails the deployment before it is recorded as
    restored. A PG18 archive must be decoded by a PG18 pg_restore, but its SQL
    prologue contains ``SET transaction_timeout`` which a PG16 server cannot
    parse. Rendering first lets us discard exactly that compatibility setting
    while streaming all other SQL unchanged into the target.
    """
    if not shutil.which("docker"):
        sys.exit("stream restore requires Docker on the runner")
    if reset_schema:
        # Reuse the normal restore helper's reset implementation without
        # downloading or opening the archive.
        _reset_schema(host, port, dbname, user, password, network, image)
    req = urllib.request.Request(url)
    for raw in headers:
        key, sep, value = raw.partition(":")
        if not sep:
            sys.exit("invalid --dump-header; expected 'Name: value'")
        req.add_header(key.strip(), value.strip())
    env = {**os.environ, "PGPASSWORD": password}
    # A killed Actions runner can leave an attached `docker run --rm` process
    # alive. Deterministic names let this run and the workflow reap only our
    # own stale helpers before retrying.
    subprocess.run(["docker", "rm", "-f", RESTORE_RENDERER, RESTORE_LOADER],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    docker = ["docker", "run", "--rm", "-i",
              "--label", "com.holochatstats.role=db-restore"]
    if network:
        docker += ["--network", network]
    render_cmd = [*docker, "--name", RESTORE_RENDERER, image, "pg_restore",
                  "--clean", "--if-exists",
                  "--no-owner", "--no-privileges", "--verbose", "--file=-"]
    load_cmd = [*docker, "--name", RESTORE_LOADER, "-e", "PGPASSWORD",
                image, "psql", "-X",
                "--set", "ON_ERROR_STOP=1", "-h", host, "-p", str(port),
                "-U", user, "-d", dbname]
    print(f"  streaming remote dump into {host}:{port}/{dbname}")
    renderer = subprocess.Popen(render_cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, env=env)
    loader = subprocess.Popen(load_cmd, stdin=subprocess.PIPE, env=env)
    relay_errors = []
    filtered = {"transaction_timeout": 0, "materialized_view_refresh": 0}

    def relay_sql():
        try:
            for line in iter(renderer.stdout.readline, b""):
                if line.strip() == b"SET transaction_timeout = 0;":
                    filtered["transaction_timeout"] += 1
                    continue
                # Materialized-view contents from the legacy database are
                # derived data. Loading them here is extremely expensive and
                # migrations rebuild/populate the current definitions later.
                if line.lstrip().startswith(b"REFRESH MATERIALIZED VIEW "):
                    filtered["materialized_view_refresh"] += 1
                    continue
                loader.stdin.write(line)
        except (BrokenPipeError, OSError) as error:
            relay_errors.append(error)
            renderer.kill()
        finally:
            try:
                loader.stdin.close()
            except (BrokenPipeError, OSError):
                pass

    relay = threading.Thread(target=relay_sql, name="pg-restore-sql-relay",
                             daemon=True)
    relay.start()
    digest = hashlib.sha256()
    downloaded = 0
    transfer_error = None
    heartbeat_stop = threading.Event()

    def heartbeat():
        while not heartbeat_stop.wait(60):
            # Download progress can remain unchanged for a long time while
            # expanded COPY data drains through PostgreSQL. Keep the Actions
            # log alive and make that backpressure visible.
            print(f"    restore active: archive={downloaded / (1 << 30):.2f} "
                  f"GiB renderer={renderer.poll()} loader={loader.poll()}",
                  flush=True)

    heartbeat_thread = threading.Thread(target=heartbeat,
                                        name="pg-restore-heartbeat",
                                        daemon=True)
    heartbeat_thread.start()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            while True:
                chunk = response.read(CHUNK)
                if not chunk:
                    break
                digest.update(chunk)
                downloaded += len(chunk)
                renderer.stdin.write(chunk)
                if downloaded % (256 * CHUNK) < CHUNK:
                    print(f"    streamed {downloaded / (1 << 30):.1f} GiB",
                          flush=True)
    except (BrokenPipeError, OSError) as error:
        transfer_error = error
    finally:
        try:
            renderer.stdin.close()
        except (BrokenPipeError, OSError):
            pass
    renderer_rc = renderer.wait()
    relay.join()
    loader_rc = loader.wait()
    heartbeat_stop.set()
    heartbeat_thread.join()
    have = digest.hexdigest()
    if transfer_error or relay_errors or renderer_rc or loader_rc:
        _reset_schema(host, port, dbname, user, password, network, image)
        details = (f"renderer={renderer_rc}, loader={loader_rc}, "
                   f"transfer_error={transfer_error!r}, "
                   f"relay_error={relay_errors[0] if relay_errors else None!r}")
        sys.exit(f"streamed restore failed ({details}); partial target was "
                 "cleared; see pg_restore/psql output above")
    if sha256 and have != sha256.lower():
        _reset_schema(host, port, dbname, user, password, network, image)
        sys.exit(f"streamed dump sha256 mismatch: expected {sha256.lower()}, "
                 f"got {have}; partial target was cleared")
    print(f"  streamed restore completed ({downloaded / (1 << 30):.1f} GiB, "
          f"sha256={have[:16]}..., filtered "
          f"transaction_timeout={filtered['transaction_timeout']}, "
          f"materialized_view_refresh="
          f"{filtered['materialized_view_refresh']})")
    return have

def _reset_schema(host, port, dbname, user, password, network, image):
    env = {**os.environ, "PGPASSWORD": password}
    cmd = ["docker", "run", "--rm"]
    if network:
        cmd += ["--network", network]
    cmd += ["-e", "PGPASSWORD", image, "psql", "-X", "-v", "ON_ERROR_STOP=1",
            "-h", host, "-p", str(port), "-U", user, "-d", dbname,
            "-c", "DROP SCHEMA IF EXISTS public CASCADE",
            "-c", "CREATE SCHEMA public"]
    subprocess.run(cmd, env=env, check=True)
def _filter_toc(raw_toc: str):
    keep, skipped = [], []
    for line in raw_toc.splitlines():
        s = line.strip()
        if s and not s.startswith(";") and _TOC_SKIP.search(s):
            skipped.append(s)
            keep.append(";" + line)          # comment out, keep for audit
        else:
            keep.append(line)
    return "\n".join(keep) + "\n", skipped
def restore(dump_path, *, host, port, dbname, user, password,
            mode="docker", network=None, image="pgvector/pgvector:pg16",
            jobs=4, has_create=False, dry_run=False, reset_schema=False):
    dump_path = Path(dump_path)
    def run(tool_args, capture=False):
        if mode == "docker":
            cmd = ["docker", "run", "--rm",
                   *(["--network", network] if network else []),
                   "-e", f"PGPASSWORD={password}",
                   "-v", f"{dump_path.parent.resolve().as_posix()}:/dump",
                   image, *tool_args]
            env = None
        else:
            cmd, env = tool_args, {**os.environ, "PGPASSWORD": password}
        return subprocess.run(cmd, env=env, capture_output=capture, text=True)
    mnt = "/dump" if mode == "docker" else dump_path.parent.resolve().as_posix()
    # --- make retries deterministic -----------------------------------------
    # A restore that died halfway leaves a half-populated schema; restoring
    # on top of it produces hundreds of spurious "already exists" errors and
    # forces you to turn --exit-on-error off, which hides the real failures.
    if reset_schema:
        print("  resetting schema public before restore")
        r = run(["psql", "-v", "ON_ERROR_STOP=1", "-h", host, "-p", str(port),
                 "-U", user, "-d", dbname,
                 "-c", "DROP SCHEMA IF EXISTS public CASCADE",
                 "-c", "CREATE SCHEMA public"], capture=True)
        if r.returncode != 0:
            sys.exit(f"schema reset failed:\n{r.stderr}")
    # --- TOC filter -----------------------------------------------------------
    r = run(["pg_restore", "-l", f"{mnt}/{dump_path.name}"], capture=True)
    if r.returncode != 0:
        sys.exit(f"pg_restore -l failed (client older than the dump?):\n{r.stderr}")
    toc, skipped = _filter_toc(r.stdout)
    toc_path = dump_path.with_suffix(dump_path.suffix + ".toc")
    toc_path.write_text(toc, encoding="utf-8")
    print(f"  TOC: skipping {len(skipped)} entr(ies):")
    for s in skipped[:10]:
        print(f"    - {s}")
    target = (["-d", "postgres", "--create"] if has_create else ["-d", dbname])
    args = ["pg_restore", "-h", host, "-p", str(port), "-U", user,
            *target, "--no-owner", "--no-privileges", "--verbose",
            "-L", f"{mnt}/{toc_path.name}"]
    if jobs and jobs > 1:
        args += ["-j", str(jobs), "--exit-on-error"]
    else:
        args += ["--single-transaction"]
    if mode == "docker":
        if not shutil.which("docker"):
            sys.exit("restore-mode=docker but docker is not on PATH")
        cmd = ["docker", "run", "--rm",
               *(["--network", network] if network else []),
               "-e", f"PGPASSWORD={password}",
               "-v", f"{dump_path.parent.resolve().as_posix()}:/dump:ro",
               image, *args, f"/dump/{dump_path.name}"]
        env = None
    else:
        if not shutil.which("pg_restore"):
            sys.exit("restore-mode=local but pg_restore is not on PATH "
                     "(install postgresql-client, or use --restore-mode docker)")
        cmd = [*args, str(dump_path)]
        env = {**os.environ, "PGPASSWORD": password}
    printable = " ".join(c if "PGPASSWORD" not in c else "PGPASSWORD=***"
                         for c in cmd)
    print(f"  $ {printable}")
    if dry_run:
        return {"dry_run": True, "command": printable}
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    tail = "\n".join((proc.stderr or proc.stdout or "").splitlines()[-25:])
    if proc.returncode != 0:
        sys.exit(f"pg_restore failed (exit {proc.returncode}):\n{tail}")
    print("  pg_restore completed")
    return {"ok": True, "log_tail": tail}
