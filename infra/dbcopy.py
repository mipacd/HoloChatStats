"""One-time, diskless logical copy from the legacy PG18 container to RDS."""
import os
import shutil
import subprocess
import sys


def copy_container_database(*, container, host, port, dbname, user, password,
                            network, image, source_db=None, source_user=None,
                            source_password=None, reset_schema=False):
    if not shutil.which("docker"):
        sys.exit("direct database copy requires Docker on the runner")

    inspect = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", container],
        capture_output=True, text=True)
    if inspect.returncode or inspect.stdout.strip().lower() != "true":
        sys.exit(f"legacy database container {container!r} is not running")

    if reset_schema:
        _reset_target(host, port, dbname, user, password, network, image)

    # Empty SOURCE_* values deliberately fall back to the POSTGRES_* values
    # already present in the old container. No source secret is printed or
    # placed in argv; `docker exec -e NAME` copies it from this process env.
    source_env = dict(os.environ)
    if source_db:
        source_env["SOURCE_DB_NAME"] = source_db
    if source_user:
        source_env["SOURCE_DB_USER"] = source_user
    if source_password:
        source_env["PGPASSWORD"] = source_password
    dump_script = (
        'db="${SOURCE_DB_NAME:-${POSTGRES_DB:-youtube_data}}"; '
        'usr="${SOURCE_DB_USER:-${POSTGRES_USER:-postgres}}"; '
        'exec pg_dump -U "$usr" -d "$db" -Fc --no-owner --no-privileges'
    )
    dump_cmd = ["docker", "exec", "-i"]
    for name in ("SOURCE_DB_NAME", "SOURCE_DB_USER", "PGPASSWORD"):
        if source_env.get(name):
            dump_cmd += ["-e", name]
    dump_cmd += [container, "sh", "-ec", dump_script]

    target_env = {**os.environ, "PGPASSWORD": password}
    restore_cmd = ["docker", "run", "--rm", "-i"]
    if network:
        restore_cmd += ["--network", network]
    restore_cmd += ["-e", "PGPASSWORD", image, "pg_restore",
                    "-h", host, "-p", str(port), "-U", user, "-d", dbname,
                    "--clean", "--if-exists", "--no-owner", "--no-privileges",
                    "--exit-on-error", "--verbose"]

    print(f"  streaming {container} directly into {host}:{port}/{dbname}")
    print("  no dump file or database password will be written to disk")
    dump = subprocess.Popen(dump_cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, env=source_env)
    restore = subprocess.Popen(restore_cmd, stdin=dump.stdout,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               env=target_env)
    dump.stdout.close()
    restore_out, restore_err = restore.communicate()
    dump_err = dump.stderr.read()
    dump_rc = dump.wait()
    if dump_rc:
        sys.exit("legacy pg_dump failed:\n" +
                 dump_err.decode(errors="replace")[-4000:])
    if restore.returncode:
        sys.exit("RDS pg_restore failed:\n" +
                 (restore_err or restore_out).decode(errors="replace")[-4000:])
    print("  direct database copy completed")
    return {"ok": True}


def _reset_target(host, port, dbname, user, password, network, image):
    env = {**os.environ, "PGPASSWORD": password}
    cmd = ["docker", "run", "--rm"]
    if network:
        cmd += ["--network", network]
    cmd += ["-e", "PGPASSWORD", image, "psql", "-X", "-v", "ON_ERROR_STOP=1",
            "-h", host, "-p", str(port), "-U", user, "-d", dbname,
            "-c", "DROP SCHEMA IF EXISTS public CASCADE",
            "-c", "CREATE SCHEMA public"]
    subprocess.run(cmd, env=env, check=True)
