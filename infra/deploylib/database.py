"""RDS, ElastiCache, schema migrations, one-time restore, backlog floor."""
import json
import sys
import time
from urllib.parse import urlparse
import botocore.exceptions
import dbmigrate
import dbrestore
import dbcopy
from . import config as C
from .awsutil import err_code, ignore, matches, wait_for
class DatabaseMixin:
    # ----------------------------------------------------------------- RDS ---
    def _find_db_instance(self, ident):
        """None if absent.  Handles real AWS (fault raised) and emulators
        (200 with an empty list)."""
        try:
            resp = self.rds.describe_db_instances(DBInstanceIdentifier=ident)
        except self.rds.exceptions.DBInstanceNotFoundFault:
            return None
        except botocore.exceptions.ClientError as e:
            if matches(e, "NotFound"):
                return None
            raise
        instances = resp.get("DBInstances") or []
        return instances[0] if instances else None
    def ensure_rds(self):
        ident = f"{C.APP}-pg"
        if self._find_db_instance(ident) is None:
            print(f"creating RDS instance {ident} ...")
            base = dict(DBInstanceIdentifier=ident, Engine="postgres",
                        DBName=self.args.db_name,
                        MasterUsername=self.args.db_user,
                        MasterUserPassword=self.args.db_password,
                        DBInstanceClass="db.t3.micro", AllocatedStorage=20)
            try:
                self.rds.create_db_instance(
                    EngineVersion=C.RDS_ENGINE_VERSION, PubliclyAccessible=True,
                    BackupRetentionPeriod=0, **base)
            except botocore.exceptions.ClientError as e:
                # Emulators reject params they do not model; retry minimal.
                print(f"  full create rejected ({err_code(e)}), retrying minimal ...")
                self.rds.create_db_instance(**base)
        status = {"last": None}
        def ready():
            inst = self._find_db_instance(ident)
            status["last"] = (inst or {}).get("DBInstanceStatus")
            if inst and status["last"] == "available" \
                    and inst.get("Endpoint", {}).get("Address"):
                return inst
            return None
        inst = wait_for(ready, timeout=600, interval=5,
                        report=lambda: print(f"  waiting for {ident}: "
                                             f"status={status['last']}"))
        if not inst:
            sys.exit(
                f"RDS instance {ident} never became usable "
                f"(last status: {status['last']!r}).\n"
                f"If status stayed None, floci accepted the create call but never "
                f"materialised the instance -- check `docker logs floci` and "
                f"`docker ps` for a spawned postgres container.")
        port = int(inst["Endpoint"]["Port"])
        internal = self.args.db_internal_host or inst["Endpoint"]["Address"]
        self._upsert_secret(C.DB_SECRET_ID, {
            "host": internal, "port": port, "dbname": self.args.db_name,
            "username": self.args.db_user, "password": self.args.db_password})
        print(f"RDS {ident}: reported {inst['Endpoint']['Address']}:{port}")
        print(f"  lambdas / ECS use : {internal}:{port}")
        print(f"  your shell uses   : {self.args.db_external_host}:{port}")
    # --------------------------------------------------------- ElastiCache ---
    def _redis_endpoint(self, group_id, require_available=False):
        try:
            groups = self.ec.describe_replication_groups(
                ReplicationGroupId=group_id).get("ReplicationGroups", [])
        except botocore.exceptions.ClientError as e:
            if not matches(e, "NotFound"):
                raise
            return None
        if not groups:
            return None
        g = groups[0]
        if require_available and g.get("Status") != "available":
            return None
        nodes = g.get("NodeGroups") or []
        ep = (nodes[0].get("PrimaryEndpoint") if nodes
            else g.get("ConfigurationEndpoint")) or {}
        if not ep.get("Address"):
            return None
        return ep["Address"], int(ep.get("Port", 6379))
    def ensure_elasticache(self):
        gid = C.ELASTICACHE_GROUP_ID
        endpoint = self._redis_endpoint(gid)
        if endpoint:
            print(f"ElastiCache {gid} already exists at {endpoint[0]}:{endpoint[1]}")
        else:
            print(f"creating ElastiCache replication group {gid} ...")
            ignore(self.ec.create_replication_group,
                ReplicationGroupId=gid,
                ReplicationGroupDescription=f"{C.APP} cache",
                Engine="valkey", CacheNodeType="cache.t3.micro",
                NumCacheClusters=1) 
            endpoint = wait_for(
                lambda: self._redis_endpoint(gid, require_available=True),
                timeout=180,
                report=lambda: print("  waiting for ElastiCache ...")) \
                or ("localhost", 6379)
        host, port = endpoint
        # floci reports 'localhost', which is useless to other containers on the
        # docker network; rewrite to the floci hostname.
        if host in ("localhost", "127.0.0.1") and self.internal_endpoint():
            host = urlparse(self.internal_endpoint()).hostname
            print(f"  ElastiCache reported localhost; rewriting to {host}:{port}")
        self.elasticache = {"host": host, "port": port}
        self.put_params({f"/{C.APP}/elasticache/host": host,
                         f"/{C.APP}/elasticache/port": port})
        print(f"ElastiCache {gid}: {host}:{port}")
    def cache_endpoint(self):
        if self.elasticache:
            return self.elasticache["host"], str(self.elasticache["port"])
        return (self.get_param(f"/{C.APP}/elasticache/host", "localhost"),
                self.get_param(f"/{C.APP}/elasticache/port", "6379"))
    # -------------------------------------------------------------- schema ---
    def sql_target(self):
        creds = self.db_creds()
        mode = self.args.schema_mode
        if mode == "lambda":
            raise RuntimeError("sql_target() called in lambda schema-mode")
        # docker: we join the compose network, so the lambdas' hostname works.
        host = creds["host"] if mode == "docker" else self.args.db_external_host
        return dict(host=host, port=int(creds.get("port", 5432)),
                    dbname=creds["dbname"], user=creds["username"],
                    password=creds["password"], mode=mode,
                    network=self.args.schema_network or self.args.restore_network,
                    image=self.args.schema_image or self.args.restore_image)
    def run_schema_migrations(self):
        if self.args.schema_mode == "lambda":
            self.run_migrate("migrate")
            return
        target = self.sql_target()
        dbmigrate.apply(**target)
        if not self.args.skip_view_refresh:
            dbmigrate.populate_matviews(**target)
    def tune_database(self):
        if self.emulated and self.args.schema_mode != "lambda":
            dbmigrate.tune(**self.sql_target())
    # ------------------------------------------------- migrate lambda calls ---
    def _invoke_migrate(self, payload, fatal=True):
        try:
            resp = self._api_call(
                self.lam.invoke, FunctionName=f"{C.APP}-migrate",
                InvocationType="RequestResponse",
                Payload=json.dumps(payload).encode())
            body = resp["Payload"].read().decode()
            error = resp.get("FunctionError")
        except (botocore.exceptions.ConnectionClosedError,
                botocore.exceptions.EndpointConnectionError) as e:
            body, error = str(e), True
        if error:
            message = f"migrate {payload.get('action')} failed: {body[:400]}"
            if fatal:
                sys.exit(message)
            print(f"  ({message})")
            return None
        return json.loads(body or "null")
    def run_migrate(self, action):
        if self.args.schema_mode == "lambda" and action == "migrate":
            self.run_schema_migrations()
            return
        print(f"invoking migrate ({action}) ...")
        print(f"migrate result: {self._invoke_migrate({'action': action})}")
    def enforce_backlog_floor(self):
        if self.args.backlog_floor:
            self._invoke_migrate({"action": "set_config", "key": "backlog_floor",
                                  "value": self.args.backlog_floor})
        out = self._invoke_migrate({"action": "retire_backlog"})
        print(f"backlog: retired {out['retired']} pre-floor job(s) "
              f"(floor {out['floor']})")
    # ------------------------------------------------------------- restore ---
    def maybe_restore(self):
        """One-time load of the legacy youtube_data database.  Runs after the
        functions exist (so migrate can report state) and before the event source
        mappings exist (so nothing consumes while we rewrite the DB)."""
        if self.args.skip_restore:
            return print("restore: skipped (--skip-restore)")
        if not self.args.dump_url and not self.args.source_db_container:
            return print("restore: no dump URL or source DB container, skipping")
        state = self._invoke_migrate({"action": "restore_state"})
        if state.get("restored") and not self.args.force_restore:
            last = state["history"][0]
            return print(f"restore: already done {last['restored_at']} from "
                         f"{last['source']} -- skipping (--force-restore to redo)")
        if state.get("has_data") and not self.args.force_restore:
            sys.exit("restore: target database already contains data "
                     f"({', '.join(state['non_empty'])}) but has no restore "
                     "record. Refusing to overwrite. Pass --force-restore if "
                     "this is intentional.")
        creds = self.db_creds()
        port = int(creds.get("port", 5432))
        host = (creds["host"] if self.args.restore_mode == "docker"
                else self.args.db_external_host)
        print(f"\nrestoring {creds['dbname']} on {host}:{port} "
              f"({self.args.restore_mode} mode)")
        # If the pipeline is already live (a forced re-restore), stop the workers
        # first: they honour service_config.paused and re-queue their messages.
        paused_here = False
        if self.args.force_restore and state.get("restored"):
            paused_here = bool(self._invoke_migrate(
                {"action": "set_config", "key": "paused", "value": "true"},
                fatal=False))
            if paused_here:
                print("  pipeline paused for the duration of the restore")
        if self.args.source_db_container:
            source = f"docker://{self.args.source_db_container}"
            digest = None
            dbcopy.copy_container_database(
                container=self.args.source_db_container,
                source_db=self.args.source_db_name,
                source_user=self.args.source_db_user,
                source_password=self.args.source_db_password,
                host=host, port=port, dbname=creds["dbname"],
                user=creds["username"], password=creds["password"],
                network=self.args.restore_network,
                image=self.args.source_db_image,
                reset_schema=self.args.force_restore)
        else:
            source = self.args.dump_url
            dump, digest = dbrestore.fetch(
                self.args.dump_url, self.args.dump_cache_dir,
                sha256=self.args.dump_sha256, headers=self.args.dump_header)
            dbrestore.restore(
                dump, host=host, port=port, dbname=creds["dbname"],
                user=creds["username"], password=creds["password"],
                mode=self.args.restore_mode, network=self.args.restore_network,
                image=self.args.restore_image, jobs=self.args.restore_jobs,
                has_create=self.args.dump_has_create,
                reset_schema=self.args.force_restore)
        # pg_restore leaves reltuples = -1 and no column statistics; everything
        # downstream plans blind until this runs.
        if not self.args.skip_analyze and self.args.schema_mode != "lambda":
            dbmigrate.analyze(**self.sql_target())
        if not self.args.keep_migration_log:
            cleared = self._invoke_migrate({"action": "reset_migration_log"})
            if cleared and cleared.get("cleared"):
                print(f"  cleared {cleared['cleared']} inherited "
                      f"schema_migrations row(s)")
        rec = self._invoke_migrate({"action": "mark_restored",
                                    "source": source,
                                    "sha256": digest,
                                    "dbname": creds["dbname"]})
        print(f"  restored row counts: {rec['row_counts']}")
        if paused_here:
            self._invoke_migrate({"action": "set_config", "key": "paused",
                                  "value": "false"}, fatal=False)
            print("  pipeline un-paused")
