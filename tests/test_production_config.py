import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class ProductionConfigTests(unittest.TestCase):
    def test_floci_owns_postgres_and_uses_pgvector(self):
        compose = (ROOT / "infra" / "docker-compose.yml").read_text()
        self.assertIn(
            "FLOCI_SERVICES_RDS_DEFAULT_POSTGRES_IMAGE: pgvector/pgvector:pg16",
            compose)
        self.assertNotIn(
            "FLOCI_SERVICES_RDS_DEFAULT_POSTGRES_IMAGE: pgvector/pgvector\n",
            compose)
        self.assertNotIn("\n  postgres:", compose)

    def test_rds_and_restore_are_pg16(self):
        config = (ROOT / "infra" / "deploylib" / "config.py").read_text()
        database = (ROOT / "infra" / "deploylib" / "database.py").read_text()
        self.assertIn('RDS_ENGINE_VERSION = "pg16"', config)
        self.assertIn('POSTGRES_CLIENT_IMAGE = "pgvector/pgvector:pg16"', config)
        self.assertIn("describe_db_instances()", database)
        self.assertIn("_already_exists", database)

    def test_etl_floor_is_exact_utc_in_upgrade_migration(self):
        migration = (ROOT / "migrations" / "007_production_ingest_floor.sql").read_text()
        self.assertGreaterEqual(migration.count("2026-07-01T00:00:00+00:00"), 2)

    def test_scan_does_not_bypass_month_dispatcher(self):
        scan = (ROOT / "handlers" / "scan.py").read_text()
        claim_block = scan.split("if _upsert_and_claim", 1)[1].split("except Exception", 1)[0]
        self.assertNotIn("DOWNLOAD_QUEUE_URL", claim_block)

    def test_long_schema_operations_emit_heartbeats(self):
        migrate = (ROOT / "infra" / "dbmigrate.py").read_text()
        self.assertIn("schema operation still active", migrate)

    def test_github_workflow_is_in_discoverable_directory(self):
        self.assertTrue((ROOT / ".github" / "workflows" / "deploy.yml").is_file())

    def test_first_deploy_streams_remote_dump(self):
        workflow = (ROOT / ".github" / "workflows" / "deploy.yml").read_text()
        restore = (ROOT / "infra" / "dbrestore.py").read_text()
        self.assertIn("inputs.wipe_docker", workflow)
        self.assertIn("DUMP_URL: ${{ secrets.DUMP_URL }}", workflow)
        self.assertIn("--stream-restore", workflow)
        self.assertIn("def stream_restore", restore)
        self.assertIn("hashlib.sha256()", restore)
        self.assertIn('startswith(b"REFRESH MATERIALIZED VIEW ")', restore)
        self.assertIn("timeout-minutes: 360", workflow)

    def test_frontend_port_and_admin_are_production_safe(self):
        workflow = (ROOT / ".github" / "workflows" / "deploy.yml").read_text()
        frontend = (ROOT / "infra" / "deploylib" / "frontend.py").read_text()
        apis = (ROOT / "infra" / "deploylib" / "apis.py").read_text()
        self.assertIn("--frontend-host-port 80", workflow)
        self.assertIn("location /admin/", frontend)
        self.assertIn('$http_cf_connecting_ip != ""', frontend)
        self.assertIn("allow 192.168.0.0/16", frontend)
        self.assertIn("replacing cached $default admin URL", frontend)
        self.assertLess(apis.index("if rest:"), apis.index("if http:"))


if __name__ == "__main__":
    unittest.main()
