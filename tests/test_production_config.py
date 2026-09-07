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

    def test_emulated_web_uses_discovered_host_port(self):
        webapi = (ROOT / "infra" / "deploylib" / "webapi.py").read_text()
        branch = webapi.split("if self.emulated:", 1)[1].split(
            "elif public_ip", 1)[0]
        self.assertIn("discover_forwarded_port", branch)
        self.assertNotIn("reusing the previously published port", webapi)
        self.assertIn("refusing to update its SSM", webapi)

    def test_cookie_secret_single_download_and_retry(self):
        workflow = (ROOT / ".github" / "workflows" / "deploy.yml").read_text()
        config = (ROOT / "infra" / "deploylib" / "config.py").read_text()
        download = (ROOT / "handlers" / "download.py").read_text()
        migrate = (ROOT / "handlers" / "migrate.py").read_text()
        self.assertIn("YOUTUBE_COOKIES_B64", workflow)
        self.assertIn('"download": {"handler": "handlers.download.handler", "timeout": 900, "memory": 1024, "rc": 1}', config)
        self.assertIn("pg_try_advisory_lock", download)
        self.assertNotIn("yielding to recovery queue", download)
        self.assertIn("retry_cookie_failures", migrate)
        self.assertIn("drain_retry_queue", migrate)
        self.assertIn('@app.get("/healthz/llm")',
                      (ROOT / "llm_chat" / "main.py").read_text())
        self.assertIn('LLM_HEALTH_PATH = "/healthz/llm"', config)
        webapi = (ROOT / "infra" / "deploylib" / "webapi.py").read_text()
        uvicorn_at = webapi.index("nohup /opt/web/venv/bin/uvicorn")
        index_at = webapi.index("python init_tool_store.py", uvicorn_at)
        self.assertGreater(index_at, uvicorn_at)
        self.assertIn("llm indexing started in background", webapi)
        model = (ROOT / "llm_chat" / "llm" / "model.py").read_text()
        self.assertIn('"reasoning": reasoning', model)
        self.assertNotIn('"extra_body"', model)
        self.assertIn("HTTP {exc.response.status_code}", model)
        self.assertIn("/proc/1/fd/1", webapi)
        frontend = (ROOT / "infra" / "deploylib" / "frontend.py").read_text()
        zero = frontend.index("desiredCount=0")
        update = frontend.index("self._api_call(self.ecs.update_service, **kwargs)")
        self.assertLess(zero, update)
        self.assertIn("_stop_orphaned_emulator_frontends", frontend)
        self.assertIn('startswith("floci-ecs-")', frontend)

    def test_download_rechecks_month_order_at_execution(self):
        download = (ROOT / "handlers" / "download.py").read_text()
        ingest = (ROOT / "handlers" / "ingest.py").read_text()
        ordering = (ROOT / "common" / "month_order.py").read_text()
        self.assertIn("_defer_future_month(msg)", download)
        self.assertIn("dispatched_at=NULL", download)
        self.assertIn("user_data_current", ordering)
        self.assertIn("monthly_merge_state", ordering)
        self.assertIn("future-month ingest deferred", ingest)

    def test_lan_llm_requests_are_quota_exempt_and_diagnostic(self):
        main = (ROOT / "llm_chat" / "main.py").read_text()
        limiter = (ROOT / "llm_chat" / "rate_limit.py").read_text()
        admin = (ROOT / "handlers" / "admin.py").read_text(encoding="utf-8")
        self.assertIn('request.headers.get("X-Real-IP")', main)
        self.assertIn("if cf_ip and is_local_client(proxy_ip)", main)
        self.assertIn("admin or local_client", main)
        self.assertIn('"quota_exempt": local_client', main)
        self.assertIn("exempt: bool = False", limiter)
        self.assertIn('cfg.get("backlog_floor", "")', admin)
        planner = (ROOT / "llm_chat" / "llm" / "planner.py").read_text()
        self.assertIn('relevant_tools = tools_description or ""', planner)

    def test_chart_urls_and_admin_news_are_proxied(self):
        main = (ROOT / "llm_chat" / "main.py").read_text()
        widget = (ROOT / "frontend" / "src" / "components" / "eri" /
                  "EriWidget.tsx").read_text(encoding="utf-8")
        nginx = (ROOT / "infra" / "deploylib" / "frontend.py").read_text()
        admin = (ROOT / "handlers" / "admin.py").read_text(encoding="utf-8")
        webapi = (ROOT / "web" / "api.py").read_text()
        deploy = (ROOT / "infra" / "deploy.py").read_text()
        self.assertIn("/llm/charts/{filename}", main)
        self.assertIn("chart[1]", widget)
        self.assertIn("location ^~ /llm/", nginx)
        self.assertIn('path == "/api/news"', admin)
        self.assertIn('NEWS_KEY = "news.txt"', admin)
        self.assertIn("Key=NEWS_KEY", admin)
        self.assertIn('id="news-text"', admin)
        self.assertIn('Key="news.txt"', webapi)
        self.assertIn("stack.ensure_news_seed()", deploy)


if __name__ == "__main__":
    unittest.main()
