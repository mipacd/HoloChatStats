import ast
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class ProductionConfigTests(unittest.TestCase):
    def test_floci_owns_postgres_and_uses_pgvector(self):
        compose = (ROOT / "infra" / "docker-compose.yml").read_text()
        self.assertIn("restart: unless-stopped", compose)
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
        self.assertIn("unterminated string", migrate)
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

    def test_dev_requirements_cover_runtime_modules_imported_by_tests(self):
        dev = (ROOT / "requirements-dev.txt").read_text().lower()
        self.assertIn("-r requirements.txt", dev)
        self.assertIn("boto3", dev)
        self.assertIn("botocore", dev)
        self.assertIn("sqlalchemy", dev)

    def test_replay_transients_and_quiet_outros_do_not_stall_a_month(self):
        youtube = (ROOT / "common" / "youtube.py").read_text(
            encoding="utf-8")
        download = (ROOT / "handlers" / "download.py").read_text(
            encoding="utf-8")
        migrate = (ROOT / "handlers" / "migrate.py").read_text(
            encoding="utf-8")
        self.assertIn("for attempt in range(8)", youtube)
        self.assertIn("_extract_video_info(url)", youtube)
        self.assertIn('"default", "web_embedded"', youtube)
        self.assertIn("requests.exceptions.HTTPError", youtube)
        self.assertIn('response.headers.get("Retry-After")', youtube)
        self.assertIn("REPLAY_END_SILENCE_SECONDS = 15 * 60", download)
        self.assertIn('"live event", "will begin"', download)
        self.assertIn('"age-restricted"', download)
        self.assertIn('"confirm your age"', download)
        self.assertIn('"video_start_ts": replay.video_start_ts', download)
        self.assertIn("video_start_ts=(msg.get", download)
        self.assertIn("duration=duration", download)
        self.assertIn("quiet tail; accepting completion", download)
        self.assertNotIn('RuntimeError(f"truncated: last message', download)
        self.assertNotIn("video_start_ts is None or continuation is None",
                         youtube)
        self.assertIn('"page needs to be reloaded"', youtube)
        for marker in ("503 Server Error", "Service Unavailable",
                       "truncated: last message", "will begin",
                       "age-restricted", "page needs to be reloaded"):
            self.assertIn(marker, migrate)
        scan = (ROOT / "handlers" / "scan.py").read_text(encoding="utf-8")
        self.assertIn('"age-restricted", "age restricted"', scan)
        self.assertIn("derived_start_ts", download)
        self.assertIn("v.end_time - COALESCE", download)
        self.assertIn("FOR UPDATE OF j", download)
        self.assertIn("invalid replay continuation; restarting", download)
        self.assertIn("reset_checkpoint=stale_checkpoint", download)
        self.assertIn("_find_live_chat_renderer", youtube)
        self.assertIn("_fetch_initial_chat", youtube)
        self.assertIn("INNERTUBE_CONTEXT", youtube)
        self.assertIn("X-Goog-Visitor-Id", youtube)
        self.assertIn("currentPlayerState", youtube)
        self.assertIn("class RawPartCorrupt", (ROOT / "handlers" /
                                               "ingest.py").read_text())
        self.assertIn("corrupt raw part; awaiting operator retry",
                      (ROOT / "handlers" / "ingest.py").read_text())
        self.assertIn("400 Client Error", migrate)
        workflow = (ROOT / ".github" / "workflows" / "deploy.yml").read_text()
        self.assertIn("python scripts/validate_youtube_cookies.py", workflow)
        self.assertIn("YOUTUBE_USER_AGENT is required", workflow)

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

    def test_locale_default_and_site_metrics_websocket(self):
        navbar = (ROOT / "frontend" / "src" / "components" /
                  "Navbar.tsx").read_text(encoding="utf-8")
        admin = (ROOT / "handlers" / "admin.py").read_text(encoding="utf-8")
        nginx = (ROOT / "infra" / "deploylib" / "frontend.py").read_text()
        webapi = (ROOT / "infra" / "deploylib" / "webapi.py").read_text()
        server = (ROOT / "web" / "server.py").read_text(encoding="utf-8")
        self.assertIn("i18n.resolvedLanguage", navbar)
        self.assertIn("const activeLanguage", navbar)
        self.assertEqual(navbar.count("value={activeLanguage}"), 2)
        self.assertIn("location ^~ /socket.io/", nginx)
        self.assertIn('Connection "upgrade"', nginx)
        socket_block = nginx.split("location ^~ /socket.io/", 1)[1].split(
            "location", 1)[0]
        self.assertIn('proxy_pass            http://${API_BACKEND};',
                      socket_block)
        self.assertNotIn("__UPSTREAM__", socket_block)
        self.assertIn("--workers 1", webapi)
        self.assertIn("--threads 16", webapi)
        self.assertIn("$http_x_forwarded_proto", socket_block)
        self.assertIn("async_mode='threading'", server)
        requirements = (ROOT / "web" / "requirements.txt").read_text()
        self.assertIn("simple-websocket", requirements)
        self.assertNotIn("eventlet", requirements)
        self.assertIn("socketio.start_background_task(metrics_updates)", server)
        self.assertIn("to=request.sid", server)

    def test_spa_page_views_exclude_internal_traffic(self):
        app = (ROOT / "frontend" / "src" / "App.tsx").read_text(
            encoding="utf-8")
        server = (ROOT / "web" / "server.py").read_text(encoding="utf-8")
        utils = (ROOT / "web" / "utils.py").read_text(encoding="utf-8")
        self.assertIn("function PageViewTracker()", app)
        self.assertIn('fetch("/api/metrics/page-view"', app)
        self.assertIn("[pathname]", app)
        self.assertIn('pathname.startsWith("/stream_stats/")', app)
        self.assertIn("def page_view_metric", server)
        self.assertIn("if not is_public_page(path)", server)
        self.assertIn('"stored": record_page_view(path)', server)
        self.assertIn('METRICS_NAMESPACE = "v2"', utils)
        for internal in ("/api/", "/static/", "/socket.io/", "/admin/",
                         "/health", "/favicon.ico"):
            self.assertIn(internal, utils)
        self.assertIn("is_public_page(page)", utils)
        self.assertIn('path.startswith("/stream_stats/")', utils)
        self.assertIn("scan_iter", utils)

    def test_redis_outage_is_fast_and_does_not_break_web_or_llm(self):
        utils = (ROOT / "web" / "utils.py").read_text(encoding="utf-8")
        rate_limit = (ROOT / "llm_chat" / "rate_limit.py").read_text(
            encoding="utf-8")
        migrate = (ROOT / "handlers" / "migrate.py").read_text(
            encoding="utf-8")
        workflow = (ROOT / ".github" / "workflows" /
                    "deploy.yml").read_text(encoding="utf-8")
        invoke = (ROOT / "scripts" / "invoke.py").read_text(encoding="utf-8")
        self.assertIn('REDIS_IO_TIMEOUT_SECONDS", "0.5"', utils)
        self.assertIn("_redis_down_until", utils)
        self.assertIn("_local_cache", utils)
        self.assertIn("except redis.RedisError", rate_limit)
        self.assertIn("return False", rate_limit)
        self.assertIn("return settings.LLM_DAILY_LIMIT", rate_limit)
        self.assertIn('stage("redis", _redis)', migrate)
        database = (ROOT / "infra" / "deploylib" / "database.py").read_text(
            encoding="utf-8")
        self.assertIn('f"floci-valkey-{gid}"', database)
        self.assertNotIn("urlparse(self.internal_endpoint()).hostname", database)
        self.assertIn("floci-valkey-", workflow)
        self.assertIn("--restart unless-stopped", workflow)
        self.assertIn("--require-stage postgres --require-stage redis", workflow)
        self.assertIn('"--require-stage"', invoke)

    def test_month_finalize_invalidates_only_aggregate_web_caches(self):
        invalidation = (ROOT / "common" /
                        "cache_invalidation.py").read_text(encoding="utf-8")
        merge = (ROOT / "handlers" / "merge.py").read_text(encoding="utf-8")
        requirements = (ROOT / "requirements.txt").read_text()
        for pattern in (
            "channel_recommendations:*", "monthly_streaming_hours_*",
            "exclusive_chat_users_*", "message_type_percents_*",
            "jp_user_percent_*", "stream_frequency_*", "stream_calendar_*",
            "channel_names", "date_ranges", "number_of_chat_logs",
            "num_messages", "published_coverage_v2:*",
        ):
            self.assertIn(f'"{pattern}"', invalidation)
        for protected in ("rate:*", "v2:*", "cache_hits:*", "cache_misses:*"):
            self.assertNotIn(f'"{protected}"', invalidation)
        self.assertIn("scan_iter", invalidation)
        self.assertNotIn("flushdb", invalidation.lower())
        self.assertIn("cache_finalized_month", merge)
        self.assertIn("if dry_run else _sync_cache_invalidation", merge)
        self.assertLess(merge.index("invalidate_finalized_month_caches("),
                        merge.index('("cache_finalized_month", str(latest))'))
        self.assertIn("redis", requirements.splitlines())

    def test_expensive_bounded_endpoints_are_warmed_in_background(self):
        warmer = (ROOT / "web" / "cache_warmer.py").read_text(
            encoding="utf-8")
        server = (ROOT / "web" / "server.py").read_text(encoding="utf-8")
        self.assertIn("/api/get_exclusive_chat_users", warmer)
        self.assertIn("WHERE active", warmer)
        self.assertIn("CACHE_WARM_INTERVAL_SECONDS", warmer)
        self.assertIn("CACHE_WARM_QUIET_SECONDS", warmer)
        self.assertIn("CACHE_WARM_MAX_LOAD_PER_CPU", warmer)
        self.assertIn("CACHE_WARM_DELAY_SECONDS", warmer)
        self.assertIn("def _enabled(app)", warmer)
        self.assertIn("cache_warmer_enabled", warmer)
        self.assertIn('result.get("paused")', warmer)
        self.assertNotIn("/api/get_user_info", warmer)
        self.assertIn("HoloChatStats-cache-warmer/1.0", server)
        self.assertIn("if not is_cache_warmer", server)
        admin = (ROOT / "handlers" / "admin.py").read_text(encoding="utf-8")
        for action in ("cache_warmer_enable", "cache_warmer_disable"):
            self.assertIn(action, admin)
        self.assertIn('id="cache-warmer-state"', admin)

    def test_modest_host_limits_ingest_and_allows_slow_s3_reads(self):
        config = (ROOT / "infra" / "deploylib" / "config.py").read_text()
        aws = (ROOT / "common" / "aws.py").read_text()
        assignments = {}
        for node in ast.parse(config).body:
            if (isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id in {
                        "FUNCTIONS", "EVENT_SOURCE_MAPPINGS"
                    }):
                assignments[node.targets[0].id] = ast.literal_eval(node.value)
        self.assertEqual(assignments["FUNCTIONS"]["ingest"]["rc"], 1)
        self.assertEqual(assignments["EVENT_SOURCE_MAPPINGS"]["ingest-q"],
                         ("ingest", 1, 2))
        self.assertIn('S3_READ_TIMEOUT_SECONDS", "120"', aws)
        self.assertIn('service == "s3"', aws)

    def test_analytics_cache_entries_have_no_expiration(self):
        api = (ROOT / "web" / "api.py").read_text(encoding="utf-8")
        utils = (ROOT / "web" / "utils.py").read_text(encoding="utf-8")
        invalidation = (ROOT / "common" /
                        "cache_invalidation.py").read_text(encoding="utf-8")
        self.assertNotIn("ttl=", api)
        cache_helper = utils.split("def get_or_compute_cached", 1)[1].split(
            "def cached_json", 1)[0]
        self.assertNotIn("ex=", cache_helper)
        self.assertIn("FINALIZED_MONTH_KEY_TEMPLATES", invalidation)
        self.assertIn("_persist_analytics_caches",
                      (ROOT / "web" / "cache_warmer.py").read_text())
        self.assertIn("persist(redis_key)", cache_helper)
        # Operational keys still need bounded lifetimes for correct rate
        # limiting and finite metrics storage.
        self.assertIn("pipe.expire(key, rate_limit_window)",
                      (ROOT / "web" / "server.py").read_text())

    def test_active_download_progress_reserves_100_percent_for_completion(self):
        admin = (ROOT / "handlers" / "admin.py").read_text(encoding="utf-8")
        self.assertIn("min(99.9", admin)
        self.assertIn('r[3] in ("downloaded", "ingesting")', admin)

    def test_late_replays_are_bounded_and_republish_finalized_months(self):
        scan = (ROOT / "handlers" / "scan.py").read_text(encoding="utf-8")
        ingest = (ROOT / "handlers" / "ingest.py").read_text(encoding="utf-8")
        refresh = (ROOT / "handlers" / "refresh.py").read_text(encoding="utf-8")
        self.assertIn('prior["status"] == "skipped" and pages == 1', scan)
        self.assertIn("_recheckable_skip", scan)
        self.assertIn("reopen_skipped=recheck_skip", scan)
        self.assertIn("reached_floor and pages == 1 and not recheck_skip", scan)
        self.assertIn("end_date < floor and not recheck_skip", scan)
        self.assertIn("successful watermark preserved", scan)
        error_writer = scan.split("def _record_error", 1)[1].split(
            "def _channel_baseline", 1)[0]
        self.assertNotIn("last_scanned_at = NOW()", error_writer)
        self.assertIn("late_data_month:", ingest)
        self.assertIn("late_data_month:%", refresh)
        self.assertIn("invalidate_finalized_month_caches(finalized_month=m)",
                      refresh)
        self.assertIn("updated_at=%s", refresh)
        self.assertIn("late_data_published:", refresh)
        self.assertIn('event.get("publish_months")', refresh)
        admin = (ROOT / "handlers" / "admin.py").read_text(encoding="utf-8")
        self.assertIn('out["months"]', admin)
        self.assertIn('"republish_month"', admin)
        self.assertIn('id="months"', admin)
        self.assertIn("Re-publish", admin)
        self.assertIn('"publish_month"', admin)
        self.assertIn("Channel checks", admin)

    def test_admin_can_safely_unpublish_and_reports_eri_usage(self):
        admin = (ROOT / "handlers" / "admin.py").read_text(encoding="utf-8")
        merge = (ROOT / "handlers" / "merge.py").read_text(encoding="utf-8")
        ingest = (ROOT / "handlers" / "ingest.py").read_text(encoding="utf-8")
        llm = (ROOT / "llm_chat" / "rate_limit.py").read_text(
            encoding="utf-8")
        self.assertIn('"unpublish_month"', admin)
        self.assertIn("unpublishMonth", admin)
        self.assertIn('id="eri-usage"', admin)
        self.assertIn('class="table-scroll months"', admin)
        self.assertIn('class="table-scroll channels"', admin)
        self.assertIn("max-height:330px", admin)
        self.assertIn("max-height:520px", admin)
        self.assertIn('event.get("unpublish_months")', merge)
        self.assertIn("publication_hold:", merge)
        self.assertIn("unpublish_cache_pending:", merge)
        self.assertIn("_retry_unpublish_caches(conn)", merge)
        self.assertIn("DELETE FROM user_data u USING videos v", merge)
        self.assertIn("INSERT INTO user_data_current", merge)
        self.assertIn("FOR SHARE", ingest)
        self.assertIn("llm_usage_total:", llm)
        self.assertIn("timedelta(days=62)", llm)

    def test_failed_jobs_are_terminal_until_an_admin_retry(self):
        download = (ROOT / "handlers" / "download.py").read_text()
        ingest = (ROOT / "handlers" / "ingest.py").read_text()
        ordering = (ROOT / "common" / "month_order.py").read_text()
        merge = (ROOT / "handlers" / "merge.py").read_text()
        admin = (ROOT / "handlers" / "admin.py").read_text(encoding="utf-8")
        deploy = (ROOT / "infra" / "deploy.py").read_text()
        self.assertIn('("done", "failed", "skipped", "ingesting", "downloaded")',
                      download)
        self.assertIn("WHERE video_id=%s AND status = 'downloaded'", ingest)
        self.assertIn("SET status='failed'", ingest)
        self.assertIn("'done', 'failed', 'skipped'", ordering)
        self.assertNotIn("if failed and not ignore_failed", merge)
        self.assertIn('"retry_job"', admin)
        self.assertIn('id="btn-retry-all"', admin)
        self.assertIn("Stream ended (UTC)", admin)
        self.assertNotIn('id="btn-retry"', admin)
        self.assertNotIn('stack.run_migrate("retry_cookie_failures")', deploy)

    def test_month_barrier_targets_every_unscanned_channel(self):
        merge = (ROOT / "handlers" / "merge.py").read_text()
        discover = (ROOT / "handlers" / "discover.py").read_text()
        self.assertIn("_unscanned_channel_ids", merge)
        self.assertIn("_request_barrier_scan", merge)
        self.assertIn('"force": True, "channels": channel_ids', merge)
        self.assertIn("merge_barrier_scan:", merge)
        self.assertIn("INTERVAL '30 minutes'", merge)
        targeted = discover.split("def _eligible_channels", 1)[1].split(
            "def _apply_concurrency", 1)[0]
        self.assertIn("if only:", targeted)
        self.assertLess(targeted.index("if only:"),
                        targeted.index("eligible[:limit]"))

    def test_home_shows_next_month_publication_backlog_only_when_behind(self):
        api = (ROOT / "web" / "api.py").read_text(encoding="utf-8")
        home = (ROOT / "frontend" / "src" / "pages" /
                "Home.tsx").read_text(encoding="utf-8")
        endpoint = api.split("def get_publication_progress", 1)[1].split(
            "@api_bp.route", 1)[0]
        self.assertIn("MAX(observed_month)", endpoint)
        self.assertIn("latest + INTERVAL '1 month'", endpoint)
        self.assertIn("j.status NOT IN ('done', 'failed', 'skipped')", endpoint)
        self.assertNotIn("remaining_channel_checks", endpoint)
        self.assertNotIn("channel_watermarks", endpoint)
        decorators = api.split(
            "@api_bp.route('/api/get_publication_progress'", 1)[1].split(
                "def get_publication_progress", 1)[0]
        self.assertNotIn("@cached_json", decorators)
        self.assertIn('api.get("/get_publication_progress")', home)
        self.assertIn("publication?.behind", home)
        self.assertIn("remaining_chat_logs", home)
        self.assertNotIn("remaining_channel_checks", home)
        self.assertNotIn("{{channels}} channel checks", home)

    def test_home_coverage_is_fenced_to_published_months_and_cached(self):
        api = (ROOT / "web" / "api.py").read_text(encoding="utf-8")
        coverage = api.split(
            "@api_bp.route('/api/get_date_ranges'", 1)[1].split(
                "@api_bp.route('/api/get_funniest_timestamps'", 1)[0]
        for endpoint in ("date_ranges", "number_of_chat_logs", "num_messages"):
            self.assertIn(f'published_coverage_v2:{endpoint}', coverage)
        self.assertGreaterEqual(coverage.count("MAX(observed_month)"), 3)
        self.assertGreaterEqual(coverage.count("status = 'merged'"), 3)
        self.assertGreaterEqual(coverage.count("v.end_time < p.cutoff"), 3)
        # Progress must remain live while the current month drains.
        progress_decorators = api.split(
            "@api_bp.route('/api/get_publication_progress'", 1)[1].split(
                "def get_publication_progress", 1)[0]
        self.assertNotIn("@cached_json", progress_decorators)

    def test_public_month_analytics_are_fenced_by_merge_watermark(self):
        api = (ROOT / "web" / "api.py").read_text(encoding="utf-8")
        picker = (ROOT / "frontend" / "src" / "components" / "ui" /
                  "month-picker.tsx").read_text(encoding="utf-8")
        merge = (ROOT / "handlers" / "merge.py").read_text(encoding="utf-8")
        self.assertIn("@api_bp.before_request", api)
        self.assertIn('"code": "month_not_published"', api)
        self.assertIn("MAX(observed_month)", api)
        self.assertIn("disabled={isUnpublished}", picker)
        self.assertIn("request_warm=False", merge)

    def test_stream_stats_are_aggregate_only_and_public_on_ingest(self):
        migration = (ROOT / "migrations" / "009_video_stream_stats.sql").read_text(
            encoding="utf-8")
        ingest = (ROOT / "handlers" / "ingest.py").read_text(encoding="utf-8")
        api = (ROOT / "web" / "api.py").read_text(encoding="utf-8")
        reaper = (ROOT / "handlers" / "reap.py").read_text(encoding="utf-8")
        config = (ROOT / "infra" / "deploylib" / "config.py").read_text(
            encoding="utf-8")
        app = (ROOT / "frontend" / "src" / "App.tsx").read_text(encoding="utf-8")
        navbar = (ROOT / "frontend" / "src" / "components" /
                  "Navbar.tsx").read_text(encoding="utf-8")
        stream_page = (ROOT / "frontend" / "src" / "pages" /
                       "StreamStats.tsx").read_text(encoding="utf-8")
        admin = (ROOT / "handlers" / "admin.py").read_text(encoding="utf-8")
        self.assertIn("CREATE TABLE IF NOT EXISTS video_stream_stats", migration)
        for forbidden in ("username", "user_id", "message_text"):
            self.assertNotIn(forbidden, migration)
        self.assertIn("upsert_stream_stats(cur, video_id, aggregate)", ingest)
        self.assertIn("stream_stats_backfill_enabled", ingest)
        self.assertIn('request.path.startswith("/api/stream-stats")', api)
        self.assertIn('"groups": sorted(', api)
        self.assertIn("_retained_stream_video_ids()", admin)
        self.assertIn('Delimiter="/"', admin)
        for action in ("stream_stats_backfill_start",
                       "stream_stats_backfill_pause",
                       "stream_stats_backfill_retry"):
            self.assertIn(action, admin)
        self.assertIn("LIMIT 1 FOR UPDATE OF s SKIP LOCKED", reaper)
        self.assertIn("DelaySeconds=0", reaper)
        self.assertIn("s.attempts < 3", reaper)
        self.assertIn("status IN ('queued','processing')", reaper)
        self.assertIn('"unavailable" if unavailable else "failed"', ingest)
        self.assertIn('FunctionName=f"{APP}-reap"', ingest)
        self.assertIn('InvocationType="Event"', ingest)
        self.assertIn('"ingest":   {"handler": "handlers.ingest.handler",   '
                      '"timeout": 900, "memory": 1024, "rc": 1}', config)
        self.assertIn('"legacy-stats-import": {"handler": '
                      '"handlers.legacy_stats_import.handler"', config)
        for parameter in ("month", "channel", "group", "page", "page_size"):
            self.assertIn(f'request.args.get("{parameter}"', api)
        self.assertIn("/api/stream-stats/<video_id>", api)
        self.assertIn('/stream_stats/:videoId', app)
        self.assertIn('t("Language Percentages / Rates")', navbar)
        self.assertIn("whitespace-nowrap", navbar)
        self.assertIn("layoutWordCloud", stream_page)
        self.assertIn("bg-popover", stream_page)
        self.assertNotIn('t("Aggregate chat statistics become available',
                         stream_page)
        self.assertNotIn("😂", stream_page)


if __name__ == "__main__":
    unittest.main()
