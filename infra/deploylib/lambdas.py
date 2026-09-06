"""Lambda packaging, role, and function lifecycle."""
import os
import shutil
import subprocess
import sys
import time
import zipfile
import botocore.exceptions
from . import config as C
class LambdaMixin:
    # ---------------------------------------------------------------- role ---
    def ensure_lambda_role(self):
        # NOTE: wide-open for the emulator.  For real AWS, replace the inline
        # policy with scoped grants (sqs:* on the queues, s3:* on the buckets,
        # secretsmanager:GetSecretValue on the two secrets, logs:*, and
        # lambda:UpdateEventSourceMapping on its own ESM for the concurrency knob).
        return self.ensure_role(f"{C.APP}-lambda-role", "lambda.amazonaws.com",
                                inline=C.WILDCARD_POLICY)
    # --------------------------------------------------------------- build ---
    def build_zip(self):
        if C.BUILD_DIR.exists():
            shutil.rmtree(C.BUILD_DIR)
        C.BUILD_DIR.mkdir(parents=True)
        if self.args.build_in_docker:
            print("pip-installing dependencies inside a linux container ...")
            install = ("pip install -q --no-cache-dir "
                       "-r /src/requirements.txt --target /src/build")
            # Always return bind-mounted output to the runner, even on failure.
            if hasattr(os, "getuid"):
                owner = f"{os.getuid()}:{os.getgid()}"
                install = (f"trap 'chown -R {owner} /src/build "
                           f"2>/dev/null || true' EXIT; {install}")
            subprocess.check_call([
                "docker", "run", "--rm", "-v", f"{C.ROOT.as_posix()}:/src",
                "-w", "/src", f"python:{self.args.python_version}-slim",
                "sh", "-c", install])
        else:
            print("pip-installing dependencies with host pip ...")
            cmd = [sys.executable, "-m", "pip", "install", "-q",
                   "-r", str(C.ROOT / "requirements.txt"),
                   "--target", str(C.BUILD_DIR)]
            if self.args.target_platform:
                cmd += ["--platform", self.args.target_platform,
                        "--only-binary", ":all:",
                        "--python-version", self.args.python_version,
                        "--implementation", "cp"]
            subprocess.check_call(cmd)
        self._assert_no_foreign_wheels()
        for pkg in ("common", "handlers", "migrations"):
            shutil.copytree(C.ROOT / pkg, C.BUILD_DIR / pkg,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        C.ZIP_PATH.unlink(missing_ok=True)
        with zipfile.ZipFile(C.ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(C.BUILD_DIR.rglob("*")):
                if not f.is_file() or "__pycache__" in f.parts:
                    continue
                info = zipfile.ZipInfo(f.relative_to(C.BUILD_DIR).as_posix())
                info.external_attr = (0o755 if f.suffix == ".so" else 0o644) << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                zf.writestr(info, f.read_bytes())
        self._code_cache = None
        print(f"built {C.ZIP_PATH} ({C.ZIP_PATH.stat().st_size / 1e6:.1f} MB)")
    def _assert_no_foreign_wheels(self):
        """Fail loudly at build time rather than at lambda init."""
        bad = [p.relative_to(C.BUILD_DIR).as_posix() for p in C.BUILD_DIR.rglob("*")
               if p.suffix in (".pyd", ".dll")
               or p.name.startswith("_delvewheel_patch")]
        if bad:
            sys.exit("ERROR: Windows/macOS binaries found in the build -- these "
                     "cannot run in the lambda container:\n  "
                     + "\n  ".join(bad[:10]) + "\n\nrebuild with --build-in-docker")
        if not any(C.BUILD_DIR.rglob("*.so")):
            print("WARNING: no .so files in build -- psycopg2 native extension "
                  "may be missing")
    def code_arg(self):
        """Inline zip, or an S3 stage when the package is too big.  Cached so we
        upload once per run, not once per function."""
        if self._code_cache is None:
            size = C.ZIP_PATH.stat().st_size
            if size < C.DIRECT_ZIP_LIMIT:
                self._code_cache = {"ZipFile": C.ZIP_PATH.read_bytes()}
            else:
                key = "deploy/build.zip"
                self.s3.upload_file(str(C.ZIP_PATH), C.BUCKETS["config"], key)
                print(f"zip is {size / 1e6:.1f} MB -> staged via "
                      f"s3://{C.BUCKETS['config']}/{key}")
                self._code_cache = {"S3Bucket": C.BUCKETS["config"], "S3Key": key}
        return self._code_cache
    # ----------------------------------------------------------------- env ---
    def common_env(self):
        env = {
            "APP_NAME": C.APP,
            "DB_SECRET_ID": C.DB_SECRET_ID,
            "YT_SECRET_ID": C.YT_SECRET_ID,
            "RAW_BUCKET": C.BUCKETS["raw"],
            "CONFIG_BUCKET": C.BUCKETS["config"],
            "SCAN_QUEUE_URL": self.queue_urls["scan-q"],
            "DOWNLOAD_QUEUE_URL": self.queue_urls["download-q"],
            "DOWNLOAD_RETRY_QUEUE_URL": self.queue_urls["download-retry-q"],
            "DOWNLOAD_DLQ_URL": self.queue_urls["download-dlq"],
            "INGEST_QUEUE_URL": self.queue_urls["ingest-q"],
            "METRIC_NAMESPACE": C.METRIC_NAMESPACE,
            "DISCOVER_RULE": f"{C.APP}-discover-schedule",
            "ADMIN_REFRESH_SECONDS": "5",
        }
        if self.args.lambda_endpoint:
            env["INGEST_ENDPOINT_URL"] = self.args.lambda_endpoint
        if self.elasticache:
            host, port = self.elasticache["host"], str(self.elasticache["port"])
            env.update(ELASTICACHE_HOST=host, ELASTICACHE_PORT=port,
                       REDIS_HOST=host, REDIS_PORT=port)
        return env
    # ----------------------------------------------------------- functions ---
    def ensure_functions(self, role_arn):
        env, code = self.common_env(), self.code_arg()
        for name, spec in C.FUNCTIONS.items():
            fn = f"{C.APP}-{name}"
            try:
                self.lam.get_function(FunctionName=fn)
                exists = True
            except self.lam.exceptions.ResourceNotFoundException:
                exists = False
            if exists:
                self.lam.update_function_code(FunctionName=fn, **code)
                self._wait_updated(fn)
                self.lam.update_function_configuration(
                    FunctionName=fn, Handler=spec["handler"],
                    Timeout=spec["timeout"], MemorySize=spec["memory"],
                    Environment={"Variables": env})
                print(f"updated function {fn}")
            else:
                self.lam.create_function(
                    FunctionName=fn, Runtime=C.PYTHON_RUNTIME, Role=role_arn,
                    Handler=spec["handler"], Code=code,
                    Timeout=spec["timeout"], MemorySize=spec["memory"],
                    Environment={"Variables": env})
                print(f"created function {fn}")
            self._wait_active(fn)
            if spec["rc"] is not None:
                try:
                    self.lam.put_function_concurrency(
                        FunctionName=fn,
                        ReservedConcurrentExecutions=spec["rc"])
                except botocore.exceptions.ClientError as e:
                    print(f"  (reserved concurrency unsupported: {e}; relying on "
                          f"ESM MaximumConcurrency only)")
            self.function_arns[name] = self.lam.get_function(
                FunctionName=fn)["Configuration"]["FunctionArn"]
    def update_all_function_code(self):
        code = self.code_arg()
        for name in C.FUNCTIONS:
            fn = f"{C.APP}-{name}"
            self.lam.update_function_code(FunctionName=fn, **code)
            self._wait_updated(fn)
            print(f"updated code: {fn}")
    def _wait_for_state(self, fn, key, blocked):
        for _ in range(60):
            cfg = self.lam.get_function(FunctionName=fn)["Configuration"]
            if cfg.get(key, blocked[0]) not in blocked:
                return
            time.sleep(2)
    def _wait_active(self, fn):
        self._wait_for_state(fn, "State", ("Pending",))
    def _wait_updated(self, fn):
        self._wait_for_state(fn, "LastUpdateStatus", ("InProgress",))
