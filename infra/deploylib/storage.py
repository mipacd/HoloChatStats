"""S3 buckets and Secrets Manager."""
import base64
import binascii
import json
import secrets
import sys
from pathlib import Path
import botocore.exceptions
from . import config as C
class StorageMixin:
    # ----------------------------------------------------------------- S3 ---
    def ensure_bucket(self, name):
        try:
            self.s3.head_bucket(Bucket=name)
            return False
        except botocore.exceptions.ClientError:
            kwargs = {"Bucket": name}
            if self.args.region != "us-east-1":
                kwargs["CreateBucketConfiguration"] = {
                    "LocationConstraint": self.args.region}
            self.s3.create_bucket(**kwargs)
            print(f"created bucket {name}")
            return True
    def ensure_buckets(self):
        for name in C.BUCKETS.values():
            self.ensure_bucket(name)
        self.ensure_raw_lifecycle()
        if self.args.channels_file:
            self.s3.put_object(
                Bucket=C.BUCKETS["config"], Key="channels.json",
                Body=Path(self.args.channels_file).read_bytes(),
                ContentType="application/json")
            print(f"uploaded {self.args.channels_file} -> "
                  f"s3://{C.BUCKETS['config']}/channels.json")
        self.ensure_news_seed()

    def ensure_raw_lifecycle(self):
        self.s3.put_bucket_lifecycle_configuration(
            Bucket=C.BUCKETS["raw"],
            LifecycleConfiguration={"Rules": [{
                "ID": f"expire-raw-chat-{C.RAW_RETENTION_DAYS}d",
                "Status": "Enabled",
                "Filter": {"Prefix": ""},
                "Expiration": {"Days": C.RAW_RETENTION_DAYS},
                "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 7},
            }, {
                "ID": ("expire-legacy-stream-stats-import-"
                       f"{C.LEGACY_IMPORT_RETENTION_DAYS}d"),
                "Status": "Enabled",
                "Filter": {"Prefix": C.LEGACY_IMPORT_PREFIX},
                "Expiration": {"Days": C.LEGACY_IMPORT_RETENTION_DAYS},
                "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 1},
            }]})
        print(f"lifecycle: {C.BUCKETS['raw']} objects expire after "
              f"{C.RAW_RETENTION_DAYS} days; legacy import staging expires "
              f"after {C.LEGACY_IMPORT_RETENTION_DAYS} day")
    def ensure_news_seed(self):
        """Create the editable news object once; never overwrite admin edits."""
        key = "news.txt"
        try:
            self.s3.head_object(Bucket=C.BUCKETS["config"], Key=key)
            return
        except botocore.exceptions.ClientError:
            pass
        source = C.ROOT / "web" / "news.txt"
        body = source.read_bytes() if source.exists() else b""
        self.s3.put_object(Bucket=C.BUCKETS["config"], Key=key, Body=body,
                           ContentType="text/plain; charset=utf-8")
        print(f"seeded s3://{C.BUCKETS['config']}/{key} for admin editing")
    def make_bucket_public(self, bucket):
        """Anonymous GET.  Only used for the frontend tarball in bootstrap mode;
        real AWS deployments should use ECR mode instead."""
        self._api_call(self.s3.put_bucket_policy, Bucket=bucket,
                       Policy=json.dumps({"Version": "2012-10-17", "Statement": [{
                           "Sid": "PublicRead", "Effect": "Allow",
                           "Principal": "*", "Action": "s3:GetObject",
                           "Resource": f"arn:aws:s3:::{bucket}/*"}]}))
    # ------------------------------------------------------------- secrets ---
    def _upsert_secret(self, secret_id, payload):
        body = json.dumps(payload)
        try:
            self.sm.create_secret(Name=secret_id, SecretString=body)
            print(f"created secret {secret_id}")
        except self.sm.exceptions.ResourceExistsException:
            self.sm.put_secret_value(SecretId=secret_id, SecretString=body)
            print(f"updated secret {secret_id}")
    def _ensure_secret(self, secret_id, payload, overwrite):
        try:
            self.sm.describe_secret(SecretId=secret_id)
            if overwrite and payload:
                self._upsert_secret(secret_id, payload)
        except self.sm.exceptions.ResourceNotFoundException:
            if not payload:
                sys.exit(f"secret {secret_id} does not exist and no value was "
                         f"supplied -- pass the relevant CLI flags on first deploy")
            self._upsert_secret(secret_id, payload)
    def ensure_secrets(self):
        # With --create-rds, ensure_rds() writes the DB secret once it knows the
        # real endpoint; writing it here would be premature and wrong.
        if not self.args.create_rds:
            db = None
            if self.args.db_host:
                db = {"host": self.args.db_host, "port": self.args.db_port,
                      "dbname": self.args.db_name, "username": self.args.db_user,
                      "password": self.args.db_password}
            self._ensure_secret(C.DB_SECRET_ID, db, self.args.update_secrets)
        yt = ({"api_key": self.args.youtube_api_key}
              if self.args.youtube_api_key else None)
        if self.args.youtube_cookies_b64:
            try:
                raw = base64.b64decode(self.args.youtube_cookies_b64,
                                       validate=True)
                first = raw.decode("utf-8-sig").splitlines()[0]
            except (binascii.Error, UnicodeDecodeError, IndexError) as exc:
                sys.exit(f"YOUTUBE_COOKIES_B64 is not a valid UTF-8 base64 "
                         f"cookie file: {exc}")
            if first not in ("# HTTP Cookie File", "# Netscape HTTP Cookie File"):
                sys.exit("YouTube cookie secret must be a Netscape cookies.txt "
                         "file with its required header")
            yt = {**(yt or {}), "cookies_b64": self.args.youtube_cookies_b64}
            if self.args.youtube_user_agent:
                yt["user_agent"] = self.args.youtube_user_agent
        # Supplying cookies is an explicit rotation and must update an existing
        # API-key-only secret during an ordinary code deployment.
        self._ensure_secret(C.YT_SECRET_ID, yt,
                            self.args.update_secrets or
                            bool(self.args.youtube_cookies_b64))
        self.ensure_web_secret()
    def ensure_web_secret(self):
        try:
            self.sm.get_secret_value(SecretId=C.WEB_SECRET_ID)
        except self.sm.exceptions.ResourceNotFoundException:
            self._upsert_secret(C.WEB_SECRET_ID,
                                {"session_key": secrets.token_urlsafe(48)})
    def db_creds(self):
        return json.loads(self.sm.get_secret_value(
            SecretId=C.DB_SECRET_ID)["SecretString"])
    def youtube_key(self):
        try:
            return json.loads(self.sm.get_secret_value(
                SecretId=C.YT_SECRET_ID)["SecretString"]).get("api_key", "")
        except Exception:
            return ""
    def web_session_key(self):
        return json.loads(self.sm.get_secret_value(
            SecretId=C.WEB_SECRET_ID)["SecretString"])["session_key"]
    def sync_llm_secret(self):
        """Sync CI JSON (preferred) or a local .env into Secrets Manager."""
        if self.args.skip_llm:
            return
        if self.args.llm_secrets_json:
            try:
                desired = json.loads(self.args.llm_secrets_json)
            except json.JSONDecodeError as exc:
                sys.exit(f"--llm-secrets-json is not valid JSON: {exc}")
            if not isinstance(desired, dict) or not desired:
                sys.exit("--llm-secrets-json must be a non-empty JSON object")
        else:
            env_path = Path(self.args.llm_dir) / ".env"
            if not env_path.exists():
                print(f"llm secret: no {env_path}; leaving {C.LLM_SECRET_ID} untouched")
                return
            desired = {}
            for ln in env_path.read_text().splitlines():
                ln = ln.strip()
                if ln and not ln.startswith("#") and "=" in ln:
                    k, _, v = ln.partition("=")
                    desired[k.strip()] = v.strip().strip('"').strip("'")
        try:
            current = json.loads(self.sm.get_secret_value(
                SecretId=C.LLM_SECRET_ID)["SecretString"])
        except self.sm.exceptions.ResourceNotFoundException:
            current = None
        if current == desired:
            print(f"llm secret: {C.LLM_SECRET_ID} unchanged ({len(desired)} keys)")
            return
        self._upsert_secret(C.LLM_SECRET_ID, desired)          # logs id only
        touched = sorted(set(desired) ^ set(current or {}) |
                        {k for k in desired
                        if current and current.get(k) != desired[k]})
        print(f"llm secret: synced {len(desired)} keys (changed: {', '.join(touched)})")
