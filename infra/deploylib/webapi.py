"""The API server: a single EC2 instance running gunicorn on WEB_PORT.
The frontend container reaches it on the VPC-internal address published to
SSM at /{APP}/web/internal_url.
"""
import base64
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys
import tarfile
import time
import re
import shlex
import botocore.exceptions
from . import config as C
from .awsutil import http_ok, scan_local_ports, wait_for
class WebApiMixin:
    # ------------------------------------------------------ instance state ---
    def _find_web_instance(self):
        resp = self._api_call(self.ec2.describe_instances, Filters=[
            {"Name": "tag:Name", "Values": [C.WEB_INSTANCE_NAME]},
            {"Name": "instance-state-name", "Values": ["running", "pending"]}])
        for reservation in resp.get("Reservations", []):
            for inst in reservation.get("Instances", []):
                return inst
        return None
    def _terminate_web_instance(self):
        old = self._find_web_instance()
        if not old:
            return
        print(f"terminating previous web instance {old['InstanceId']} ...")
        self._api_call(self.ec2.terminate_instances,
                       InstanceIds=[old["InstanceId"]])
        wait_for(lambda: self._find_web_instance() is None, timeout=60)
        old_ports = [self.get_param(f"/{C.APP}/{service}/host_port")
                     for service in ("web", "llm")]
        for old_port in filter(None, old_ports):
            wait_for(lambda p=int(old_port): p not in
                     scan_local_ports(*C.PORT_SCAN_RANGE), timeout=30)
        time.sleep(2)
    # ------------------------------------------------------------ artifacts ---
    def build_web_bundle(self):
        """Build the venv (cached in S3) and the code tarball (always)."""
        cached = False
        if not self.args.rebuild_web_deps:
            try:
                self.s3.head_object(Bucket=C.BUCKETS["config"],
                                    Key=C.WEB_VENV_KEY)
                cached = True
                print("web venv already in S3 (--rebuild-web-deps to force)")
            except botocore.exceptions.ClientError:
                pass
        if not cached:
            self._build_web_venv()
        self.upload_web_code()
    def _build_web_venv(self):
        """pip-install the web deps inside Amazon Linux (matching the EC2
        runtime), then upload venv + model cache to S3."""
        print("building web venv in an Amazon Linux container "
              "(several minutes the first time) ...")
        staging = C.BUILD_DIR / "web-venv-staging"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        ownership_trap = ""
        if hasattr(os, "getuid"):
            owner = f"{os.getuid()}:{os.getgid()}"
            ownership_trap = (f"trap 'chown -R {owner} /output "
                              f"2>/dev/null || true' EXIT")
        script = rf"""
set -ex
{ownership_trap}
dnf install -y --allowerasing {C.WEB_PYTHON} {C.WEB_PYTHON}-pip {C.WEB_PYTHON}-devel \
               gcc gcc-c++ postgresql-devel tar gzip findutils
{C.WEB_PYTHON} -m venv /opt/web/venv
source /opt/web/venv/bin/activate
pip install --no-cache-dir --upgrade pip wheel
# CPU-only PyTorch, shared by the API server and the LLM service
pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
# one resolve over both requirement sets: a conflict fails here, not at import
pip install --no-cache-dir -r /src/web/requirements.txt \
                           -r /src/llm_chat/requirements.txt
pip install --no-cache-dir gunicorn "uvicorn[standard]"
python -c "
from sentence_transformers import SentenceTransformer
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
"
mkdir -p /opt/web/model_cache
cp -r /root/.cache/huggingface /opt/web/model_cache/
tar -czf /output/web-venv.tar.gz -C /opt/web venv model_cache
"""
        subprocess.check_call([
            "docker", "run", "--rm",
            "-v", f"{C.ROOT / 'web'!s}:/src/web:ro",
            "-v", f"{self.args.llm_dir!s}:/src/llm_chat:ro",
            "-v", f"{staging!s}:/output",
            "public.ecr.aws/amazonlinux/amazonlinux:2023",
            "bash", "-c", script])
        tar_path = staging / "web-venv.tar.gz"
        print(f"web venv: {tar_path.stat().st_size / 1e6:.1f} MB -> "
              f"s3://{C.BUCKETS['config']}/{C.WEB_VENV_KEY}")
        self.s3.upload_file(str(tar_path), C.BUCKETS["config"], C.WEB_VENV_KEY)
    def _upload_code_tree(self, src_dir, key, skip):
        tar_path = C.BUILD_DIR / Path(key).name
        tar_path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path, "w:gz") as tf:
            for f in sorted(Path(src_dir).rglob("*")):
                rel = f.relative_to(src_dir)
                if f.is_file() and not any(s in rel.parts or s in f.name
                                           or f.suffix == s for s in skip):
                    tf.add(f, arcname=rel.as_posix())
        self.s3.upload_file(str(tar_path), C.BUCKETS["config"], key)
        print(f"uploaded {Path(src_dir).name} code "
              f"({tar_path.stat().st_size / 1024:.0f} KB) -> "
              f"s3://{C.BUCKETS['config']}/{key}")
    def upload_web_code(self):
        self._upload_code_tree(
            C.ROOT / "web", C.WEB_CODE_KEY,
            skip={"__pycache__", ".pyc", "usage.db", ".env", "entrypoint.sh"})
        if not self.args.skip_llm:
            # .env is replaced by exported vars; generated_charts is runtime
            # output; Dockerfile is irrelevant on EC2.
            self._upload_code_tree(
                self.args.llm_dir, C.LLM_CODE_KEY,
                skip={"__pycache__", ".pyc", ".env", "Dockerfile",
                      "generated_charts"})
    # ------------------------------------------------------------- UserData ---
    def _web_userdata(self):
        creds = self.db_creds()
        ec_host, ec_port = self.cache_endpoint()
        s3_base = self.internal_endpoint() or "http://localhost:4566"
        bucket = C.BUCKETS["config"]
        venv_url = f"{s3_base}/{bucket}/{C.WEB_VENV_KEY}"
        code_url = f"{s3_base}/{bucket}/{C.WEB_CODE_KEY}"
        llm_url = f"{s3_base}/{bucket}/{C.LLM_CODE_KEY}"
        llm = not self.args.skip_llm
        llm_exports = ""
        
        # No systemd in the AL2023 minimal guest -- export env and nohup
        # gunicorn.  PID 1 is `tail -f /dev/null`, so the background process
        # survives the script exiting.
        db_password = shlex.quote(creds["password"])
        web_session_key = shlex.quote(self.web_session_key())
        youtube_key = shlex.quote(self.youtube_key())
        return f"""#!/bin/bash

set -ex
exec > /var/log/web-api-init.log 2>&1
echo "=== web-api UserData starting $(date) ==="
# The venv is built against {C.WEB_PYTHON}; the AL2023 guest ships only
# python3.9, so its interpreter symlink + console-script shebangs dangle
# until the matching runtime (and libpython) are present.
dnf install -y {C.WEB_PYTHON} {C.WEB_PYTHON}-libs \
  || yum install -y {C.WEB_PYTHON} {C.WEB_PYTHON}-libs
mkdir -p /opt/web/app
python3 -c "
import urllib.request, sys, time
for url, dest in [
    ('{venv_url}', '/tmp/web-venv.tar.gz'),
    ('{code_url}', '/tmp/web-code.tar.gz'),
    {f"('{llm_url}', '/tmp/llm-code.tar.gz')," if llm else ""}
]:
    print(f'downloading {{url}} ...', flush=True)
    for attempt in range(10):
        try:
            urllib.request.urlretrieve(url, dest)
            print(f'  saved to {{dest}}', flush=True)
            break
        except Exception as e:
            print(f'  attempt {{attempt+1}} failed: {{e}}', flush=True)
            time.sleep(3)
    else:
        sys.exit(f'FATAL: could not download {{url}}')
"
python3 << 'PYEOF'
import tarfile, os
for archive, dest in [("/tmp/web-venv.tar.gz", "/opt/web"),
                      ("/tmp/web-code.tar.gz", "/opt/web/app"),
    {'("/tmp/llm-code.tar.gz", "/opt/web/llm"),' if llm else ""}]:
    os.makedirs(dest, exist_ok=True)
    print(f"extracting {{archive}} -> {{dest}}", flush=True)
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(dest)
    os.remove(archive)
PYEOF
export PATH="/opt/web/venv/bin:/usr/local/bin:/usr/bin:/bin"
export HF_HOME="/opt/web/model_cache/huggingface"
export TRANSFORMERS_CACHE="/opt/web/model_cache/huggingface"
export POSTGRES_USER='{creds["username"]}'
export POSTGRES_HOST='{creds["host"]}'
export POSTGRES_PORT='{creds.get("port", 5432)}'
export POSTGRES_DB='{creds["dbname"]}'
export REDIS_HOST='{ec_host}'
export REDIS_PORT='{ec_port}'
export REDIS_URL='redis://{ec_host}:{ec_port}/0'
export ELASTICACHE_HOST='{ec_host}'
export ELASTICACHE_PORT='{ec_port}'
export RATE_LIMIT_WINDOW='60'
export MAX_REQUESTS_PER_WINDOW='120'
export AWS_ENDPOINT_URL='{self.internal_endpoint() or "http://localhost:4566"}'
export AWS_REGION='{self.args.region}'
export AWS_DEFAULT_REGION='{self.args.region}'
export LLM_SECRET_ID='{C.LLM_SECRET_ID}'
set +x                       # keep credentials out of the init log
export AWS_ACCESS_KEY_ID='test'
export AWS_SECRET_ACCESS_KEY='test'
export POSTGRES_PASSWORD={db_password}
export YOUTUBE_API_KEY={youtube_key}
export SECRET_KEY={web_session_key}
set -x
cd /opt/web/app
nohup /opt/web/venv/bin/gunicorn \\
    --bind 0.0.0.0:{self.web_port} \\
    --timeout 120 \\
    --workers {self.args.web_workers} \\
    --threads 4 \\
    --access-logfile /var/log/web-api-access.log \\
    --error-logfile /var/log/web-api-error.log \\
    server:app &
GUNICORN_PID=$!
sleep 3
# ---- LLM chat server (FastAPI) -----------------------------------------
export WEB_API_URL='http://127.0.0.1:{self.web_port}/'
export LLM_PORT='{self.args.llm_port}'
{llm_exports}cd /opt/web/llm
mkdir -p generated_charts
nohup /opt/web/venv/bin/uvicorn main:app \\
    --host 0.0.0.0 --port {self.args.llm_port} \\
    --log-level info > /var/log/llm-server.log 2>&1 &
LLM_PID=$!
sleep 3
if kill -0 $LLM_PID 2>/dev/null; then
    echo "=== llm server running (PID $LLM_PID) ==="
    # Indexing can take many minutes on the production host. It is idempotent
    # and must not hold Floci's EC2 UserData/port-forwarding lifecycle open.
    # Start it after Uvicorn owns the service port so readiness is independent.
    nohup bash -c '
      python init_tool_store.py && python init_knowledge.py
    ' >> /var/log/llm-init.log 2>&1 &
    echo "=== llm indexing started in background (PID $!) ==="
else
    echo "=== FATAL: llm server exited immediately ==="
    tail -50 /var/log/llm-init.log /var/log/llm-server.log || true
    exit 1
fi
if kill -0 $GUNICORN_PID 2>/dev/null; then
    echo "=== web-api UserData done, gunicorn running (PID $GUNICORN_PID) ==="
else
    echo "=== FATAL: gunicorn exited immediately ==="
    cat /var/log/web-api-error.log 2>/dev/null || true
    exit 1
fi
"""
    # --------------------------------------------------------------- launch ---
    def ensure_web_instance(self):
        self._terminate_web_instance()
        vpc_id, _ = self.default_network()
        ports = [self.web_port] + ([self.args.llm_port]
                                   if not self.args.skip_llm else [])
        sg_id = self.ensure_security_group(
            f"{C.APP}-web-sg", ports, f"{C.APP} web API + LLM", vpc_id=vpc_id)
        # Snapshot listening ports before launch so we can spot floci's forwarder.
        before = scan_local_ports(*C.PORT_SCAN_RANGE)
        print(f"launching web instance (AMI={C.WEB_AMI}) ...")
        kwargs = dict(ImageId=C.WEB_AMI, InstanceType="t3.micro",
                      MinCount=1, MaxCount=1, UserData=self._web_userdata(),
                      TagSpecifications=[{"ResourceType": "instance", "Tags": [
                          {"Key": "Name", "Value": C.WEB_INSTANCE_NAME},
                          {"Key": "app", "Value": C.APP}]}])
        if sg_id:
            kwargs["SecurityGroupIds"] = [sg_id]
        instance_id = self._api_call(
            self.ec2.run_instances, **kwargs)["Instances"][0]["InstanceId"]
        print(f"  instance {instance_id}")
        state = {"name": None}
        def running():
            try:
                inst = self._api_call(self.ec2.describe_instances,
                                      InstanceIds=[instance_id]
                                      )["Reservations"][0]["Instances"][0]
            except (botocore.exceptions.ClientError, IndexError, KeyError):
                return None
            state["name"] = inst["State"]["Name"]
            return inst if state["name"] == "running" else None
        inst = wait_for(running, timeout=150,
                        report=lambda: print(f"  waiting: state={state['name']}"))
        if not inst:
            sys.exit("web instance never reached running state; refusing to "
                     "deploy the frontend with a stale backend address")
        private_ip = inst.get("PrivateIpAddress")
        public_ip = inst.get("PublicIpAddress")
        print(f"  public_ip={public_ip}  private_ip={private_ip}")
        # Real AWS: the public IP is routable.  Emulator: 127.0.0.1 plus a
        # random published host port.
        # Floci may report a bridge address as PublicIpAddress. It is not a
        # host-published endpoint and ECS tasks on docker0 cannot route to the
        # EC2 container's compose bridge. Emulator mode must always discover
        # the socat-forwarded host port, regardless of the reported address.
        if self.emulated:
            host_port = self.discover_forwarded_port(before, probe="/health",
                                                     timeout=300)
            if not host_port:
                self._print_instance_log(instance_id)
                sys.exit("web API has no healthy host-forwarded port; refusing "
                         "to publish a frontend with a stale backend address")
            external_host = "localhost"
        elif public_ip in (None, "", "127.0.0.1", "localhost"):
            sys.exit("web instance has no routable public address")
        else:
            external_host, host_port = public_ip, self.web_port
        llm_host_port = None
        if not self.args.skip_llm:
            if external_host == "localhost":
                llm_host_port = self.discover_forwarded_port(
                    before, probe=C.LLM_HEALTH_PATH, timeout=600,
                    exclude=[host_port])          # model load is slow
            else:
                llm_host_port = self.args.llm_port
        external_url = f"http://{external_host}:{host_port}"
        internal_url = f"http://{private_ip}:{self.web_port}"
        if not self._health_check(f"{external_url}/health", instance_id):
            sys.exit("web API health check failed; refusing to update its SSM "
                     "endpoint or deploy the frontend")
        params = {
            f"/{C.APP}/web/url": external_url,
            f"/{C.APP}/web/internal_url": internal_url,
            f"/{C.APP}/web/instance_id": instance_id,
            f"/{C.APP}/web/host_port": host_port,
        }
        if not self.args.skip_llm and not llm_host_port:
            self._print_instance_log(instance_id)
            sys.exit("LLM server has no healthy host-forwarded port; refusing "
                     "to retain or publish a stale LLM endpoint")
        if llm_host_port:
            llm_url = f"http://{external_host}:{llm_host_port}"
            params.update({
                f"/{C.APP}/llm/url": llm_url,
                f"/{C.APP}/llm/host_port": llm_host_port,
                f"/{C.APP}/llm/internal_url":
                    f"http://{private_ip}:{self.args.llm_port}"})
            self.summary.append(("LLM", llm_url))
        self.put_params(params)
    def _health_check(self, url, instance_id):
        print(f"  waiting for web service at {url} ...")
        for attempt in range(60):
            if http_ok(url):
                print("  health check passed")
                return True
            if attempt in (15, 30, 45):
                print("  checking init log ...")
                self._print_instance_log(instance_id)
            time.sleep(3)
        print(f"  WARNING: health check never passed at {url}")
        self._print_instance_log(instance_id)
        return False
    def _print_instance_log(self, instance_id):
        """Best-effort: console output (real AWS), then SSM RunCommand."""
        _SENSITIVE = re.compile(r"(PASSWORD|SECRET|TOKEN|API_?KEY|AWS_\w*KEY)", re.I)
        try:
            output = self._api_call(self.ec2.get_console_output,
                                    InstanceId=instance_id).get("Output", "")
            if output:
                try:
                    output = base64.b64decode(output).decode(errors="replace")
                except Exception:
                    pass
                for line in output.strip().splitlines()[-15:]:
                    print("    " + (_SENSITIVE.sub(r"\1=<redacted>",
                                    re.sub(r"(=).*", r"\1<redacted>", line))
                                    if _SENSITIVE.search(line) else line))
                return
        except botocore.exceptions.ClientError:
            pass
        try:
            cmd = self._api_call(
                self.ssm.send_command, InstanceIds=[instance_id],
                DocumentName="AWS-RunShellScript",
                Parameters={"commands": [
                    "cat /var/log/web-api-init.log 2>/dev/null; echo '---'; "
                    "cat /var/log/web-api-error.log 2>/dev/null; echo '---'; "
                    "cat /var/log/llm-init.log 2>/dev/null; echo '---'; "
                    "cat /var/log/llm-server.log 2>/dev/null"]})
            time.sleep(3)
            out = self._api_call(
                self.ssm.get_command_invocation,
                CommandId=cmd["Command"]["CommandId"],
                InstanceId=instance_id).get("StandardOutputContent", "")
            if out:
                for line in out.strip().splitlines()[-15:]:
                    print("    " + (_SENSITIVE.sub(r"\1=<redacted>",
                                    re.sub(r"(=).*", r"\1<redacted>", line))
                                    if _SENSITIVE.search(line) else line))
                return
        except botocore.exceptions.ClientError:
            pass
        print("    (could not retrieve logs via AWS APIs) check manually:")
        print("      ssh -p 2200 root@localhost cat /var/log/web-api-init.log")
