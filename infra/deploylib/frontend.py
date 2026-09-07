"""
Frontend delivery on ECS.
A single nginx container serves the Vite build and reverse-proxies /api/* to the
EC2 API server on its VPC-internal address, so the browser talks to exactly one
origin (no CORS, no CloudFront -- which floci cannot fully emulate).
Image strategies (both valid on real AWS and on floci):
  ecr        docker build an nginx image with the site baked in + push to ECR.
             Preferred for production.
  bootstrap  run the public nginx image and let it pull a tarball of the build
             from S3 on container start.  No registry, no docker required.
  auto       try ecr, fall back to bootstrap.              (default)
The backend address is never baked into the image: nginx renders
/etc/nginx/templates/default.conf.template through envsubst at start-up using
API_BACKEND from the task definition, so a new EC2 private IP only needs a new
deployment, not a new image.
"""
import base64
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import socket
from pathlib import Path
from urllib.parse import urlparse
import botocore.exceptions
from . import config as C
from .awsutil import err_code, http_ok, ignore, matches, scan_local_ports, wait_for
# __PORT__ / __UPSTREAM__ are substituted here; ${API_BACKEND} is substituted by
# nginx's own envsubst entrypoint at container start.  $uri & friends survive
# because envsubst only touches variables that exist in the environment.
NGINX_TEMPLATE = r"""
server {
    listen       __PORT__;
    server_name  _;
    root         /usr/share/nginx/html;
    index        index.html;
    absolute_redirect off;
    gzip on;
    gzip_types text/plain text/css application/javascript application/json
               image/svg+xml;
    # deploy + load balancer probe
    location = __HEALTH__ {
        access_log off;
        add_header Content-Type text/plain;
        return 200 "ok\n";
    }
    # internal API: straight through to the EC2 API server
    location /api/ {
        proxy_pass            http://${API_BACKEND}__UPSTREAM__;
        proxy_http_version    1.1;
        proxy_set_header      Host $host;
        proxy_set_header      X-Real-IP $remote_addr;
        proxy_set_header      X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header      X-Forwarded-Proto $scheme;
        proxy_set_header      Upgrade $http_upgrade;
        proxy_set_header      Connection "";
        proxy_connect_timeout 5s;
        proxy_read_timeout    120s;
    }
    # Socket.IO is hosted by the web API. Without this priority route the SPA
    # fallback returns index.html (HTTP 200) instead of upgrading the socket.
    location ^~ /socket.io/ {
        # Cloudflared supplies the browser-facing scheme. Direct LAN requests
        # have no such header and use nginx's own scheme instead.
        set $socket_scheme $scheme;
        if ($http_x_forwarded_proto != "") { set $socket_scheme $http_x_forwarded_proto; }
        # No URI suffix here: Socket.IO must receive the /socket.io/ path even
        # when frontend_strip_api_prefix is enabled for ordinary /api calls.
        proxy_pass            http://${API_BACKEND};
        proxy_http_version    1.1;
        proxy_set_header      Host $host;
        proxy_set_header      X-Real-IP $remote_addr;
        proxy_set_header      X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header      X-Forwarded-Proto $socket_scheme;
        proxy_set_header      Upgrade $http_upgrade;
        proxy_set_header      Connection "upgrade";
        proxy_buffering       off;
        proxy_cache           off;
        proxy_connect_timeout 5s;
        proxy_read_timeout    3600s;
        proxy_send_timeout    3600s;
    }
    # ETL administration is deliberately LAN-only. cloudflared supplies these
    # headers even though its origin connection comes from localhost, so it is
    # rejected before the RFC1918 allow-list is evaluated.
    location = /admin {
        return 308 /admin/;
    }
    location /admin/ {
        if ($http_cf_connecting_ip != "") { return 403; }
        if ($http_cf_ray != "") { return 403; }
        allow 127.0.0.1;
        allow ::1;
        allow 10.0.0.0/8;
        allow 172.16.0.0/12;
        allow 192.168.0.0/16;
        deny all;
        proxy_pass            __ADMIN_UPSTREAM__;
        proxy_http_version    1.1;
        proxy_set_header      Host $host;
        proxy_set_header      X-Real-IP $remote_addr;
        proxy_set_header      X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header      X-Forwarded-Proto $scheme;
        proxy_connect_timeout 5s;
        proxy_read_timeout    120s;
    }
    # LLM chat server: strip /llm, no buffering (streaming responses)
    # ^~ prevents the global static-asset regex below from claiming chart PNGs.
    location ^~ /llm/ {
        proxy_pass            http://${LLM_BACKEND}/;
        proxy_http_version    1.1;
        proxy_set_header      Host $host;
        proxy_set_header      X-Real-IP $remote_addr;
        proxy_set_header      X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header      X-Forwarded-Proto $scheme;
        proxy_set_header      Upgrade $http_upgrade;
        proxy_set_header      Connection "";
        proxy_buffering       off;
        proxy_cache           off;
        proxy_connect_timeout 5s;
        proxy_read_timeout    300s;
    }
    # hashed assets: cache hard
    location ~* \.(?:js|mjs|css|woff2?|ttf|eot|png|jpe?g|gif|svg|ico|webp)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
        try_files $uri =404;
    }
    # SPA routing
    location / {
        try_files $uri $uri/ /index.html;
        add_header Cache-Control "no-cache";
    }
}
"""
# Bootstrap mode container command: pull the site + nginx template from S3, then
# hand over to the stock nginx entrypoint (which runs envsubst for us).
BOOTSTRAP_CMD = (
    "set -e; "
    "echo \"[bootstrap] fetching $SITE_URL\"; "
    "wget -O /tmp/site.tgz \"$SITE_URL\"; "
    "rm -rf /tmp/site; mkdir -p /tmp/site; tar -xzf /tmp/site.tgz -C /tmp/site; "
    "rm -rf /usr/share/nginx/html/*; "
    "cp -R /tmp/site/site/. /usr/share/nginx/html/; "
    "test -f /usr/share/nginx/html/index.html "
    "  || { echo '[bootstrap] FATAL: no index.html in bundle'; ls -R /tmp/site; exit 1; }; "
    "mkdir -p /etc/nginx/templates; "
    "cp /tmp/site/default.conf.template /etc/nginx/templates/default.conf.template; "
    "echo \"[bootstrap] site installed, API_BACKEND=$API_BACKEND\"; "
    "exec /docker-entrypoint.sh nginx -g 'daemon off;'"
)

def _resolvable(host):
    try:
        socket.getaddrinfo(host.partition(":")[0], None)
        return True
    except socket.gaierror:
        return False

def _port_open(host, port, timeout=1.0):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0

class FrontendMixin:
    # =======================================================  build  =========
    def build_frontend(self):
        """npm install + vite build -> frontend/dist.  False if there is no app."""
        src = Path(self.args.frontend_dir)
        if not (src / "package.json").exists():
            print("frontend: no package.json, skipping")
            return False
        # The SPA talks to its own origin; nginx proxies /api to the EC2 backend.
        env = {"VITE_API_BASE_URL": self.args.frontend_api_base,
               "VITE_API_BASE": self.args.frontend_api_base,
               "VITE_ERI_API_URL": self.args.frontend_llm_base}
        print("frontend: building ...")
        if self.args.build_in_docker:
            docker_env = [a for k, v in env.items() for a in ("-e", f"{k}={v}")]
            build_command = ("(npm ci --prefer-offline --no-audit || npm install) "
                             "&& npx vite build")
            if hasattr(os, "getuid"):
                owner = f"{os.getuid()}:{os.getgid()}"
                build_command = (
                    f"trap 'chown -R {owner} /app/dist 2>/dev/null || true; "
                    f"chown {owner} /app/package-lock.json 2>/dev/null || true' "
                    f"EXIT; {build_command}")
            subprocess.check_call([
                "docker", "run", "--rm",
                "-v", f"{src}:/app",
                # node_modules lives in a volume: host (Windows/macOS) bind
                # mounts can't be rmdir'd cleanly by npm, and it's ~10x faster.
                "-v", f"{C.APP}-frontend-node_modules:/app/node_modules",
                "-w", "/app", *docker_env,
                f"node:{self.args.node_version}-slim", "bash", "-lc",
                build_command])
        else:
            npm = "npm.cmd" if os.name == "nt" else "npm"
            shell_env = {**os.environ, **env}
            try:
                subprocess.check_call([npm, "ci", "--no-audit"],
                                      cwd=str(src), env=shell_env)
            except subprocess.CalledProcessError:
                subprocess.check_call([npm, "install"], cwd=str(src),
                                      env=shell_env)
            subprocess.check_call([npm, "exec", "--", "vite", "build"],
                                  cwd=str(src), env=shell_env)
        dist = src / "dist"
        if not (dist / "index.html").exists():
            sys.exit("frontend build produced no dist/index.html")
        print(f"frontend: built "
              f"{sum(1 for p in dist.rglob('*') if p.is_file())} files")
        return True
    def _nginx_conf(self):
        upstream = "/" if self.args.frontend_strip_api_prefix else ""
        return (NGINX_TEMPLATE
                .replace("__PORT__", str(self.frontend_port))
                .replace("__HEALTH__", C.FRONTEND_HEALTH_PATH)
                .replace("__UPSTREAM__", upstream)
                .replace("__ADMIN_UPSTREAM__", self._admin_upstream()))
    def _admin_upstream(self):
        """Admin API URL as seen from the frontend container."""
        url = self.get_param(f"/{C.APP}/admin/url")
        if not url:
            print("frontend: admin URL unavailable; /admin will return 502")
            return "http://127.0.0.1:9/"
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            sys.exit(f"frontend: invalid cached admin URL: {url!r}")
        if "$" in parsed.path:
            # Older deployments preferred API Gateway v2's `$default` stage.
            # Repair that cached SSM value from the already-provisioned v1 API
            # so code-only deploys also recover without a manual parameter edit.
            print("frontend: replacing cached $default admin URL with REST "
                  "API stage 'admin' ...")
            api = next((a for a in self.apigw_v1.get_rest_apis().get("items", [])
                        if a.get("name") == f"{C.APP}-admin-rest"), None)
            if api:
                url = self.resolve_admin_url(
                    {"id": api["id"], "stages": ["admin"]}, None)
                parsed = urlparse(url or "")
            if not url or "$" in parsed.path or not parsed.hostname:
                sys.exit("frontend: could not resolve the REST API 'admin' "
                         "stage to an nginx-safe URL")
        host = C.DOCKER_HOST_ALIAS if self.emulated else parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        path = parsed.path.rstrip("/") + "/"
        return f"{parsed.scheme}://{host}:{port}{path}"
    def _llm_authority(self):
        if self.args.skip_llm:
            return None
        if self.emulated:
            port = self.get_param(f"/{C.APP}/llm/host_port")
            return f"{C.DOCKER_HOST_ALIAS}:{port}" if port else None
        url = self.get_param(f"/{C.APP}/llm/internal_url")
        return urlparse(url).netloc if url else None
    # =====================================================  publish  =========
    def publish_frontend(self):
        """Returns (image, extra_container_env)."""
        mode = self.args.frontend_image_mode
        if mode in ("auto", "ecr"):
            image = self._push_frontend_image()
            if image:
                return image, {}
            if mode == "ecr":
                sys.exit("frontend: --frontend-image-mode ecr requested but the "
                         "build/push failed (see above)")
            print("frontend: falling back to bootstrap mode "
                  "(public nginx image + S3 tarball)")
        return C.FRONTEND_BASE_IMAGE, {"SITE_URL": self._upload_site_tarball()}
    # -- ECR -----------------------------------------------------------------
    def _ensure_ecr_repo(self):
        try:
            repos = self._api_call(self.ecr.describe_repositories,
                                   repositoryNames=[C.FRONTEND_ECR_REPO]
                                   )["repositories"]
            return repos[0]["repositoryUri"]
        except botocore.exceptions.ClientError as e:
            if not matches(e, "RepositoryNotFound"):
                raise
        repo = self._api_call(self.ecr.create_repository,
                              repositoryName=C.FRONTEND_ECR_REPO,
                              imageScanningConfiguration={"scanOnPush": False}
                              )["repository"]
        print(f"created ECR repository {C.FRONTEND_ECR_REPO}")
        return repo["repositoryUri"]
    def _push_ref(self, repo_uri):
        """Rewrite floci's virtual registry hostname (e.g.
        000000000000.dkr.ecr.us-east-1.localhost:5100) to a host-reachable
        name.  localhost registries are implicitly insecure for docker, so no
        daemon config is needed.  No-op on real AWS."""
        host, _, path = repo_uri.partition("/")
        if self.args.ecr_push_host:
            return f"{self.args.ecr_push_host}/{path}"
        if self.emulated and not _resolvable(host):
            port = host.rpartition(":")[2] if ":" in host else "443"
            print(f"frontend: registry host {host} not resolvable; "
                  f"pushing via localhost:{port}")
            return f"127.0.0.1:{port}/{path}"
        return repo_uri
    def _image_context(self):
        ctx = C.BUILD_DIR / "frontend-image"
        if ctx.exists():
            shutil.rmtree(ctx)
        ctx.mkdir(parents=True)
        shutil.copytree(Path(self.args.frontend_dir) / "dist", ctx / "site")
        (ctx / "default.conf.template").write_text(self._nginx_conf())
        (ctx / "Dockerfile").write_text(
            f"FROM {C.FRONTEND_BASE_IMAGE}\n"
            "COPY site/ /usr/share/nginx/html/\n"
            "COPY default.conf.template "
            "/etc/nginx/templates/default.conf.template\n"
            f"EXPOSE {self.frontend_port}\n")
        return ctx
    def _push_frontend_image(self):
        if not shutil.which("docker"):
            print("frontend: docker not on PATH, cannot build an image")
            return None
        try:
            repo_uri = self._push_ref(self._ensure_ecr_repo())
            auth = self._api_call(self.ecr.get_authorization_token
                                  )["authorizationData"][0]
        except botocore.exceptions.ClientError as e:
            print(f"frontend: ECR unavailable ({err_code(e) or e})")
            return None
        except Exception as e:
            print(f"frontend: ECR unusable ({e})")
            return None
        registry = repo_uri.partition("/")[0]
        host, _, port = registry.partition(":")
        if self.emulated and not _port_open(host, int(port or 443)):
            print(f"frontend: ECR registry {registry} is not listening -- "
                  f"floci's floci-ecr-registry sidecar is not running "
                  f"(`docker ps -a --filter name=ecr`, `docker logs floci`). "
                  f"Do not publish 5100-5199 in docker-compose.yml; they are "
                  f"pre-allocated for the sidecar.")
            return None
        user, _, password = base64.b64decode(
            auth["authorizationToken"]).decode().partition(":")
        tag = f"{repo_uri}:{self.args.frontend_tag or int(time.time())}"
        latest = f"{repo_uri}:latest"
        ctx = self._image_context()
        try:
            subprocess.run(["docker", "login", "-u", user,
                            "--password-stdin", registry],
                           input=password.encode(), check=True)
            subprocess.check_call(["docker", "build", "-t", tag, "-t", latest,
                                   str(ctx)])
            for ref in (tag, latest):
                subprocess.check_call(["docker", "push", ref])
        except subprocess.CalledProcessError as e:
            print(f"frontend: docker build/push failed ({e})")
            return None
        print(f"frontend: pushed {tag}")
        return tag
    # -- bootstrap tarball ---------------------------------------------------
    def _upload_site_tarball(self):
        bucket, key = C.BUCKETS["frontend"], C.FRONTEND_SITE_KEY
        self.ensure_bucket(bucket)
        tar_path = C.BUILD_DIR / "frontend-site.tar.gz"
        tar_path.parent.mkdir(parents=True, exist_ok=True)
        conf_path = C.BUILD_DIR / "default.conf.template"
        conf_path.write_text(self._nginx_conf())
        with tarfile.open(tar_path, "w:gz") as tf:
            tf.add(Path(self.args.frontend_dir) / "dist", arcname="site")
            tf.add(conf_path, arcname="default.conf.template")
        self.s3.upload_file(str(tar_path), bucket, key,
                            ExtraArgs={"ContentType": "application/gzip"})
        print(f"frontend: uploaded site bundle "
              f"({tar_path.stat().st_size / 1e6:.1f} MB) to s3://{bucket}/{key}")
        if self.emulated:
            # Path-style URL on the in-network edge; anonymous GET.
            self.make_bucket_public(bucket)
            base = self.internal_endpoint() or "http://localhost:4566"
            return f"{base}/{bucket}/{key}"
        # Real AWS: presigned, so the bucket stays private.  Regenerated on each
        # deploy; use --frontend-image-mode ecr for long-lived prod services.
        return self.s3.generate_presigned_url(
            "get_object", Params={"Bucket": bucket, "Key": key},
            ExpiresIn=7 * 86400)
    # =========================================================  ECS  =========
    def ensure_frontend_service(self):
        backend = self._backend_authority()
        image, extra_env = self.publish_frontend()
        self._ensure_cluster()
        plan = self._launch_plan()
        task_arn = self._register_frontend_task(image, backend, extra_env, plan)
        
        target_group, lb_dns = (None, None)
        if self.args.frontend_alb and plan["subnets"]:
            lb_dns, target_group = self._ensure_alb(plan)
        before = scan_local_ports(*C.PORT_SCAN_RANGE)
        self._ensure_service(task_arn, plan, target_group)
        self._report_frontend(backend, lb_dns, before)
    def _backend_authority(self):
        """host:port of the EC2 API server, as reachable *from the task*."""
        if self.args.frontend_api_backend:
            return self.args.frontend_api_backend
        if self.emulated:
            # floci attaches bridge-mode ECS tasks to docker0 (nginx logs the
            # client as 172.17.0.1), not FLOCI_SERVICES_DOCKER_NETWORK, and
            # docker does not route between bridges -- so the instance's
            # 172.18.x address is dead from here.  Use the host-published port
            # from floci's socat sidecar instead.
            host_port = self.get_param(f"/{C.APP}/web/host_port")
            if host_port:
                return f"{C.DOCKER_HOST_ALIAS}:{host_port}"
            print("frontend: no /web/host_port in SSM; falling back to the "
                  "internal address (will not be routable in bridge mode)")
        for key in ("internal_url", "url"):
            url = self.get_param(f"/{C.APP}/web/{key}")
            if url:
                parsed = urlparse(url)
                return f"{parsed.hostname}:{parsed.port or self.web_port}"
        sys.exit("frontend: no API backend address -- deploy the web API first "
                 "(drop --skip-web) or pass --frontend-api-backend host:port")
    def _ensure_cluster(self):
        ignore(self.ecs.create_cluster, clusterName=C.ECS_CLUSTER)
        if not self.emulated:
            ignore(self.logs.create_log_group,
                   logGroupName=C.FRONTEND_LOG_GROUP,
                   only=("ResourceAlreadyExists",))
    def _launch_plan(self):
        """Launch type, network mode and networking for the task/service."""
        launch_type = self.args.frontend_launch_type.upper()
        vpc_id, subnets = self.default_network()
        if self.emulated and launch_type == "FARGATE":
            # floci runs awsvpc tasks as bare containers on the compose network:
            # no ENI, no published port, nothing reachable from the host.
            # bridge + a static hostPort is the only shape it publishes.
            print("frontend: emulator detected -- using EC2/bridge networking "
                  "so the port can be published")
            launch_type = "EC2"
        sg_id = (self.ensure_security_group(
            f"{C.APP}-frontend-sg", [self.frontend_port],
            f"{C.APP} frontend HTTP", vpc_id=vpc_id) if subnets else None)
        if launch_type == "FARGATE" and not subnets:
            print("frontend: no subnets discoverable; using EC2/bridge networking")
            launch_type = "EC2"
        awsvpc = launch_type == "FARGATE"
        network_config = ({"awsvpcConfiguration": {
            "subnets": subnets[:3],
            "securityGroups": [sg_id] if sg_id else [],
            "assignPublicIp": "ENABLED" if self.args.frontend_public_ip
                              else "DISABLED"}} if awsvpc else None)
        print(f"frontend: launch={launch_type} network="
              f"{'awsvpc' if awsvpc else 'bridge'} "
              f"publish={self.frontend_host_port}->{self.frontend_port}")
        return {"launch_type": launch_type,
                "network_mode": "awsvpc" if awsvpc else "bridge",
                "network_config": network_config,
                "vpc_id": vpc_id, "subnets": subnets, "sg_id": sg_id}

    def _tasks(self, desired="RUNNING"):
        return (ignore(self.ecs.list_tasks, cluster=C.ECS_CLUSTER,
                       serviceName=C.FRONTEND_NAME,
                       desiredStatus=desired) or {}).get("taskArns", [])
    def _describe(self, arns):
        if not arns:
            return []
        return (ignore(self.ecs.describe_tasks, cluster=C.ECS_CLUSTER,
                       tasks=list(arns)) or {}).get("tasks", [])
    def _stale_tasks(self, task_arn):
        return [t for t in self._describe(self._tasks())
                if t.get("taskDefinitionArn") != task_arn]

    def _stop_orphaned_emulator_frontends(self):
        """Stop only Floci ECS web containers publishing our fixed host port.

        Some Floci releases lose a task from list_tasks while its Docker
        container remains alive. ECS can no longer stop that orphan, but it
        still owns :80 and makes every reconciliation attempt fail.
        """
        result = subprocess.run(
            ["docker", "ps", "--filter",
             f"publish={self.frontend_host_port}",
             "--format", "{{.ID}}\t{{.Names}}"],
            text=True, capture_output=True, check=False)
        targets = []
        for line in result.stdout.splitlines():
            parts = line.split("\t", 1)
            if len(parts) == 2 and parts[1].startswith("floci-ecs-") \
                    and parts[1].endswith(f"-{C.FRONTEND_CONTAINER}"):
                targets.append(parts[0])
        if targets:
            subprocess.run(["docker", "stop", *targets], check=True)
            print(f"frontend: stopped {len(targets)} orphaned Floci "
                  "container(s) holding the static host port")
    
    def _replace_tasks(self, task_arn):
        """floci records the new task definition but never recycles the
        container, and list_tasks is briefly empty after update_service."""
        stale = []
        for _ in range(5):
            stale = self._stale_tasks(task_arn)
            if stale:
                break
            time.sleep(2)
        for t in stale:
            ignore(self.ecs.stop_task, cluster=C.ECS_CLUSTER,
                   task=t["taskArn"], reason="deploy: new task definition")
        if stale:
            print(f"frontend: stopped {len(stale)} task(s) on an old revision")
            wait_for(lambda: not self._stale_tasks(task_arn),
                     timeout=90, interval=3)
            
    def _running_task(self, task_arn=None):
        for task in self._describe(self._tasks()):
            if task.get("lastStatus") != "RUNNING":
                continue
            if task_arn and task.get("taskDefinitionArn") != task_arn:
                continue
            return task
        return None
    def _register_frontend_task(self, image, backend, extra_env, plan):
        env = {"API_BACKEND": backend, **extra_env}
        port_mapping = {"containerPort": self.frontend_port, "protocol": "tcp"}
        llm = self._llm_authority()
        env["LLM_BACKEND"] = llm or "127.0.0.1:9"
        if plan["network_mode"] == "bridge":
            # A static hostPort is what actually gets published; without it
            # floci exposes the port but never maps it ("80/tcp" in docker ps).
            port_mapping["hostPort"] = self.frontend_host_port
        container = {
            "name": C.FRONTEND_CONTAINER,
            "image": image,
            "essential": True,
            "portMappings": [port_mapping],
            "environment": [{"name": k, "value": str(v)} for k, v in env.items()],
        }
        if "SITE_URL" in extra_env:
            container["entryPoint"] = ["/bin/sh", "-c"]
            container["command"] = [BOOTSTRAP_CMD]
        if self.emulated and plan["network_mode"] == "bridge":
            container["extraHosts"] = [{"hostname": C.DOCKER_HOST_ALIAS,
                                        "ipAddress": "host-gateway"}]
        task = dict(family=C.FRONTEND_NAME,
                    containerDefinitions=[container],
                    networkMode=plan["network_mode"],
                    requiresCompatibilities=[plan["launch_type"]],
                    cpu=str(self.args.frontend_cpu),
                    memory=str(self.args.frontend_memory))
        try:
            resp = self._api_call(self.ecs.register_task_definition, **task)
        except botocore.exceptions.ClientError as e:
            if not container.pop("extraHosts", None):
                raise
            print(f"frontend: extraHosts rejected ({err_code(e) or e}); relying "
                  f"on the runtime's own {C.DOCKER_HOST_ALIAS} resolution")
            resp = self._api_call(self.ecs.register_task_definition, **task)
        if not self.emulated:
            # awslogs + container health checks are real-AWS only: emulators
            # frequently reject or silently break on them.
            container["logConfiguration"] = {
                "logDriver": "awslogs",
                "options": {"awslogs-group": C.FRONTEND_LOG_GROUP,
                            "awslogs-region": self.args.region,
                            "awslogs-stream-prefix": "nginx",
                            "awslogs-create-group": "true"}}
            container["healthCheck"] = {
                "command": ["CMD-SHELL",
                            f"wget -qO- http://127.0.0.1:{self.frontend_port}"
                            f"{C.FRONTEND_HEALTH_PATH} || exit 1"],
                "interval": 30, "timeout": 5, "retries": 3, "startPeriod": 15}
            task["executionRoleArn"] = self.ensure_role(
                f"{C.APP}-ecs-exec-role", "ecs-tasks.amazonaws.com",
                managed=[C.ECS_TASK_EXEC_POLICY])
        else:
            task["executionRoleArn"] = self.ensure_role(
                f"{C.APP}-ecs-exec-role", "ecs-tasks.amazonaws.com",
                managed=[C.ECS_TASK_EXEC_POLICY], inline=C.WILDCARD_POLICY)
        resp = self._api_call(self.ecs.register_task_definition, **task)
        stored = (resp["taskDefinition"].get("containerDefinitions") or
                  [{}])[0].get("portMappings")
        arn = resp["taskDefinition"].get("taskDefinitionArn") or C.FRONTEND_NAME
        print(f"frontend: registered task definition "
              f"{C.FRONTEND_NAME}:{resp['taskDefinition'].get('revision', '?')} "
              f"networkMode={resp['taskDefinition'].get('networkMode')} "
              f"portMappings={stored}")
        return arn
    def _find_service(self):
        try:
            services = self._api_call(
                self.ecs.describe_services, cluster=C.ECS_CLUSTER,
                services=[C.FRONTEND_NAME]).get("services", [])
        except botocore.exceptions.ClientError:
            return None
        return next((s for s in services
                     if s.get("status") in ("ACTIVE", "DRAINING")), None)
    def _ensure_service(self, task_arn, plan, target_group):
        count = self.args.frontend_count
        if self._find_service():
            if self.emulated and plan["network_mode"] == "bridge":
                # A fixed host port cannot support ECS rolling replacement.
                # Scale the old revision to zero first; otherwise Floci keeps
                # creating doomed tasks forever while the healthy old task
                # still owns :80.
                self._api_call(self.ecs.update_service,
                               cluster=C.ECS_CLUSTER,
                               service=C.FRONTEND_NAME,
                               desiredCount=0)
                for t in self._describe(self._tasks()):
                    ignore(self.ecs.stop_task, cluster=C.ECS_CLUSTER,
                           task=t["taskArn"], reason="deploy: free static hostPort")
                wait_for(lambda: not _port_open(
                    "127.0.0.1", self.frontend_host_port),
                    timeout=30, interval=2)
                if _port_open("127.0.0.1", self.frontend_host_port):
                    self._stop_orphaned_emulator_frontends()
                    wait_for(lambda: not _port_open(
                        "127.0.0.1", self.frontend_host_port),
                        timeout=30, interval=2)
                if _port_open("127.0.0.1", self.frontend_host_port):
                    sys.exit(f"frontend: host port {self.frontend_host_port} "
                             "is still occupied after scaling the old service "
                             "to zero")
                time.sleep(2)
            kwargs = dict(cluster=C.ECS_CLUSTER, service=C.FRONTEND_NAME,
                          taskDefinition=task_arn, desiredCount=count,
                          forceNewDeployment=True)
            if plan["network_config"]:
                kwargs["networkConfiguration"] = plan["network_config"]
            try:
                self._api_call(self.ecs.update_service, **kwargs)

            except botocore.exceptions.ClientError as e:
                if not matches(e, "forceNewDeployment", "ValidationException"):
                    raise
                kwargs.pop("forceNewDeployment", None)
                self._api_call(self.ecs.update_service, **kwargs)
            print(f"frontend: redeployed service {C.FRONTEND_NAME} "
                  f"(desired {count})")
            self._replace_tasks(task_arn)
            return
        kwargs = dict(cluster=C.ECS_CLUSTER, serviceName=C.FRONTEND_NAME,
                      taskDefinition=task_arn, desiredCount=count,
                      launchType=plan["launch_type"])
        if plan["network_config"]:
            kwargs["networkConfiguration"] = plan["network_config"]
        if target_group:
            kwargs["loadBalancers"] = [{
                "targetGroupArn": target_group,
                "containerName": C.FRONTEND_CONTAINER,
                "containerPort": self.frontend_port}]
            kwargs["healthCheckGracePeriodSeconds"] = 60
        self._api_call(self.ecs.create_service, **kwargs)
        print(f"frontend: redeployed service {C.FRONTEND_NAME} "
                  f"(desired {count})")
        self._replace_tasks(task_arn)
        return
    # -- optional ALB (real AWS production front door) ------------------------
    def _ensure_alb(self, plan):
        name = C.FRONTEND_NAME[:32]
        try:
            lbs = (ignore(self.elbv2.describe_load_balancers, Names=[name]) or
                   {}).get("LoadBalancers", [])
            lb = lbs[0] if lbs else self._api_call(
                self.elbv2.create_load_balancer, Name=name,
                Subnets=plan["subnets"][:3],
                SecurityGroups=[plan["sg_id"]] if plan["sg_id"] else [],
                Scheme="internet-facing", Type="application"
            )["LoadBalancers"][0]
            tgs = (ignore(self.elbv2.describe_target_groups, Names=[name]) or
                   {}).get("TargetGroups", [])
            tg = tgs[0] if tgs else self._api_call(
                self.elbv2.create_target_group, Name=name, Protocol="HTTP",
                Port=self.frontend_port, VpcId=plan["vpc_id"],
                TargetType="ip" if plan["network_mode"] == "awsvpc" else "instance",
                HealthCheckPath=C.FRONTEND_HEALTH_PATH,
                Matcher={"HttpCode": "200"})["TargetGroups"][0]
            listeners = self._api_call(
                self.elbv2.describe_listeners,
                LoadBalancerArn=lb["LoadBalancerArn"]).get("Listeners", [])
            if not any(l.get("Port") == 80 for l in listeners):
                self._api_call(
                    self.elbv2.create_listener,
                    LoadBalancerArn=lb["LoadBalancerArn"], Protocol="HTTP",
                    Port=80, DefaultActions=[
                        {"Type": "forward",
                         "TargetGroupArn": tg["TargetGroupArn"]}])
            print(f"frontend: ALB {lb['DNSName']}")
            return lb["DNSName"], tg["TargetGroupArn"]
        except botocore.exceptions.ClientError as e:
            print(f"frontend: ALB unavailable ({err_code(e) or e}); "
                  f"serving the task directly")
            return None, None
    # -- URL discovery / reporting -------------------------------------------
    def _task_address(self, task):
        eni = next((d["value"] for a in task.get("attachments", [])
                    for d in a.get("details", [])
                    if d.get("name") == "networkInterfaceId"), None)
        if eni:
            nets = (ignore(self.ec2.describe_network_interfaces,
                           NetworkInterfaceIds=[eni]) or
                    {}).get("NetworkInterfaces", [])
            if nets:
                return (nets[0].get("Association", {}).get("PublicIp") or
                        nets[0].get("PrivateIpAddress"))
        for container in task.get("containers", []):
            for net in container.get("networkInterfaces", []):
                if net.get("privateIpv4Address"):
                    return net["privateIpv4Address"]
        return None
    def _report_frontend(self, backend, lb_dns, ports_before, task_arn=None):
        task = wait_for(lambda: self._running_task(task_arn), timeout=240,
                        interval=4,
                        report=lambda: print("  waiting for the frontend task "
                                             "to reach RUNNING ..."))
        if not task:
            print("  WARNING: no RUNNING frontend task yet; check "
                  f"`aws ecs describe-services --cluster {C.ECS_CLUSTER} "
                  f"--services {C.FRONTEND_NAME}`")
        url = None
        if lb_dns:
            url = f"http://{lb_dns}"
        elif self.emulated:
            url = None
            for host in ("127.0.0.1", "localhost"):
                probe = f"http://{host}:{self.frontend_host_port}"
                if wait_for(lambda: http_ok(f"{probe}{C.FRONTEND_HEALTH_PATH}"),
                            timeout=60, interval=3):
                    url = probe
                    break
                print(f"  {probe}{C.FRONTEND_HEALTH_PATH} did not answer")
            if not url:
                port = self.discover_forwarded_port(
                    ports_before, probe=C.FRONTEND_HEALTH_PATH, timeout=60)
                url = f"http://localhost:{port}" if port else None
        elif task:
            address = self._task_address(task)
            if address:
                url = f"http://{address}"
        healthy = bool(url) and http_ok(f"{url}{C.FRONTEND_HEALTH_PATH}")
        if url:
            self.put_params({f"/{C.APP}/frontend/url": url,
                             f"/{C.APP}/frontend/api_backend": backend,
                             f"/{C.APP}/frontend/service": C.FRONTEND_NAME,
                             f"/{C.APP}/frontend/cluster": C.ECS_CLUSTER})
        if url and healthy:
            self.summary.append(("Frontend", url))
        else:
            print(f"frontend: not answering yet at {url or '(no port published)'}; "
                  f"check `docker logs` of the floci-ecs-* container "
                  f"(/api/* -> {backend})")
