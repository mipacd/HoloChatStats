"""API Gateway: /status (HTTP API) and the admin page (HTTP + REST)."""
import contextlib
import io
import json
import botocore.exceptions
from admin_endpoint import discover as discover_admin_url
from . import config as C
from .awsutil import http_ok, ignore, matches
class ApiMixin:
    def ensure_status_api(self):
        name = f"{C.APP}-status"
        api = next((a for a in self.apigw.get_apis()["Items"]
                    if a["Name"] == name), None)
        if api is None:
            api = self.apigw.create_api(
                Name=name, ProtocolType="HTTP",
                Target=self.function_arns["status"], RouteKey="GET /status")
            self._allow_apigw(f"{C.APP}-status", "apigw-invoke")
        print(f"status endpoint: "
              f"{api.get('ApiEndpoint', '(see emulator routing)')}/status")
    def _allow_apigw(self, fn, statement_id, source_arn=None):
        kwargs = {"SourceArn": source_arn} if source_arn else {}
        try:
            self.lam.add_permission(
                FunctionName=fn, StatementId=statement_id,
                Action="lambda:InvokeFunction",
                Principal="apigateway.amazonaws.com", **kwargs)
        except self.lam.exceptions.ResourceConflictException:
            pass
        except botocore.exceptions.ClientError:
            # Emulator may reject SourceArn scoping; retry unscoped.
            if source_arn:
                self._allow_apigw(fn, f"{statement_id}-any")
    def ensure_admin_api(self):
        """HTTP API in front of chat-ingest-admin.  Reached on the emulator edge
        port (4566) -- never port 80."""
        name, fn_arn = f"{C.APP}-admin", self.function_arns["admin"]
        api = next((a for a in self.apigw.get_apis()["Items"]
                    if a["Name"] == name), None)
        if api is None:
            try:
                api = self.apigw.create_api(Name=name, ProtocolType="HTTP")
                integ = self.apigw.create_integration(
                    ApiId=api["ApiId"], IntegrationType="AWS_PROXY",
                    IntegrationUri=fn_arn, IntegrationMethod="POST",
                    PayloadFormatVersion="2.0")["IntegrationId"]
                for route in ("$default", "ANY /", "ANY /{proxy+}"):
                    ignore(self.apigw.create_route, ApiId=api["ApiId"],
                           RouteKey=route, Target=f"integrations/{integ}")
                ignore(self.apigw.create_stage, ApiId=api["ApiId"],
                       StageName="$default", AutoDeploy=True)
            except botocore.exceptions.ClientError as e:
                print(f"  explicit HTTP API creation rejected ({e}); "
                      f"falling back to quick-create")
                api = self.apigw.create_api(Name=name, ProtocolType="HTTP",
                                            Target=fn_arn, RouteKey="$default")
            self._allow_apigw(f"{C.APP}-admin", "apigw-invoke")
        stages = [s["StageName"] for s in
                  (ignore(self.apigw.get_stages, ApiId=api["ApiId"]) or
                   {}).get("Items", [])]
        return {"id": api["ApiId"], "stages": stages,
                "reported": api.get("ApiEndpoint")}
    def ensure_admin_rest_api(self):
        """API Gateway v1 with ANY on '/' and '/{proxy+}' -> admin Lambda
        (AWS_PROXY), deployed to stage 'admin'.  v1 rather than v2 because the
        emulator edge exposes /restapis/{id}/{stage}/_user_request_/ -- the v1
        convention.  The Lambda accepts both event shapes."""
        agw, name = self.apigw_v1, f"{C.APP}-admin-rest"
        api = next((a for a in agw.get_rest_apis().get("items", [])
                    if a["name"] == name), None)
        if api is None:
            api = agw.create_rest_api(name=name,
                                      description="HoloChatStats Admin Page")
            print(f"created REST API {name} ({api['id']})")
        rest_id = api["id"]
        resources = {r.get("path"): r for r in
                     agw.get_resources(restApiId=rest_id).get("items", [])}
        root = resources["/"]["id"]
        proxy = resources.get("/{proxy+}") or agw.create_resource(
            restApiId=rest_id, parentId=root, pathPart="{proxy+}")
        uri = (f"arn:aws:apigateway:{self.args.region}:lambda:path/2015-03-31"
               f"/functions/{self.function_arns['admin']}/invocations")
        for resource_id in (root, proxy["id"]):
            try:
                agw.put_method(restApiId=rest_id, resourceId=resource_id,
                               httpMethod="ANY", authorizationType="NONE")
            except botocore.exceptions.ClientError as e:
                if not matches(e, "Conflict", "already exists"):
                    raise
            agw.put_integration(restApiId=rest_id, resourceId=resource_id,
                                httpMethod="ANY", type="AWS_PROXY",
                                integrationHttpMethod="POST", uri=uri)
        agw.create_deployment(restApiId=rest_id, stageName="admin")
        self._allow_apigw(
            f"{C.APP}-admin", "apigw-v1-invoke",
            f"arn:aws:execute-api:{self.args.region}:{self.account_id}:"
            f"{rest_id}/*/*/*")
        print(f"deployed REST API {rest_id} -> stage admin")
        return {"id": rest_id, "stages": ["admin"]}
    
    def _probe_admin_url(self, rest, http):
        base = (self.args.endpoint or "http://localhost:4566").rstrip("/")
        # Spawn the lambda container now so the HTTP probe doesn't eat the
        # cold start (floci takes 10-30 s to pull/start it).
        try:
            self.lam.invoke(FunctionName=f"{C.APP}-admin",
                            InvocationType="RequestResponse",
                            Payload=json.dumps({"rawPath": "/api/status"}).encode())
        except Exception:
            pass
        candidates = []
        # Prefer the v1 REST API's literal `admin` stage. The HTTP API's
        # `$default` stage works in a browser, but a raw `$` in proxy_pass is
        # parsed by nginx as a variable and is unsuitable for its template.
        if rest:
            for stage in rest.get("stages") or ["admin"]:
                candidates.append(f"{base}/restapis/{rest['id']}/{stage}/_user_request_/")
        if http:
            for stage in http.get("stages") or ["$default"]:
                candidates.append(f"{base}/restapis/{http['id']}/{stage}/_user_request_/")
        for url in candidates:
            if http_ok(url, timeout=30):
                return url
        with contextlib.redirect_stdout(io.StringIO()):       # silence the probe log
            url, _ = discover_admin_url(self.args.endpoint, rest=rest, http=http)
        return url
    
    def resolve_admin_url(self, rest, http):
        url = self._probe_admin_url(rest, http)
        if url:
            self.put_param(f"/{C.APP}/admin/url", url)
            self.summary.append(("Admin page", url))
        else:
            print("WARNING: admin page not reachable yet. Re-probe with: "
                  f"python scripts/admin_url.py --endpoint "
                  f"{self.args.endpoint or 'http://localhost:4566'}")
        return url
