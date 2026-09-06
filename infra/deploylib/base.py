"""Shared plumbing: clients, SSM, IAM, security groups, VPC, port discovery."""
import json
import time
import botocore.exceptions
from . import config as C
from .awsutil import (Clients, err_code, http_ok, ignore, matches, retry,
                      scan_local_ports)

class Base:
    BOOTSTRAP_PARAM = f"/{C.APP}/bootstrap/complete"
    SUMMARY_ORDER = ("Admin page", "API", "Frontend", "LLM")
    def __init__(self, args):
        self.args = args
        self.aws = Clients(args.region, args.endpoint)
        self.account_id = retry(self.aws.sts.get_caller_identity)["Account"]
        self.queue_urls, self.queue_arns, self.function_arns = {}, {}, {}
        self.elasticache = None
        self._code_cache = None
        self.summary = []
    def __getattr__(self, name):
        """stack.s3 / stack.lam / stack.ecs ... resolve to lazily built clients."""
        if name.startswith("_") or name in ("aws", "args"):
            raise AttributeError(name)
        return getattr(self.aws, name)
    @property
    def frontend_port(self):
        return getattr(self.args, "frontend_port", None) or C.FRONTEND_PORT
    @property
    def frontend_host_port(self):
        return (getattr(self.args, "frontend_host_port", None)
                or self.frontend_port)
    # ------------------------------------------------------------ context ---
    @property
    def emulated(self):
        """True when pointed at floci/LocalStack rather than real AWS."""
        return bool(self.args.endpoint)
    @property
    def web_port(self):
        return getattr(self.args, "web_port", None) or C.WEB_PORT
    def internal_endpoint(self):
        """Edge URL as seen from inside emulator-spawned containers."""
        return self.args.lambda_endpoint or self.args.endpoint
    def _api_call(self, method, *a, **kw):
        return retry(method, *a, **kw)
    def banner(self, title, rows):
        print("\n" + "=" * 64)
        print(f"  {title}")
        for key, value in rows:
            print(f"  {key:<22}: {value}")
        print("=" * 64)
    # ---------------------------------------------------------------- SSM ---
    def put_param(self, name, value):
        self._api_call(self.ssm.put_parameter, Name=name, Value=str(value),
                       Type="String", Overwrite=True)
    def put_params(self, mapping):
        for name, value in mapping.items():
            self.put_param(name, value)
    def get_param(self, name, default=None):
        try:
            return self._api_call(self.ssm.get_parameter,
                                  Name=name)["Parameter"]["Value"]
        except botocore.exceptions.ClientError:
            return default
    def bootstrap_marker(self):
        return self.get_param(self.BOOTSTRAP_PARAM)
    def mark_bootstrapped(self):
        self.put_param(self.BOOTSTRAP_PARAM,
                       time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    # ---------------------------------------------------------------- IAM ---
    def ensure_role(self, name, service, managed=(), inline=None):
        trust = {"Version": "2012-10-17", "Statement": [{
            "Effect": "Allow", "Principal": {"Service": service},
            "Action": "sts:AssumeRole"}]}
        try:
            self.iam.create_role(RoleName=name,
                                 AssumeRolePolicyDocument=json.dumps(trust))
            print(f"created role {name}")
        except self.iam.exceptions.EntityAlreadyExistsException:
            pass
        if inline:
            self.iam.put_role_policy(RoleName=name, PolicyName="inline",
                                     PolicyDocument=json.dumps(inline))
        for arn in managed:
            # Emulators often lack AWS-managed policies; the inline grant covers us.
            ignore(self.iam.attach_role_policy, RoleName=name, PolicyArn=arn)
        return self.iam.get_role(RoleName=name)["Role"]["Arn"]
    # ----------------------------------------------------------- EC2 / VPC ---
    def default_network(self):
        """(vpc_id, [subnet_ids]) for the default VPC.  Both may be empty on an
        emulator that does not model VPCs -- callers degrade gracefully."""
        vpc_id, subnets = None, []
        try:
            vpcs = self._api_call(self.ec2.describe_vpcs).get("Vpcs", [])
            vpc_id = next((v["VpcId"] for v in vpcs if v.get("IsDefault")),
                          vpcs[0]["VpcId"] if vpcs else None)
            flt = [{"Name": "vpc-id", "Values": [vpc_id]}] if vpc_id else []
            subnets = [s["SubnetId"] for s in self._api_call(
                self.ec2.describe_subnets, Filters=flt).get("Subnets", [])]
        except botocore.exceptions.ClientError as e:
            print(f"  (VPC lookup unavailable: {err_code(e) or e})")
        return vpc_id, subnets
    def ensure_security_group(self, name, ports, description=None, vpc_id=None):
        sg_id = None
        try:
            kwargs = {"GroupName": name, "Description": description or name}
            if vpc_id:
                kwargs["VpcId"] = vpc_id
            sg_id = self._api_call(self.ec2.create_security_group,
                                **kwargs)["GroupId"]
            print(f"created security group {name} ({sg_id})")
        except botocore.exceptions.ClientError as e:
            if not matches(e, "Duplicate", "already exists"):
                raise
            for lookup in ({"GroupNames": [name]},
                        {"Filters": [{"Name": "group-name",
                                        "Values": [name]}]}):
                groups = (ignore(self.ec2.describe_security_groups, **lookup)
                        or {}).get("SecurityGroups", [])
                if groups:
                    sg_id = groups[0]["GroupId"]
                    break
        # (re)authorise every requested port on both new and pre-existing groups
        for p in ports:
            ignore(self.ec2.authorize_security_group_ingress, GroupId=sg_id,
                IpPermissions=[{"IpProtocol": "tcp", "FromPort": p, "ToPort": p,
                                "IpRanges": [{"CidrIp": "0.0.0.0/0"}]}],
                only=("Duplicate", "already exists", "InvalidPermission"))
        return sg_id
    def discover_forwarded_port(self, before, probe=None, timeout=60,
                                exclude=()):
        """floci maps a guest port to a host port via a socat sidecar.  Prefer a
        newly-appeared port, but fall back to any port in the range that answers
        `probe` -- floci reuses host ports, so the diff alone is unreliable."""
        deadline, candidates = time.time() + timeout, []
        while time.time() < deadline:
            listening = set(scan_local_ports(*C.PORT_SCAN_RANGE)) - set(exclude)
            candidates = sorted(listening - set(before))
            if probe is None:
                if candidates:
                    return candidates[0]
            else:
                for port in candidates + sorted(listening - set(candidates)):
                    if http_ok(f"http://localhost:{port}{probe}"):
                        return port
            time.sleep(3)
        if candidates:
            return candidates[0]
        print(f"  (no port in {C.PORT_SCAN_RANGE} answered "
              f"{probe or 'a TCP connect'}; listening: "
              f"{sorted(scan_local_ports(*C.PORT_SCAN_RANGE))})")
        return None
    def print_summary(self):
        items = dict(self.summary)
        rows = [(k, items[k]) for k in self.SUMMARY_ORDER if k in items]
        if not rows:
            return
        print("\n" + "=" * 64)
        for label, url in rows:
            print(f"  {label:<12}: {url}")
        print("=" * 64)