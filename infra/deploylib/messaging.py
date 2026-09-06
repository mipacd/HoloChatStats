"""SQS queues, Lambda event source mappings, EventBridge schedules."""
import json
import botocore.exceptions
from . import config as C
class MessagingMixin:
    def ensure_queues(self):
        # DLQs first so redrive policies can reference their ARNs.
        for name in sorted(C.QUEUES, key=lambda q: 0 if q.endswith("dlq") else 1):
            spec = C.QUEUES[name]
            attrs = {"VisibilityTimeout": str(spec["visibility"]),
                     "MessageRetentionPeriod": str(14 * 86400)}
            if "dlq" in spec:
                attrs["RedrivePolicy"] = json.dumps({
                    "deadLetterTargetArn": self.queue_arns[spec["dlq"]],
                    "maxReceiveCount": spec["max_receive"]})
            queue_name = f"{C.APP}-{name}"
            try:
                url = self.sqs.create_queue(QueueName=queue_name,
                                            Attributes=attrs)["QueueUrl"]
            except self.sqs.exceptions.QueueNameExists:
                url = self.sqs.get_queue_url(QueueName=queue_name)["QueueUrl"]
                self.sqs.set_queue_attributes(QueueUrl=url, Attributes=attrs)
            self.queue_urls[name] = url
            self.queue_arns[name] = self.sqs.get_queue_attributes(
                QueueUrl=url, AttributeNames=["QueueArn"]
            )["Attributes"]["QueueArn"]
            print(f"queue {name}: {url}")
    def ensure_esms(self):
        for queue, (func, batch, max_conc) in C.EVENT_SOURCE_MAPPINGS.items():
            fn, arn = f"{C.APP}-{func}", self.queue_arns[queue]
            scaling = {"ScalingConfig": {"MaximumConcurrency": max(2, max_conc)}}
            existing = self.lam.list_event_source_mappings(
                FunctionName=fn, EventSourceArn=arn)["EventSourceMappings"]
            if existing:
                call, ident = self.lam.update_event_source_mapping, \
                    {"UUID": existing[0]["UUID"]}
            else:
                call, ident = self.lam.create_event_source_mapping, \
                    {"FunctionName": fn, "EventSourceArn": arn, "Enabled": True}
            try:
                uuid = call(BatchSize=batch, **ident, **scaling)["UUID"]
            except botocore.exceptions.ClientError:
                # Emulator may not model ScalingConfig; reserved concurrency on
                # the function is the fallback cap.
                uuid = call(BatchSize=batch, **ident)["UUID"]
                print(f"  ESM ScalingConfig unsupported for {queue}; "
                      f"using reserved concurrency as the cap")
            if not existing:
                print(f"mapped {queue} -> {fn}")
            # Persist the UUID so the runtime concurrency knob
            # (infra/apply_concurrency.py / discover) can find it.
            self.put_param(f"/{C.APP}/esm/{queue}", uuid)
    def ensure_schedules(self):
        for func, expr in C.SCHEDULES.items():
            fn, rule = f"{C.APP}-{func}", f"{C.APP}-{func}-schedule"
            rule_arn = self.events.put_rule(Name=rule, ScheduleExpression=expr,
                                            State="ENABLED")["RuleArn"]
            self.events.put_targets(Rule=rule, Targets=[
                {"Id": "1", "Arn": self.function_arns[func]}])
            try:
                self.lam.add_permission(
                    FunctionName=fn, StatementId=f"{rule}-invoke",
                    Action="lambda:InvokeFunction",
                    Principal="events.amazonaws.com", SourceArn=rule_arn)
            except self.lam.exceptions.ResourceConflictException:
                pass
            print(f"scheduled {fn}: {expr}")