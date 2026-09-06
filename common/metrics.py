import json, time, os
NAMESPACE = os.environ.get("METRIC_NAMESPACE", "ChatIngestion")
def emit(metrics: dict, dimensions: dict | None = None, **properties):
    dims = dimensions or {}
    doc = {
        "_aws": {
            "Timestamp": int(time.time() * 1000),
            "CloudWatchMetrics": [{
                "Namespace": NAMESPACE,
                "Dimensions": [list(dims)] if dims else [[]],
                "Metrics": [{"Name": k, "Unit": u} for k, (_, u) in metrics.items()],
            }],
        },
        **dims,
        **{k: v for k, (v, _) in metrics.items()},
        **properties,
    }
    print(json.dumps(doc))
COUNT, SECONDS = "Count", "Seconds"