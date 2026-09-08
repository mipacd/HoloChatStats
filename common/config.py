import json, os, time
from common.aws import client
_TTL = 60
_cache, _cache_at = {}, 0.0
def _load_from_db():
    from common.db import get_conn
    with get_conn().cursor() as cur:
        cur.execute("SELECT key, value FROM service_config")
        return dict(cur.fetchall())
def settings(force: bool = False) -> dict:
    """Env var wins (useful for per-invocation overrides in tests), then DB."""
    global _cache, _cache_at
    if force or not _cache or time.time() - _cache_at > _TTL:
        _cache = _load_from_db()
        _cache_at = time.time()
    merged = dict(_cache)
    for k in list(merged):
        env = os.environ.get(k.upper())
        if env is not None:
            merged[k] = env
    return merged
def setting(key, default=None, cast=str):
    v = settings().get(key, default)
    return default if v is None else cast(v)
def secret(name: str) -> dict:
    sm = client("secretsmanager")
    return json.loads(sm.get_secret_value(SecretId=name)["SecretString"])