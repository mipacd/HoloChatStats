-- Production downloads are deliberately serialized. This UPDATE changes
-- databases that already received the older seed value; the advisory lock in
-- handlers/download.py remains the authoritative cross-invocation guard.
INSERT INTO service_config (key, value, updated_at)
VALUES ('max_concurrent_downloads', '1', NOW())
ON CONFLICT (key) DO UPDATE
  SET value = '1', updated_at = NOW();
