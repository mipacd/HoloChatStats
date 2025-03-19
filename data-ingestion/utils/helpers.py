import os
from utils.logging_utils import get_logger
from config.settings import get_config

# Create necessary directories for cache
def setup_cache_dir():
    logger = get_logger()
    """Create cache directories if they do not exist.

    The cache directories are created inside the configured cache directory.
    The cache directory is divided into two subdirectories: "videos" and "chat_logs".
    """
    cache_dir = get_config("Settings", "CacheDir")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    if not os.path.exists(cache_dir + "/videos"):
        os.makedirs(os.path.join(cache_dir, "videos"))
    if not os.path.exists(cache_dir + "/chat_logs"):
        os.makedirs(os.path.join(cache_dir, "chat_logs"))
    logger.info("Cache directories configured.")

def deduplicate_batch(batch, key_indices):
    """Deduplicates a batch of rows based on the unique key indices."""
    seen = set()
    deduplicated_batch = []

    for row in batch:
        # Extract the unique key (e.g., a tuple of the key indices)
        key = tuple(row[index] for index in key_indices)
        if key not in seen:
            deduplicated_batch.append(row)
            seen.add(key)

    return deduplicated_batch
