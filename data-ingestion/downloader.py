import multiprocessing
import os
import sys
import argparse
from datetime import datetime
from multiprocessing import get_context

from db.connection import init_db_pool
from db.queries import create_database_and_tables, create_indexes_and_views, refresh_materialized_views
from config.settings import get_config
from utils.logging_utils import get_logger
from utils.helpers import setup_cache_dir
from cacheutil.cache_manager import process_cache_dir
from workers.chat_downloader import download_chat_log
from workers.db_worker import db_worker
from workers.metadata_fetcher import get_metadata_for_date_range

def parse_args():
    parser = argparse.ArgumentParser(description="Chat log and metadata downloader")
    parser.add_argument(
        "--year", type=int, default=datetime.now().year,
        help="Year to process (default: current year)"
    )
    parser.add_argument(
        "--month", type=int, default=datetime.now().month,
        help="Month to process (default: current month)"
    )
    return parser.parse_args()
    
if get_config("Settings", "TestMode") == "True":
    os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:8080'
    os.environ['HTTP_PROXY'] = 'http://127.0.0.1:8080'
    caller_script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.environ['REQUESTS_CA_BUNDLE'] = os.path.join(caller_script_dir, 'mitmproxy-ca-cert.pem')
    os.environ['SSL_CERT_FILE'] = os.path.join(caller_script_dir, 'mitmproxy-ca-cert.pem')

#download_queue = []
#DB_POOL = None

def main():
    args = parse_args()
    YEAR = args.year
    MONTH = args.month

    # Setup
    logger = get_logger()

    logger.info("Starting chat log and metadata downloader...")
    init_db_pool()
    create_database_and_tables()
    create_indexes_and_views()
    setup_cache_dir()

    # Initialize database queue
    with multiprocessing.Manager() as manager:
        download_queue = manager.list()
        queue = manager.Queue()

        process_cache_dir(download_queue)

        get_metadata_for_date_range(YEAR, MONTH, download_queue)

        db_worker_process = multiprocessing.Process(target=db_worker, args=(queue,))
        db_worker_process.start()

        # Deduplicate download queue
        download_queue[:] = list(dict.fromkeys(list(download_queue)))

        logger.info(f"Processing {len(download_queue)} chat logs...")

        # Download chat logs using multiprocessing
        with get_context("spawn").Pool(processes=int(get_config("Settings", "NumThreads")), initializer=init_db_pool) as pool:
            pool.starmap(download_chat_log, [(channel_id, video_id, queue, YEAR, MONTH) for channel_id, video_id in download_queue])

        # Stop database worker process
        queue.put(None)
        db_worker_process.join()

    refresh_materialized_views()

    logger.info("All chat logs downloaded and processed.")

if __name__ == "__main__":
    main()