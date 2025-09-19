import multiprocessing
import os
import sys
import argparse
from datetime import datetime, timedelta
from multiprocessing import get_context

# --- Local Imports ---
from db.connection import init_db_pool
from db.queries import create_database_and_tables, create_indexes_and_views, refresh_materialized_views
from config.settings import get_config
from utils.logging_utils import get_logger
from utils.helpers import setup_cache_dir
from cacheutil.cache_manager import process_cache_dir, gzip_uncompressed_chat_logs
from workers.chat_downloader import download_chat_log
from workers.db_worker import db_worker
from workers.metadata_fetcher import get_metadata_for_date_range

# --- NEW: Import the AI summarizer function ---
from utils.ai_summarizer import populate_video_highlights

def parse_args():
    """Parses command-line arguments for the year and month to process."""
    parser = argparse.ArgumentParser(description="Chat log and metadata downloader")
    today = datetime.now()
    if today.day == 1:
        # If it's the 1st of the month, default to the previous month
        last_month = today - timedelta(days=1)
        default_year = last_month.year
        default_month = last_month.month
    else:
        default_year = today.year
        default_month = today.month

    parser.add_argument(
        "--year", type=int, default=default_year,
        help="Year to process (default: current year unless 1st of month)"
    )
    parser.add_argument(
        "--month", type=int, default=default_month,
        help="Month to process (default: current month unless 1st of month)"
    )
    return parser.parse_args()
    
# --- Proxy setup for testing (if needed) ---
if get_config("Settings", "TestMode") == "True":
    os.environ['HTTPS_PROXY'] = 'http://122.0.0.1:8080'
    os.environ['HTTP_PROXY'] = 'http://122.0.0.1:8080'
    caller_script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.environ['REQUESTS_CA_BUNDLE'] = os.path.join(caller_script_dir, 'mitmproxy-ca-cert.pem')
    os.environ['SSL_CERT_FILE'] = os.path.join(caller_script_dir, 'mitmproxy-ca-cert.pem')

def main():
    args = parse_args()
    YEAR = args.year
    MONTH = args.month

    # --- Setup ---
    logger = get_logger()
    logger.info("Starting data pipeline...")
    init_db_pool()
    create_database_and_tables()
    create_indexes_and_views()
    gzip_uncompressed_chat_logs()
    setup_cache_dir()

    # --- Step 1: Download Metadata and Chat Logs ---
    with multiprocessing.Manager() as manager:
        download_queue = manager.list()
        queue = manager.Queue()

        process_cache_dir(download_queue, YEAR, MONTH)
        get_metadata_for_date_range(YEAR, MONTH, download_queue)

        db_worker_process = multiprocessing.Process(target=db_worker, args=(queue,))
        db_worker_process.start()

        # Deduplicate download queue
        download_queue[:] = list(dict.fromkeys(list(download_queue)))

        logger.info(f"Processing {len(download_queue)} chat logs for {YEAR}-{MONTH:02d}...")
        
        # Download chat logs using multiprocessing
        num_threads = int(get_config("Settings", "NumThreads"))
        with get_context("spawn").Pool(processes=num_threads, initializer=init_db_pool) as pool:
            pool.starmap(download_chat_log, [(channel_id, video_id, queue, YEAR, MONTH) for channel_id, video_id in download_queue])

        # Stop database worker process
        queue.put(None)
        db_worker_process.join()

    # --- Step 2: Finalize Database ---
    refresh_materialized_views()
    logger.info("Chat log download and DB processing complete.")

    # --- NEW: Step 3: Generate AI Highlights ---
    logger.info("Starting AI highlight generation for the month...")
    try:
        populate_video_highlights(YEAR, MONTH)
        logger.info("AI highlight generation complete.")
    except Exception as e:
        logger.error(f"An error occurred during AI highlight generation: {e}", exc_info=True)

    logger.info("All tasks are complete.")

if __name__ == "__main__":
    main()