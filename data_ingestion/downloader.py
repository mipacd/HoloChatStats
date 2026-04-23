import multiprocessing
import os
import sys
import argparse
from datetime import datetime, timedelta
from multiprocessing import get_context

from db.connection import init_db_pool, get_db_connection
from db.queries import (
    create_database_and_tables,
    create_indexes_and_views,
    refresh_materialized_views,
    purge_month_db_data,
)
from config.settings import get_config
from utils.logging_utils import get_logger
from utils.helpers import setup_cache_dir
from cacheutil.cache_manager import (
    process_cache_dir,
    gzip_uncompressed_chat_logs,
    get_cached_videos_for_month,
    purge_month_chat_log_cache,
)
from workers.chat_downloader import download_chat_log
from workers.db_worker import db_worker
from workers.metadata_fetcher import get_metadata_for_date_range

from utils.ai_summarizer import populate_video_highlights
from utils.forecaster import StreamingHoursForecaster


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
    parser.add_argument(
        "--disable_ai_summarization", "-d", action="store_true",
        help="Disable AI summarization for video highlights"
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Purge all DB data and cached chat logs for the given --year/--month, "
             "then re-download and rebuild that month from scratch. Other months are untouched."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only valid with --rebuild. Print every action that would be taken "
             "without modifying the database or cache and without any network calls."
    )
    return parser.parse_args()


# Proxy setup for testing
if get_config("Settings", "TestMode") == "True":
    os.environ['HTTPS_PROXY'] = 'http://122.0.0.1:8080'
    os.environ['HTTP_PROXY'] = 'http://122.0.0.1:8080'
    caller_script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.environ['REQUESTS_CA_BUNDLE'] = os.path.join(caller_script_dir, 'mitmproxy-ca-cert.pem')
    os.environ['SSL_CERT_FILE'] = os.path.join(caller_script_dir, 'mitmproxy-ca-cert.pem')


def _plan_and_purge_month(year, month, dry_run, disable_ai_summarization):
    """
    Print the rebuild plan for the given month and, unless dry_run is True,
    execute the destructive parts (DB purge + chat-log cache purge).

    The actual re-download / re-ingest is performed by the normal pipeline in
    main() after this returns (only when dry_run is False). In dry-run mode this
    function is strictly read-only: it issues SELECT COUNT(*) queries and reads
    the filesystem, nothing more.
    """
    logger = get_logger()
    tag = "[DRY RUN] " if dry_run else ""

    logger.info(f"{tag}=== Rebuild plan for {year}-{month:02d} ===")

    month_videos = get_cached_videos_for_month(year, month)
    logger.info(f"{tag}{len(month_videos)} video(s) found in metadata cache for {year}-{month:02d}.")

    logger.info(f"{tag}-- Step 1/4: purge database rows for {year}-{month:02d} --")
    purge_month_db_data(year, month, dry_run=dry_run)

    logger.info(f"{tag}-- Step 2/4: purge cached chat logs for {year}-{month:02d} --")
    purge_month_chat_log_cache(month_videos, dry_run=dry_run)

    logger.info(f"{tag}-- Step 3/4: re-download & re-ingest chat logs --")
    if month_videos:
        for channel_id, video_id in month_videos:
            logger.info(f"{tag}  re-download chat log: channel={channel_id} video={video_id}")
    else:
        logger.info(f"{tag}  (no videos found in metadata cache for this month)")

    logger.info(f"{tag}-- Step 4/4: post-processing --")
    logger.info(f"{tag}  refresh materialized views + membership summary for {year}-{month:02d}")
    logger.info(f"{tag}  run streaming-hours forecasting")
    if not disable_ai_summarization:
        logger.info(f"{tag}  run AI highlight generation for {year}-{month:02d}")

    logger.info(f"{tag}=== End of rebuild plan ===")


def main():
    args = parse_args()
    YEAR = args.year
    MONTH = args.month
    DISABLE_AI_SUMMARIZATION = args.disable_ai_summarization
    REBUILD = args.rebuild
    DRY_RUN = args.dry_run

    logger = get_logger()

    if DRY_RUN and not REBUILD:
        logger.error("--dry-run can only be used together with --rebuild.")
        sys.exit(1)

    init_db_pool()

    if REBUILD:
        _plan_and_purge_month(YEAR, MONTH, dry_run=DRY_RUN,
                              disable_ai_summarization=DISABLE_AI_SUMMARIZATION)
        if DRY_RUN:
            logger.info("[DRY RUN] No changes were made and no network requests were issued. Exiting.")
            return
        logger.info(f"Purge complete. Proceeding with full re-download for {YEAR}-{MONTH:02d}.")

    logger.info("Starting data pipeline...")
    create_database_and_tables()
    create_indexes_and_views()
    gzip_uncompressed_chat_logs()
    setup_cache_dir()

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
            pool.starmap(
                download_chat_log,
                [(channel_id, video_id, queue, YEAR, MONTH) for channel_id, video_id in download_queue],
            )

        # Stop database worker process
        queue.put(None)
        db_worker_process.join()

    refresh_materialized_views(YEAR, MONTH)

    # Streaming Hours Forecasting
    logger.info("Starting streaming hours forecasting...")
    forecaster = StreamingHoursForecaster(get_db_connection())
    result = forecaster.run_forecasting_pipeline()
    logger.info(f"Streaming hours forecasting results: {result}")

    logger.info("Chat log download and DB processing complete.")

    if not DISABLE_AI_SUMMARIZATION:
        logger.info("Starting AI highlight generation for the month...")
        try:
            populate_video_highlights(YEAR, MONTH)
            logger.info("AI highlight generation complete.")
        except Exception as e:
            logger.error(f"An error occurred during AI highlight generation: {e}", exc_info=True)

    logger.info("All tasks are complete.")


if __name__ == "__main__":
    main()