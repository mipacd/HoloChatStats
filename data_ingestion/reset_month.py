#!/usr/bin/env python3
"""
Reset ETL pipeline data for a specific month (YYYY-MM format).
Removes data from both JSON cache and PostgreSQL database.
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
import psycopg2
from psycopg2.extras import execute_values
import sys

def parse_month(month_str):
    """Parse YYYY-MM string and return year and month."""
    try:
        date = datetime.strptime(month_str, "%Y-%m")
        return date.year, date.month
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid month format: {month_str}. Use YYYY-MM")

def is_in_month(timestamp_str, year, month):
    """Check if timestamp string is in the given year-month."""
    try:
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return dt.year == year and dt.month == month
    except (ValueError, AttributeError):
        return False

def clean_video_cache(cache_dir, year, month, dry_run=False):
    """
    Remove videos from cache that match the given month.
    Returns list of video IDs that were removed.
    """
    videos_dir = Path(cache_dir) / "videos"
    if not videos_dir.exists():
        print(f"Warning: Videos cache directory not found: {videos_dir}")
        return []
    
    removed_video_ids = []
    
    for json_file in videos_dir.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            original_count = len(data)
            videos_to_remove = []
            
            # Find videos to remove
            for video_id, video_data in data.items():
                if 'end_time' in video_data:
                    if is_in_month(video_data['end_time'], year, month):
                        videos_to_remove.append(video_id)
                        removed_video_ids.append(video_id)
            
            # Remove videos
            if videos_to_remove:
                if not dry_run:
                    for video_id in videos_to_remove:
                        del data[video_id]
                    
                    # Write back to file
                    with open(json_file, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=4)
                
                print(f"  {json_file.name}: Removed {len(videos_to_remove)}/{original_count} videos")
            
        except json.JSONDecodeError as e:
            print(f"Error reading {json_file}: {e}", file=sys.stderr)
        except Exception as e:
            print(f"Error processing {json_file}: {e}", file=sys.stderr)
    
    return removed_video_ids

def clean_chat_logs(cache_dir, video_ids, dry_run=False):
    """Remove chat logs for the given video IDs."""
    chat_logs_dir = Path(cache_dir) / "chat_logs"
    if not chat_logs_dir.exists():
        print(f"Warning: Chat logs directory not found: {chat_logs_dir}")
        return
    
    removed_count = 0
    for video_id in video_ids:
        chat_log_path = chat_logs_dir / f"{video_id}.jsonl.gz"
        if chat_log_path.exists():
            if not dry_run:
                chat_log_path.unlink()
            removed_count += 1
            if removed_count <= 10:  # Don't spam console
                print(f"  Removed: {chat_log_path.name}")
    
    if removed_count > 10:
        print(f"  ... and {removed_count - 10} more chat logs")
    elif removed_count > 0:
        print(f"  Total: {removed_count} chat logs removed")

def clean_database(db_config, year, month, dry_run=False):
    """Remove data from PostgreSQL database for the given month."""
    conn = None
    try:
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        
        # Create date range for the month
        start_date = f"{year}-{month:02d}-01"
        if month == 12:
            end_date = f"{year + 1}-01-01"
        else:
            end_date = f"{year}-{month + 1:02d}-01"
        
        print("\nDatabase operations:")
        
        # Delete from videos table using end_time
        cursor.execute("""
            SELECT COUNT(*) FROM videos 
            WHERE end_time >= %s AND end_time < %s
        """, (start_date, end_date))
        videos_count = cursor.fetchone()[0]
        
        if not dry_run:
            cursor.execute("""
                DELETE FROM videos 
                WHERE end_time >= %s AND end_time < %s
            """, (start_date, end_date))
        print(f"  videos: {videos_count} rows {'would be' if dry_run else ''} deleted")
        
        # Delete from user_data table using last_message_at
        cursor.execute("""
            SELECT COUNT(*) FROM user_data 
            WHERE last_message_at >= %s AND last_message_at < %s
        """, (start_date, end_date))
        user_data_count = cursor.fetchone()[0]
        
        if not dry_run:
            cursor.execute("""
                DELETE FROM user_data 
                WHERE last_message_at >= %s AND last_message_at < %s
            """, (start_date, end_date))
        print(f"  user_data: {user_data_count} rows {'would be' if dry_run else ''} deleted")
        
        # Truncate forecast tables
        cursor.execute("SELECT COUNT(*) FROM streaming_forecasts")
        forecasts_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM forecast_model_metrics")
        metrics_count = cursor.fetchone()[0]
        
        if not dry_run:
            cursor.execute("TRUNCATE TABLE streaming_forecasts CASCADE")
            cursor.execute("TRUNCATE TABLE forecast_model_metrics CASCADE")
        
        print(f"  streaming_forecasts: {forecasts_count} rows {'would be' if dry_run else ''} deleted (truncated)")
        print(f"  forecast_model_metrics: {metrics_count} rows {'would be' if dry_run else ''} deleted (truncated)")
        
        # Refresh materialized views
        if not dry_run:
            print("\nRefreshing materialized views:")
            views = [
                "mv_user_monthly_activity",
                "mv_membership_data",
                "mv_user_activity",
                "chat_language_stats_mv",
                "mv_user_language_per_month"
            ]
            
            for view in views:
                try:
                    print(f"  Refreshing {view}...")
                    cursor.execute(f"REFRESH MATERIALIZED VIEW {view}")
                except psycopg2.Error as e:
                    print(f"  Warning: Could not refresh {view}: {e}")
        else:
            print("\n  Materialized views would be refreshed")
        
        if not dry_run:
            conn.commit()
            print("\nDatabase changes committed.")
        else:
            print("\nDry run - no database changes made.")
        
    except psycopg2.Error as e:
        print(f"Database error: {e}", file=sys.stderr)
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()

def main():
    parser = argparse.ArgumentParser(
        description="Reset ETL pipeline data for a specific month",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s 2025-02 --cache-dir ./cache --db-host localhost --db-name mydb
  %(prog)s 2024-12 --dry-run  # Preview changes without making them
        """
    )
    
    parser.add_argument(
        "month",
        help="Month to reset in YYYY-MM format (e.g., 2025-02)"
    )
    
    parser.add_argument(
        "--cache-dir",
        default="cache",
        help="Path to cache directory (default: cache)"
    )
    
    parser.add_argument(
        "--db-host",
        default=os.getenv("DB_HOST", "localhost"),
        help="Database host (default: localhost or DB_HOST env var)"
    )
    
    parser.add_argument(
        "--db-port",
        type=int,
        default=int(os.getenv("DB_PORT", "5432")),
        help="Database port (default: 5432 or DB_PORT env var)"
    )
    
    parser.add_argument(
        "--db-name",
        default=os.getenv("DB_NAME", "postgres"),
        help="Database name (default: postgres or DB_NAME env var)"
    )
    
    parser.add_argument(
        "--db-user",
        default=os.getenv("DB_USER", "postgres"),
        help="Database user (default: postgres or DB_USER env var)"
    )
    
    parser.add_argument(
        "--db-password",
        default=os.getenv("DB_PASSWORD", ""),
        help="Database password (default: empty or DB_PASSWORD env var)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without making them"
    )
    
    parser.add_argument(
        "--skip-cache",
        action="store_true",
        help="Skip cache cleaning, only clean database"
    )
    
    parser.add_argument(
        "--skip-db",
        action="store_true",
        help="Skip database cleaning, only clean cache"
    )
    
    args = parser.parse_args()
    
    # Parse month
    year, month = parse_month(args.month)
    
    print(f"{'DRY RUN: ' if args.dry_run else ''}Resetting data for {year}-{month:02d}")
    print("=" * 60)
    
    # Clean cache
    removed_video_ids = []
    if not args.skip_cache:
        print("\nCleaning video cache:")
        removed_video_ids = clean_video_cache(args.cache_dir, year, month, args.dry_run)
        print(f"Total videos removed: {len(removed_video_ids)}")
        
        if removed_video_ids:
            print("\nCleaning chat logs:")
            clean_chat_logs(args.cache_dir, removed_video_ids, args.dry_run)
    
    # Clean database
    if not args.skip_db:
        db_config = {
            "host": args.db_host,
            "port": args.db_port,
            "database": args.db_name,
            "user": args.db_user,
            "password": args.db_password
        }
        
        clean_database(db_config, year, month, args.dry_run)
    
    print("\n" + "=" * 60)
    print(f"Reset {'preview' if args.dry_run else 'complete'} for {year}-{month:02d}")
    
    if args.dry_run:
        print("\nThis was a dry run. Use without --dry-run to apply changes.")

if __name__ == "__main__":
    main()