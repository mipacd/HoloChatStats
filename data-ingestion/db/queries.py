from db.connection import get_db_connection, release_db_connection
from utils.logging_utils import get_logger
import psycopg2
from psycopg2.extras import execute_values

# Create database if it doesn't exist
def create_database_and_tables():
    logger = get_logger()
    """
    Ensures that the database and tables required for the chat downloader are set up.

    Connects to the database, creates the required tables if they do not exist, and then commits and closes the connection.

    This function should only be called once per program execution.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS channels (
            channel_id TEXT PRIMARY KEY,
            channel_name TEXT NOT NULL,
            channel_group TEXT
        );
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
        user_id TEXT PRIMARY KEY,
        username TEXT NOT NULL
        );
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS videos (
            video_id TEXT PRIMARY KEY,
            channel_id TEXT,
            title TEXT NOT NULL,
            end_time TIMESTAMP WITH TIME ZONE NOT NULL,
            duration INTERVAL,
            processed_at TIMESTAMP DEFAULT NOW(),
            has_chat_log BOOLEAN DEFAULT FALSE,
            funniest_timestamp INT
        );
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_data (
            user_id TEXT,
            channel_id TEXT,
            first_message_at TIMESTAMP WITH TIME ZONE NOT NULL,
            last_message_at TIMESTAMP WITH TIME ZONE NOT NULL,
            video_id TEXT,
            membership_rank INT NOT NULL,
            jp_count INT DEFAULT 0,
            kr_count INT DEFAULT 0,
            ru_count INT DEFAULT 0,
            emoji_count INT DEFAULT 0,
            es_en_id_count INT DEFAULT 0,
            total_message_count INT DEFAULT 0,
            PRIMARY KEY (user_id, channel_id, last_message_at, video_id)
        );
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_language_stats (
            channel_id TEXT,
            observed_month DATE NOT NULL,
            jp_count INT DEFAULT 0,
            kr_count INT DEFAULT 0,
            ru_count INT DEFAULT 0,
            emoji_count INT DEFAULT 0,
            es_en_id_count INT DEFAULT 0,
            total_messages INT DEFAULT 0,
            PRIMARY KEY (channel_id, observed_month)
    );
    """)
    conn.commit()
    release_db_connection(conn)
    logger.info("Database and tables are configured.")

def create_indexes_and_views():
    logger = get_logger()
    conn = get_db_connection()

    cursor = conn.cursor()

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_data_user_id_last_message_at ON user_data (user_id, last_message_at);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_data_user_id ON user_data (user_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_data_channel_id ON user_data (channel_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_channels_channel_id ON channels (channel_id);")

    # Create membership data view
    cursor.execute("""CREATE MATERIALIZED VIEW IF NOT EXISTS mv_membership_data AS
                        WITH latest_memberships AS (
                            SELECT 
                                m.user_id,
                                m.channel_id,
                                DATE_TRUNC('month', m.last_message_at)::DATE AS observed_month,
                                m.membership_rank,
                                m.last_message_at,
                                ROW_NUMBER() OVER (
                                    PARTITION BY m.user_id, m.channel_id, DATE_TRUNC('month', m.last_message_at) 
                                    ORDER BY m.last_message_at DESC
                                ) AS row_num
                            FROM user_data m
                        )
                        SELECT
                            c.channel_group,
                            c.channel_name,
                            lm.observed_month,
                            lm.membership_rank,
                            COUNT(lm.user_id) AS membership_count,
                            ROUND(
                                COUNT(lm.user_id)::DECIMAL / NULLIF(SUM(COUNT(*)) OVER (PARTITION BY c.channel_name, lm.observed_month), 0) * 100, 2
                            ) AS percentage_total
                            FROM latest_memberships lm
                            JOIN channels c ON lm.channel_id = c.channel_id
                            WHERE lm.row_num = 1
                            GROUP BY c.channel_group, c.channel_name, lm.observed_month, lm.membership_rank
                            ORDER BY c.channel_group, c.channel_name, lm.observed_month, lm.membership_rank;
    """)

    cursor.execute("""CREATE MATERIALIZED VIEW IF NOT EXISTS mv_user_monthly_activity AS
            SELECT
                user_id,
                channel_id,
                DATE_TRUNC('month', last_message_at) AS observed_month,
                SUM(total_message_count) AS monthly_message_count
            FROM user_data
            GROUP BY user_id, channel_id, observed_month;
            """
        )
    
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_mv_user_monthly_activity ON mv_user_monthly_activity (observed_month, channel_id);")

    conn.commit()
    release_db_connection(conn)
    logger.info("Indexes and views are configured.")

# Refresh materialized views
def refresh_materialized_views():
    logger = get_logger()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("REFRESH MATERIALIZED VIEW mv_common_chatters;")
    cursor.execute("REFRESH MATERIALIZED VIEW mv_membership_data;")
    conn.commit()
    release_db_connection(conn)
    logger.info("Materialized views refreshed.")

# Insert video metadata into database
def insert_video_metadata(channel_id, video_id, title, end_time, duration, has_chat_log=False):
    logger = get_logger()
    """
    Inserts or updates video metadata in the database.

    This function inserts a new record into the 'videos' table with the provided metadata.
    If a record with the same video_id already exists, it updates the existing record
    with the new metadata. The duration is converted to an INTERVAL type in PostgreSQL.

    Args:
        channel_id (str): The ID of the channel the video belongs to.
        video_id (str): The unique ID of the video.
        title (str): The title of the video.
        end_time (datetime): The end time of the video.
        duration (int): The duration of the video in seconds.
        has_chat_log (bool, optional): Flag indicating if the video has an associated chat log. Defaults to False.
    """

    conn = None

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO videos (video_id, channel_id, title, end_time, duration, processed_at, has_chat_log)
            VALUES (%s, %s, %s, %s, make_interval(secs => %s), NOW(), %s)
            ON CONFLICT (video_id) DO UPDATE 
            SET title = EXCLUDED.title, 
                end_time = EXCLUDED.end_time, 
                duration = EXCLUDED.duration,
                processed_at = NOW(),
                has_chat_log = EXCLUDED.has_chat_log;
            """,
            (video_id, channel_id, title, end_time, duration, has_chat_log)
        )
        conn.commit()
    except psycopg2.OperationalError as e:
        logger.error(f"Error connecting to database: {e}")
        if conn:
            conn.close()
        raise
    except psycopg2.DatabaseError as e:
        logger.error(f"Error inserting video metadata: {e}")
        conn.rollback()
        conn.close()
        conn = get_db_connection()
    finally:
        if conn:
            try:
                cursor.close()
                release_db_connection(conn)
            except Exception as e:
                logger.error(f"Error releasing DB connection: {e}")

def insert_batches(cursor, user_data_batch, user_batch, chat_stats_batch):
    execute_values(cursor, """
                   INSERT INTO user_data (user_id, channel_id, last_message_at, video_id, membership_rank, jp_count, kr_count, ru_count, emoji_count, es_en_id_count, total_message_count)
                VALUES %s
                ON CONFLICT (user_id, channel_id, last_message_at, video_id) DO UPDATE
                SET last_message_at = EXCLUDED.last_message_at,
                    membership_rank = EXCLUDED.membership_rank,
                    jp_count = user_data.jp_count + EXCLUDED.jp_count,
                    kr_count = user_data.kr_count + EXCLUDED.kr_count,
                    ru_count = user_data.ru_count + EXCLUDED.ru_count,
                    emoji_count = user_data.emoji_count + EXCLUDED.emoji_count,
                    es_en_id_count = user_data.es_en_id_count + EXCLUDED.es_en_id_count,
                    total_message_count = user_data.total_message_count + EXCLUDED.total_message_count;
                """, user_data_batch)
    
    execute_values(cursor, """
                   INSERT INTO users (user_id, username)
                VALUES %s
                ON CONFLICT (user_id) DO UPDATE
                SET username = EXCLUDED.username;
                """, user_batch)
    
    execute_values(cursor, """
                   INSERT INTO chat_language_stats (channel_id, observed_month, jp_count, kr_count, ru_count, emoji_count, es_en_id_count, total_messages)
                VALUES %s
                ON CONFLICT (channel_id, observed_month) DO UPDATE
                SET jp_count = chat_language_stats.jp_count + EXCLUDED.jp_count,
                    kr_count = chat_language_stats.kr_count + EXCLUDED.kr_count,
                    ru_count = chat_language_stats.ru_count + EXCLUDED.ru_count,
                    emoji_count = chat_language_stats.emoji_count + EXCLUDED.emoji_count,
                    es_en_id_count = chat_language_stats.es_en_id_count + EXCLUDED.es_en_id_count,
                    total_messages = chat_language_stats.total_messages + EXCLUDED.total_messages;
                """, chat_stats_batch)
    
# Check if metadata is processed into database for a given video id
def is_metadata_processed(video_id):
    logger = get_logger()
    """Check if metadata is processed into database for a given video id
    
    Args:
        video_id (str): The ID of the video to check.
    
    Returns:
        bool: True if the metadata is processed, False otherwise.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT EXISTS (SELECT 1 FROM videos WHERE video_id = %s)", (video_id,))
        result = cursor.fetchone()
        return result[0]
    except psycopg2.DatabaseError as e:
        logger.error(f"Database error in is_chat_log_processed: {e}")
        return False
    finally:
        cursor.close()
        release_db_connection(conn)


# Check if chat log is processed into database for a given video id
def is_chat_log_processed(video_id):
    logger = get_logger()
    """Check if chat log is processed into the database for a given video ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT EXISTS (SELECT 1 FROM user_data WHERE video_id = %s)", (video_id,))
        result = cursor.fetchone()
        return result[0]
    except psycopg2.DatabaseError as e:
        logger.error(f"Database error in is_chat_log_processed: {e}")
        return False
    finally:
        cursor.close()
        release_db_connection(conn)