from db.queries import insert_batches
from db.connection import get_db_connection, release_db_connection
from utils.helpers import deduplicate_batch


def db_worker(queue):
    """Worker process that takes user data from the queue and inserts it into the database."""
    conn = get_db_connection()
    cursor = conn.cursor()

    batch = 100
    user_data_batch = []
    user_batch = []
    chat_stats_batch = []

    while True:
        task = queue.get()
        if task is None:
            break

        (user_id, username, channel_id, last_message_at, video_id, membership_rank,
         jp_count, kr_count, ru_count, emoji_count, es_en_id_count, total_message_count, 
         observed_month, is_gift) = task  # NEW: Added is_gift
        
        user_data_batch.append((user_id, channel_id, last_message_at, video_id, membership_rank,
                                jp_count, kr_count, ru_count, emoji_count, es_en_id_count, 
                                total_message_count, is_gift))  # NEW: Added is_gift
        user_batch.append((user_id, username))
        chat_stats_batch.append((channel_id, observed_month, jp_count, kr_count, ru_count, 
                                 emoji_count, es_en_id_count, total_message_count))

        user_data_batch = deduplicate_batch(user_data_batch, key_indices=(0, 1, 2, 3))
        user_batch = deduplicate_batch(user_batch, key_indices=(0,))
        chat_stats_batch = deduplicate_batch(chat_stats_batch, key_indices=(0, 1))

        if len(user_data_batch) >= batch:
            insert_batches(cursor, user_data_batch, user_batch, chat_stats_batch)
            conn.commit()
            user_data_batch.clear()
            user_batch.clear()
            chat_stats_batch.clear()
    
    if user_data_batch:
        user_data_batch = deduplicate_batch(user_data_batch, key_indices=(0, 1, 2, 3))
        user_batch = deduplicate_batch(user_batch, key_indices=(0,))
        chat_stats_batch = deduplicate_batch(chat_stats_batch, key_indices=(0, 1))
        insert_batches(cursor, user_data_batch, user_batch, chat_stats_batch)
        conn.commit()

    cursor.close()
    release_db_connection(conn)