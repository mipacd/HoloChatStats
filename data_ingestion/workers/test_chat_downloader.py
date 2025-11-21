import unittest
from unittest.mock import Mock, patch
from workers.chat_downloader import download_chat_log


class TestDownloadChatLog(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.channel_id = "UCgnfPPb9JI3e9A4cXHnWbyg"
        self.video_id = "HNpT88tiKLM3"
        # Create a mock queue instead of a real Queue
        self.queue = Mock()
        self.queue.qsize.return_value = 0
        self.queue.full.return_value = False  # Important: Always return False for full()
        self.queue_items = []  # Store items that would be put in queue
        
        # Mock the put method to store items in our list
        def mock_put(item):
            self.queue_items.append(item)
            self.queue.qsize.return_value = len(self.queue_items)
        
        self.queue.put = mock_put
        self.year = 2024
        self.month = 1
        
    @patch('workers.chat_downloader.iter_youtube_chat')
    @patch('workers.chat_downloader.get_db_connection')
    @patch('workers.chat_downloader.release_db_connection')
    @patch('workers.chat_downloader.write_chat_log_to_cache')
    @patch('workers.chat_downloader.update_feature_timestamps')
    @patch('workers.chat_downloader.get_feature_timestamps')
    @patch('workers.chat_downloader.parse_membership_rank')
    @patch('workers.chat_downloader.categorize_message')
    @patch('workers.chat_downloader.has_humor')
    @patch('workers.chat_downloader.get_config')
    @patch('workers.chat_downloader.get_logger')
    def test_download_with_regular_messages(
        self, mock_logger, mock_config, mock_humor, mock_categorize,
        mock_parse_rank, mock_get_features, mock_get_conn, mock_iter_chat
    ):
        """Test downloading chat log with regular chat messages"""
        # Setup mocks
        mock_logger.return_value = Mock()
        mock_config.return_value = "3"
        mock_parse_rank.return_value = 1
        mock_categorize.return_value = "es_en_id"
        mock_humor.return_value = False
        mock_get_features.return_value = {}
        
        # Mock database connection
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn
        
        # Mock chat messages
        test_messages = [
            {
                "author": {
                    "id": "user_001",
                    "name": "@TestUser1",
                    "badges": ["Member (1 month)"]
                },
                "message": "Hello world!",
                "timestamp": 1000.0,
                "message_type": "chat"
            },
            {
                "author": {
                    "id": "user_002",
                    "name": "@TestUser2",
                    "badges": []
                },
                "message": "Nice stream",
                "timestamp": 2000.0,
                "message_type": "chat"
            }
        ]
        mock_iter_chat.return_value = iter(test_messages)
        
        # Execute function
        download_chat_log(self.channel_id, self.video_id, self.queue, self.year, self.month)
        
        # Verify queue has correct data
        self.assertEqual(len(self.queue_items), 2)
        
        # Sort items by user_id for deterministic testing
        sorted_items = sorted(self.queue_items, key=lambda x: x[0])
        
        # Check first queue item (user_001)
        item1 = sorted_items[0]
        self.assertEqual(item1[0], "user_001")  # user_id
        self.assertEqual(item1[1], "@TestUser1")  # username
        self.assertEqual(item1[2], self.channel_id)  # channel_id
        self.assertEqual(item1[5], 1)  # membership_rank
        
        # Check second queue item (user_002)
        item2 = sorted_items[1]
        self.assertEqual(item2[0], "user_002")  # user_id
        self.assertEqual(item2[1], "@TestUser2")  # username
        
        # Verify database update was called
        mock_cursor.execute.assert_called()
        mock_conn.commit.assert_called_once()
        
    @patch('workers.chat_downloader.iter_youtube_chat')
    @patch('workers.chat_downloader.get_db_connection')
    @patch('workers.chat_downloader.release_db_connection')
    @patch('workers.chat_downloader.write_chat_log_to_cache')
    @patch('workers.chat_downloader.update_feature_timestamps')
    @patch('workers.chat_downloader.get_feature_timestamps')
    @patch('workers.chat_downloader.parse_membership_rank')
    @patch('workers.chat_downloader.categorize_message')
    @patch('workers.chat_downloader.has_humor')
    @patch('workers.chat_downloader.get_config')
    @patch('workers.chat_downloader.get_logger')
    def test_download_with_gift_membership(
        self, mock_logger, mock_config, mock_humor, mock_categorize,
        mock_parse_rank, mock_get_features, 
        mock_write_cache,  mock_get_conn, mock_iter_chat
    ):
        """Test downloading chat log with gift membership notifications"""
        # Reset queue items for this test
        self.queue_items = []
        
        # Setup mocks
        mock_logger.return_value = Mock()
        mock_config.return_value = "3"
        mock_parse_rank.return_value = 1
        mock_categorize.return_value = "es_en_id"
        mock_humor.return_value = False
        mock_get_features.return_value = {}
        
        # Mock database connection
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn
        
        # Mock gift membership message
        test_messages = [
            {
                "author": {
                    "id": "user_003",
                    "name": "@GiftRecipient",
                    "badges": []
                },
                "message": "",
                "timestamp": 1500.0,
                "message_type": "gift_member",
                "gifter": "@GenerousUser"
            }
        ]
        mock_iter_chat.return_value = iter(test_messages)
        
        # Execute function
        download_chat_log(self.channel_id, self.video_id, self.queue, self.year, self.month)
        
        # Verify queue has correct data
        self.assertEqual(len(self.queue_items), 1)
        
        # Check queue item
        item = self.queue_items[0]
        self.assertEqual(item[0], "user_003")  # user_id
        self.assertEqual(item[1], "@GiftRecipient")  # username
        self.assertEqual(item[5], 0)  # membership_rank should be 0 for gift with no badges
        
        # Verify cache was written with gift info
        mock_write_cache.assert_called_once()
        cache_entries = mock_write_cache.call_args[0][2]
        self.assertEqual(len(cache_entries), 1)
        self.assertEqual(cache_entries[0]["message_type"], "gift_member")
        self.assertEqual(cache_entries[0]["gifter"], "@GenerousUser")
        
    @patch('workers.chat_downloader.iter_youtube_chat')
    @patch('workers.chat_downloader.get_db_connection')
    @patch('workers.chat_downloader.release_db_connection')
    @patch('workers.chat_downloader.write_chat_log_to_cache')
    @patch('workers.chat_downloader.update_feature_timestamps')
    @patch('workers.chat_downloader.get_feature_timestamps')
    @patch('workers.chat_downloader.parse_membership_rank')
    @patch('workers.chat_downloader.categorize_message')
    @patch('workers.chat_downloader.has_humor')
    @patch('workers.chat_downloader.get_config')
    @patch('workers.chat_downloader.get_logger')
    def test_download_with_new_member(
        self, mock_logger, mock_config, mock_humor, mock_categorize,
        mock_parse_rank, mock_get_features,
        mock_write_cache, mock_get_conn, mock_iter_chat
    ):
        """Test downloading chat log with new member notifications"""
        # Reset queue items for this test
        self.queue_items = []
        
        # Setup mocks
        mock_logger.return_value = Mock()
        mock_config.return_value = "3"
        mock_parse_rank.return_value = 1
        mock_categorize.return_value = "es_en_id"
        mock_humor.return_value = False
        mock_get_features.return_value = {}
        
        # Mock database connection
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn
        
        # Mock new member message
        test_messages = [
            {
                "author": {
                    "id": "user_004",
                    "name": "@NewMember",
                    "badges": ["Member (1 month)"]
                },
                "message": "",
                "timestamp": 1800.0,
                "message_type": "new_member"
            }
        ]
        mock_iter_chat.return_value = iter(test_messages)
        
        # Execute function
        download_chat_log(self.channel_id, self.video_id, self.queue, self.year, self.month)
        
        # Verify queue has correct data
        self.assertEqual(len(self.queue_items), 1)
        
        # Check queue item
        item = self.queue_items[0]
        self.assertEqual(item[0], "user_004")  # user_id
        self.assertEqual(item[1], "@NewMember")  # username
        self.assertEqual(item[5], 1)  # membership_rank from badge
        
        # Verify cache was written
        mock_write_cache.assert_called_once()
        cache_entries = mock_write_cache.call_args[0][2]
        self.assertEqual(len(cache_entries), 1)
        self.assertEqual(cache_entries[0]["message_type"], "new_member")
        self.assertEqual(cache_entries[0]["message"], "")
        
    @patch('workers.chat_downloader.iter_youtube_chat')
    @patch('workers.chat_downloader.get_config')
    @patch('workers.chat_downloader.get_logger')
    def test_download_with_runtime_error(
        self, mock_logger, mock_config, mock_iter_chat
    ):
        """Test handling of RuntimeError when no chat replay is available"""
        # Reset queue items for this test
        self.queue_items = []
        
        # Setup mocks
        mock_logger_instance = Mock()
        mock_logger.return_value = mock_logger_instance
        mock_config.return_value = "3"
        
        # Mock RuntimeError
        mock_iter_chat.side_effect = RuntimeError("No chat replay available")
        
        # Execute function
        download_chat_log(self.channel_id, self.video_id, self.queue, self.year, self.month)
        
        # Verify warning was logged
        mock_logger_instance.warning.assert_called_with(
            f"No chat replay found for {self.video_id}."
        )
        
        # Verify queue is empty
        self.assertEqual(len(self.queue_items), 0)
    
    @patch('workers.chat_downloader.iter_youtube_chat')
    @patch('workers.chat_downloader.get_db_connection')
    @patch('workers.chat_downloader.release_db_connection')
    @patch('workers.chat_downloader.write_chat_log_to_cache')
    @patch('workers.chat_downloader.update_feature_timestamps')
    @patch('workers.chat_downloader.get_feature_timestamps')
    @patch('workers.chat_downloader.parse_membership_rank')
    @patch('workers.chat_downloader.categorize_message')
    @patch('workers.chat_downloader.has_humor')
    @patch('workers.chat_downloader.get_config')
    @patch('workers.chat_downloader.get_logger')
    @patch('time.sleep')  # Mock sleep to speed up test
    def test_download_with_queue_full(
        self, mock_sleep, mock_logger, mock_config, mock_humor, mock_categorize,
        mock_parse_rank, mock_get_features, mock_get_conn, mock_iter_chat
    ):
        """Test behavior when queue is full"""
        # Reset queue items for this test
        self.queue_items = []
        full_calls = []
        
        def mock_full():
            # Track calls and return True only once, then False
            full_calls.append(True)
            return len(full_calls) == 1
        
        # Override the full method for this test
        self.queue.full = mock_full
        
        # Setup mocks
        mock_logger_instance = Mock()
        mock_logger.return_value = mock_logger_instance
        mock_config.return_value = "3"
        mock_parse_rank.return_value = 1
        mock_categorize.return_value = "es_en_id"
        mock_humor.return_value = False
        mock_get_features.return_value = {}
        
        # Mock database connection
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_get_conn.return_value = mock_conn
        
        # Use two users to ensure we iterate through the loop multiple times
        test_messages = [
            {
                "author": {
                    "id": "user_005",
                    "name": "@TestUser5",
                    "badges": []
                },
                "message": "Test message 1",
                "timestamp": 1000.0,
                "message_type": "chat"
            },
            {
                "author": {
                    "id": "user_006",
                    "name": "@TestUser6",
                    "badges": []
                },
                "message": "Test message 2",
                "timestamp": 2000.0,
                "message_type": "chat"
            }
        ]
        mock_iter_chat.return_value = iter(test_messages)
        
        # Execute function
        download_chat_log(self.channel_id, self.video_id, self.queue, self.year, self.month)
        
        # Verify that the "Queue full" message was logged
        mock_logger_instance.info.assert_any_call("Queue full. Waiting for space...")
        
        # Verify sleep was called when queue was full
        mock_sleep.assert_called_with(1)
        
        # Verify both items were eventually added (after queue was no longer full)
        # The first user triggers the full queue, waits, then continues
        # The second user should be added normally
        self.assertEqual(len(self.queue_items), 2)
        
        # Verify we had at least 2 calls to queue.full() 
        self.assertGreaterEqual(len(full_calls), 2)


if __name__ == '__main__':
    # Run the unit tests
    unittest.main()