import unittest
from unittest.mock import Mock, patch, MagicMock
import logging
from multiprocessing import Queue
import time
from logic.pipeline_logging import LoggingMixin, QueueLoggerHandler

class TestClass(LoggingMixin):
    """Test implementation of LoggingMixin"""
    def __init__(self):
        self.__class__.__name__ = "TestLogger"

class TestQueueLoggerHandler(unittest.TestCase):
    def setUp(self):
        # Create mock queue with proper put method using MagicMock
        self.log_queue = MagicMock()
        self.handler = QueueLoggerHandler(self.log_queue)
        
    def test_emit_success(self):
        """Test successful log emission"""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        self.handler.emit(record)
        self.log_queue.put.assert_called_once()
        
    def test_emit_error(self):
        """Test error handling in emit"""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        self.log_queue.put.side_effect = Exception("Test error")
        with patch.object(self.handler, 'handleError') as mock_handle_error:
            self.handler.emit(record)
            mock_handle_error.assert_called_once_with(record)

class TestLoggingMixin(unittest.TestCase):
    def setUp(self):
        self.test_obj = TestClass()
        # Create mock queue with proper put method using MagicMock
        self.log_queue = MagicMock()
        
    def test_setup_logger(self):
        """Test logger setup"""
        self.test_obj.setup_logger(self.log_queue)
        
        # Verify logger configuration
        self.assertEqual(self.test_obj.logger.name, "TestLogger")
        self.assertEqual(self.test_obj.logger.level, logging.INFO)
        self.assertEqual(len(self.test_obj.logger.handlers), 1)
        self.assertIsInstance(self.test_obj.logger.handlers[0], QueueLoggerHandler)
        
    def test_setup_logger_removes_existing_handlers(self):
        """Test that existing handlers are removed"""
        # Add a test handler first
        self.test_obj.setup_logger(self.log_queue)
        initial_handler = self.test_obj.logger.handlers[0]
        
        # Setup logger again
        self.test_obj.setup_logger(self.log_queue)
        
        # Verify old handler was removed
        self.assertNotIn(initial_handler, self.test_obj.logger.handlers)
        self.assertEqual(len(self.test_obj.logger.handlers), 1)
        
    def test_log_message(self):
        """Test log message creation and sending"""
        self.test_obj.setup_logger(self.log_queue)
        
        test_message = "Test log message"
        test_level = "info"
        test_extra = {"key": "value"}
        
        with patch.object(self.test_obj.logger, test_level) as mock_log:
            self.test_obj.log_message(test_level, test_message, test_extra)
            
            # Verify log call
            mock_log.assert_called_once()
            log_entry = mock_log.call_args[0][0]
            
            # Verify log entry contents
            self.assertEqual(log_entry['source'], 'testlogger')
            self.assertEqual(log_entry['type'], 'log')
            self.assertEqual(log_entry['level'], test_level)
            self.assertEqual(log_entry['message'], test_message)
            self.assertEqual(log_entry['extra'], test_extra)
            self.assertIn('timestamp', log_entry)
            
    def test_log_message_without_extra(self):
        """Test log message without extra data"""
        self.test_obj.setup_logger(self.log_queue)
        
        test_message = "Test log message"
        test_level = "info"
        
        with patch.object(self.test_obj.logger, test_level) as mock_log:
            self.test_obj.log_message(test_level, test_message)
            
            # Verify log call
            mock_log.assert_called_once()
            log_entry = mock_log.call_args[0][0]
            
            # Verify log entry contents
            self.assertNotIn('extra', log_entry)
            
    def test_log_message_without_logger(self):
        """Test log message behavior when logger is not set up"""
        test_message = "Test log message"
        test_level = "info"
        
        # Should not raise any exceptions
        self.test_obj.log_message(test_level, test_message)

if __name__ == '__main__':
    unittest.main()
