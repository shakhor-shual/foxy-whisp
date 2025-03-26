import unittest
from unittest.mock import Mock, patch, MagicMock
from multiprocessing import Event, Queue
import logging
from logic.pipeline_element import PipelineElement

class TestPipeElement(PipelineElement):
    """Test implementation of PipelineElement"""
    def process_data(self, data):
        if data == "raise_error":
            raise ValueError("Test error")
        # Send processed data through the pipeline
        self.send_data(f"processed_{data}")
        return f"processed_{data}"

class TestPipelineElement(unittest.TestCase):
    def setUp(self):
        # Suppress logging during tests
        logging.disable(logging.ERROR)
        
        # Create mocks with required attributes
        self.stop_event = Mock()
        self.stop_event.is_set = Mock(return_value=False)
        self.stop_event.set = Mock()

        self.in_queue = Mock()
        self.in_queue.get = Mock()
        self.in_queue.empty = Mock(return_value=False)
        
        self.out_queue = Mock()
        self.out_queue.put = Mock()
        
        self.element = TestPipeElement(
            self.stop_event,
            self.in_queue,
            self.out_queue
        )

    def tearDown(self):
        # Re-enable logging after tests
        logging.disable(logging.NOTSET)

    def test_init(self):
        """Test initialization"""
        self.assertEqual(self.element.name, "TestPipeElement")
        self.assertEqual(self.element.stop_event, self.stop_event)
        self.assertEqual(self.element.in_queue, self.in_queue)
        self.assertEqual(self.element.out_queue, self.out_queue)

    def test_pipe_read(self):
        """Test reading from input queue"""
        test_data = "test_data"
        self.in_queue.get.return_value = test_data
        
        result = self.element.pipe_read()
        self.assertEqual(result, test_data)
        self.in_queue.get.assert_called_once()

    def test_pipe_read_error(self):
        """Test reading error handling"""
        self.in_queue.get.side_effect = Exception("Test error")
        
        result = self.element.pipe_read()
        self.assertIsNone(result)

    def test_pipe_write(self):
        """Test writing to output queue"""
        test_data = "test_data"
        
        result = self.element.pipe_write(test_data)
        self.assertTrue(result)
        self.out_queue.put.assert_called_once_with(test_data)

    def test_pipe_write_error(self):
        """Test writing error handling"""
        self.out_queue.put.side_effect = Exception("Test error")
        
        result = self.element.pipe_write("test_data")
        self.assertFalse(result)

    def test_process_control_commands_stop(self):
        """Test processing stop command"""
        msg = Mock()
        msg.content = {"command": "stop"}
        
        result = self.element.process_control_commands(msg)
        self.assertTrue(result)
        self.stop_event.set.assert_called_once()

    def test_process_control_commands_other(self):
        """Test processing non-stop command"""
        msg = Mock()
        msg.content = {"command": "other"}
        
        result = self.element.process_control_commands(msg)
        self.assertFalse(result)
        self.stop_event.set.assert_not_called()

    def test_process_control_commands_error(self):
        """Test command processing error handling"""
        msg = Mock()
        msg.content = None  # Will cause AttributeError
        
        result = self.element.process_control_commands(msg)
        self.assertFalse(result)

    def test_send_data(self):
        """Test sending data message"""
        test_data = "test_data"
        
        self.element.send_data(test_data)
        self.out_queue.put.assert_called_once_with({
            'type': 'data',
            'payload': test_data
        })

    def test_send_status(self):
        """Test sending status message"""
        test_status = "test_status"
        
        self.element.send_status(test_status)
        self.out_queue.put.assert_called_once_with({
            'type': 'status',
            'status': test_status
        })

    def test_run(self):
        """Test main processing loop"""
        # Setup mocks
        self.stop_event.is_set.side_effect = [False, True]  # Run once then stop
        test_data = "test_data"
        self.in_queue.get.return_value = test_data
        
        # Run the pipeline element
        self.element.run()
        
        # Verify interactions
        self.in_queue.get.assert_called_once()
        # The send_data method will be called by process_data
        self.out_queue.put.assert_called_once_with({
            'type': 'data',
            'payload': f'processed_{test_data}'
        })

    def test_run_error_handling(self):
        """Test error handling in run loop"""
        self.stop_event.is_set.side_effect = [False, True]
        self.in_queue.get.return_value = "raise_error"
        
        self.element.run()
        
        self.stop_event.set.assert_called_once()

if __name__ == '__main__':
    unittest.main()
