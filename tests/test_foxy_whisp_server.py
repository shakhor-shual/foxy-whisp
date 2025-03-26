import pytest
from unittest.mock import Mock, patch, MagicMock, call
from multiprocessing import Queue, Event as MPEvent
from foxy_whisp_server import FoxyWhispServer, PipelineMessage, QueueHandler
import logging

@pytest.fixture
def mock_queues():
    """Create properly mocked queues with required methods"""
    from_gui = MagicMock()
    from_gui.get.return_value = None
    from_gui.put.return_value = None
    to_gui = MagicMock()
    to_gui.get.return_value = None
    to_gui.put.return_value = None
    return {
        'from_gui': from_gui,
        'to_gui': to_gui,
    }

@pytest.fixture
def server(mock_queues):
    with patch('signal.signal'):
        server = FoxyWhispServer(
            from_gui=mock_queues['from_gui'],
            to_gui=mock_queues['to_gui'],
            args={'chunk_size': 512}
        )
        # Ensure queues are properly set
        server.queues.from_gui = mock_queues['from_gui']
        server.queues.to_gui = mock_queues['to_gui']
    return server

def test_server_initialization(server):
    assert not server._shutdown_requested
    # Correct the isinstance check for stop_event
    assert isinstance(server.stop_event, type(MPEvent()))
    assert server.pipe_chunk == 512

def test_send_gui_message(server):
    test_msg = PipelineMessage.create_log(
        source='test',
        message='test message',
        level='info'
    )
    server._send_gui_message(test_msg)
    server.queues.to_gui.put.assert_called_once()

def test_handle_gui_disconnect(server):
    with patch('logging.getLogger') as mock_get_logger:
        mock_root_logger = MagicMock()
        mock_get_logger.return_value = mock_root_logger

        # Simulate existing handlers
        mock_root_logger.handlers = [MagicMock(spec=logging.Handler)]

        server._handle_gui_disconnect()

        # Verify QueueHandlers were removed
        assert all(
            not isinstance(handler, QueueHandler)
            for handler in mock_root_logger.handlers
        )

        # Verify a StreamHandler was added
        stream_handler_added = any(
            isinstance(handler, logging.StreamHandler)
            for handler in mock_root_logger.handlers
        )
        assert stream_handler_added, "StreamHandler was not added to the logger"

        # Verify the StreamHandler's configuration
        added_handler = next(
            (handler for handler in mock_root_logger.handlers if isinstance(handler, logging.StreamHandler)),
            None
        )
        assert added_handler is not None
        assert added_handler.level == logging.INFO

        # Verify the logger logged the test message
        mock_root_logger.info.assert_called_with("StreamHandler added to logger")

@pytest.mark.parametrize("msg_type,handler_method", [
    ('log', '_handle_log'),
    ('status', '_handle_status'),
    ('data', '_handle_data'),
    ('command', '_handle_command'),
    ('control', '_handle_control')
])
def test_message_handling(server, msg_type, handler_method):
    with patch.object(server, handler_method) as mock_handler:
        msg = MagicMock()
        # Setup all is_* methods to return False by default
        for method in ['is_log', 'is_status', 'is_data', 'is_command', 'is_control']:
            setattr(msg, method, Mock(return_value=False))
        # Set the tested method to return True
        getattr(msg, f'is_{msg_type}').return_value = True
        
        server._handle_message(msg)
        mock_handler.assert_called_once_with(msg)

def test_start_pipeline(server):
    with patch('foxy_whisp_server.SRCstage') as mock_src, \
         patch('foxy_whisp_server.ASRstage') as mock_asr:
        
        mock_src_instance = mock_src.return_value
        mock_asr_instance = mock_asr.return_value
        
        server.start_pipeline()
        
        mock_src.assert_called_once()
        mock_asr.assert_called_once()
        assert server.processes['src'] is mock_src_instance
        assert server.processes['asr'] is mock_asr_instance
        mock_src_instance.start.assert_called_once()
        mock_asr_instance.start.assert_called_once()

def test_stop_pipeline(server):
    server.processes['src'] = Mock()
    server.processes['asr'] = Mock()
    
    server.stop_pipeline()
    
    assert server.stop_event.is_set()
    assert server.processes['src'].join.called
    assert server.processes['asr'].join.called

def test_cleanup(server):
    # Create a mock queue with all required methods
    mock_queue = Mock(spec=Queue)
    mock_queue.close = Mock()
    mock_queue.join_thread = Mock()
    
    server.queues.test_queue = mock_queue
    server._cleanup()
    
    mock_queue.close.assert_called_once()
    mock_queue.join_thread.assert_called_once()

def test_handle_signal(server):
    with patch('logging.getLogger'):
        server._handle_signal(15, None)  # SIGTERM
        
        assert server._shutdown_requested
        assert server.stop_event.is_set()

def test_handle_command_shutdown(server):
    msg = PipelineMessage.create_command(
        source='gui',
        command='shutdown'
    )
    
    server._handle_command(msg)
    
    assert server._shutdown_requested

@pytest.mark.parametrize("log_level", ['info', 'warning', 'error', 'critical'])
def test_handle_log_levels(server, log_level):
    with patch('foxy_whisp_server.logger') as mock_logger:
        msg = PipelineMessage.create_log(
            source='test',
            message='test message',
            level=log_level
        )

        server._handle_log(msg)

        log_level_value = getattr(logging, log_level.upper())
        mock_logger.log.assert_called_once_with(log_level_value, '[test] test message')

def test_run(server):
    from queue import Empty
    
    def queue_get_with_timeout(*args, **kwargs):
        if not server._shutdown_requested:
            server._shutdown_requested = True
            return PipelineMessage.create_command(
                source='test',
                command='shutdown'
            )
        raise Empty()

    server.queues.from_gui.get = Mock(side_effect=queue_get_with_timeout)
    
    # Run server with timeout
    with patch.object(server, '_handle_message') as mock_handle:
        server.run()
        
        # Verify shutdown was requested
        assert server._shutdown_requested
        # Verify message was handled
        mock_handle.assert_called_once()
