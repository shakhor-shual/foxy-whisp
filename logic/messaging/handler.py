from multiprocessing import Queue as MPQueue
from typing import Optional, Dict, Any, Callable
import logging
import queue
from .message import PipelineMessage, MessageType
from .formatter import MessageFormatter
from .validator import MessageSource

class MessageHandler:
    """Централизованный обработчик сообщений"""
    
    def __init__(self, gui_to_server: MPQueue, server_to_gui: MPQueue):
        self.gui_to_server = gui_to_server
        self.server_to_gui = server_to_gui
        self.logger = logging.getLogger('gui.messages')
        self.formatter = MessageFormatter()
        self.callbacks = {
            'log': [],
            'status': [],
            'data': [],
            'vad': [],
            'audio_level': [],
            'button': [],
            'device_info': []
        }
        self.message_queue = queue.Queue()

    # ...existing methods...
