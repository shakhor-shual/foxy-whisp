from typing import Optional, Dict, Any, Union
from multiprocessing import Queue
import time
from .messaging.types import MessageType, MessageSource
from .messaging.validator import MessageValidator

class PipelineMessage:
    def __init__(self, source: str, type: MessageType, content: Dict[str, Any]):
        self.source = MessageSource.normalize(source)
        self.type = type
        self.content = content
        self.timestamp = time.time()
        
        if not MessageValidator.validate_message(self):
            raise ValueError(f"Invalid message format for type {type}")

    @classmethod
    def create_log(cls, source: str, message: str, level: str = "info", **kwargs) -> 'PipelineMessage':
        """Create a log message with enhanced validation"""
        content = {
            'message': message,
            'level': level.lower(),
            'timestamp': time.time(),
            'source_details': {
                'component': kwargs.pop('component', 'main'),
                'context': kwargs
            }
        }
        return cls(source, MessageType.LOG, content)

    @classmethod
    def create_status(cls, source: str, status: str, **details) -> 'PipelineMessage':
        """Create a status message with metadata"""
        content = {
            'status': status,
            'timestamp': time.time(),
            'details': details,
            'metadata': {
                'source_type': source.split('.')[0],
                'component': details.get('component', 'main')
            }
        }
        return cls(source, MessageType.STATUS, content)

    @classmethod
    def create_data(cls, source: str, data_type: str, data: Any, **metadata) -> 'PipelineMessage':
        """Create a data message with validation"""
        content = {
            'data_type': data_type,
            'payload': data,
            'timestamp': time.time(),
            'metadata': {
                **metadata,
                'source_type': source.split('.')[0]
            }
        }
        return cls(source, MessageType.DATA, content)

    @classmethod
    def create_command(cls, source: str, command: str, **params) -> 'PipelineMessage':
        """Create a command message"""
        content = {
            'command': command,
            'params': params
        }
        return cls(source, MessageType.COMMAND, content)

    def send(self, queue: Queue) -> bool:
        """Send message to queue with validation"""
        if queue is None:
            return False
        try:
            queue.put(self)
            return True
        except:
            return False

    @staticmethod
    def receive(queue: Queue, timeout: float = 0.1) -> Optional['PipelineMessage']:
        """Receive message from queue with validation"""
        if queue is None:
            return None
        try:
            msg = queue.get(timeout=timeout)
            if isinstance(msg, PipelineMessage):
                return msg
        except:
            pass
        return None

    def is_log(self) -> bool:
        return self.type == MessageType.LOG

    def is_status(self) -> bool:
        return self.type == MessageType.STATUS

    def is_data(self) -> bool:
        return self.type == MessageType.DATA

    def is_command(self) -> bool:
        return self.type == MessageType.COMMAND

    def is_control(self) -> bool:
        return self.type == MessageType.CONTROL