from enum import Enum, auto
from typing import Optional, Dict, Any, Union
from multiprocessing import Queue
import time
from .message_validator import MessageValidator, MessageSource

class MessageType(Enum):
    LOG = auto()
    STATUS = auto()
    DATA = auto()
    COMMAND = auto()
    CONTROL = auto()

class PipelineMessage:
    def __init__(self, source: str, type: MessageType, content: Dict[str, Any]):
        self.source = MessageSource.normalize(source)
        self.type = type
        self.content = content
        self.timestamp = time.time()
        
        if not self._validate():
            raise ValueError(f"Invalid message format for type {type}")

    def _validate(self) -> bool:
        """Validate message format"""
        if not MessageValidator.validate_source(self.source):
            return False
            
        if not isinstance(self.content, dict):
            return False
            
        validators = {
            MessageType.LOG: MessageValidator.validate_log_content,
            MessageType.STATUS: MessageValidator.validate_status_content,
            MessageType.DATA: MessageValidator.validate_data_content,
            MessageType.COMMAND: MessageValidator.validate_command_content
        }
        
        return validators.get(self.type, lambda x: True)(self.content)

    @classmethod
    def create_log(cls, source: str, message: str, level: str = "info", **kwargs) -> 'PipelineMessage':
        """Create a log message"""
        content = {
            'message': message,
            'level': level.lower(),
            **kwargs
        }
        return cls(source, MessageType.LOG, content)

    @classmethod
    def create_status(cls, source: str, status: str, **details) -> 'PipelineMessage':
        """Create a status message"""
        content = {
            'status': status,
            'details': details
        }
        return cls(source, MessageType.STATUS, content)

    @classmethod
    def create_data(cls, source: str, data_type: str, data: Any, **metadata) -> 'PipelineMessage':
        """Create a data message"""
        content = {
            'data_type': data_type,
            'payload': data,
            **metadata
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