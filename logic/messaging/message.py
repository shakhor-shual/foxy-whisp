from enum import Enum, auto
from typing import Optional, Dict, Any, Union
from multiprocessing import Queue
import time
from .validator import MessageValidator, MessageSource

class MessageType(Enum):
    LOG = auto()
    STATUS = auto()
    DATA = auto()
    COMMAND = auto()
    CONTROL = auto()

class PipelineMessage:
    """Централизованное определение формата сообщений"""
    def __init__(self, source: str, type: MessageType, content: Dict[str, Any]):
        self.source = MessageSource.normalize(source)
        self.type = type
        self.content = content
        self.timestamp = time.time()
        
        if not MessageValidator.validate_message(self):
            raise ValueError(f"Invalid message format for type {type}")

    @classmethod
    def create_log(cls, source: str, message: str, level: str = "info", **kwargs):
        """Create log message with validation"""
        content = {
            'message': message,
            'level': level.lower(),
            **kwargs
        }
        return cls(source, MessageType.LOG, content)

    # ...existing create methods...
