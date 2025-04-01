from multiprocessing import Queue as MPQueue
from typing import Optional, Dict, Any, Callable
from .types import MessageType, MessageSource
from ..foxy_message import PipelineMessage
import logging

class BaseMessageHandler:
    """Base class for message handling functionality"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.source_mappings = {
            'srcstage': 'src',
            'asrstage': 'asr',
            'system': 'server',
            'audio_device': 'src',
            'tcp': 'src',
            'vad': 'src'
        }

    def _format_message(self, text: str) -> Optional[tuple]:
        """Common message formatting logic"""
        try:
            if text.startswith('[') and ']' in text:
                parts = text[1:].split(']', 1)
                if len(parts) == 2:
                    source_parts = parts[0].lower().split('.')
                    message = parts[1].strip()
                    
                    base_source = source_parts[0]
                    level = source_parts[-1] if len(source_parts) > 1 else 'info'
                    component = '.'.join(source_parts[1:-1]) if len(source_parts) > 2 else 'main'
                    
                    base_source = self.source_mappings.get(base_source, base_source)
                    return base_source, component, level, message
            return None
        except Exception as e:
            self.logger.error(f"Message parsing error: {e}, Text: {text}")
            return None

    def _send_message(self, queue: MPQueue, message: PipelineMessage) -> bool:
        """Common message sending logic"""
        try:
            if queue is None:
                return False
            return message.send(queue)
        except Exception as e:
            self.logger.error(f"Error sending message: {e}")
            return False
