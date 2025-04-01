from typing import Optional, Dict, Any, Tuple
import logging
from .types import MessageType, MessageSource

class BaseMessageHandler:
    """Base handler with integrated validation"""
    
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

    @classmethod
    def validate_source(cls, source: str) -> bool:
        """Validate message source"""
        try:
            base_source = source.split('.')[0].lower()
            normalized = MessageSource.normalize(base_source)
            return normalized in [member.value for member in MessageSource]
        except:
            return False

    @classmethod
    def validate_log_content(cls, content: Dict[str, Any]) -> bool:
        """Validate log message content"""
        if not isinstance(content, dict):
            return False
            
        required_fields = {'message', 'level'}
        if not all(field in content for field in required_fields):
            return False
            
        valid_levels = {'debug', 'info', 'warning', 'error', 'critical'}
        if content['level'].lower() not in valid_levels:
            return False
            
        return True

    @classmethod
    def validate_message(cls, msg) -> bool:
        """Validate entire message structure"""
        if not isinstance(msg.content, dict):
            return False
            
        if not cls.validate_source(msg.source):
            return False
            
        validators = {
            MessageType.LOG: cls.validate_log_content,
            MessageType.STATUS: cls.validate_status_content,
            MessageType.DATA: cls.validate_data_content,
            MessageType.COMMAND: cls.validate_command_content
        }
        
        validator = validators.get(msg.type, lambda x: True)
        return validator(msg.content)

    def _format_message(self, text: str) -> Optional[Tuple[str, str, str, str]]:
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
