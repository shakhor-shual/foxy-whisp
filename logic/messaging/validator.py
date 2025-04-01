from typing import Dict, Any
from .types import MessageType, MessageSource

class MessageValidator:
    """Централизованная валидация сообщений"""
    
    @staticmethod
    def validate_source(source: str) -> bool:
        """Validate message source with component support"""
        try:
            base_source = source.split('.')[0].lower()
            normalized = MessageSource.normalize(base_source)
            return normalized in [member.value for member in MessageSource]
        except:
            return False

    @staticmethod
    def validate_log_content(content: Dict[str, Any]) -> bool:
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

    @staticmethod
    def validate_status_content(content: Dict[str, Any]) -> bool:
        """Validate status message content"""
        if not isinstance(content, dict):
            return False
            
        if 'status' not in content or not isinstance(content['status'], str):
            return False
            
        if 'details' in content and not isinstance(content['details'], dict):
            return False
            
        return True

    @staticmethod
    def validate_data_content(content: Dict[str, Any]) -> bool:
        """Validate data message content"""
        if not isinstance(content, dict):
            return False
            
        required_fields = {'data_type', 'payload'}
        if not all(field in content for field in required_fields):
            return False
            
        return True

    @staticmethod
    def validate_command_content(content: Dict[str, Any]) -> bool:
        """Validate command message content"""
        if not isinstance(content, dict):
            return False
            
        if 'command' not in content or not isinstance(content['command'], str):
            return False
            
        if 'params' in content and not isinstance(content['params'], dict):
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
