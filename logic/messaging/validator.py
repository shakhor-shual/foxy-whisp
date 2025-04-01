from enum import Enum
from typing import Dict, Any, Union
import re

class MessageSource(Enum):
    SERVER = "server"
    SRC = "src"
    ASR = "asr"
    VAD = "vad"
    GUI = "gui"
    TEST = "test"
    SYSTEM = "system"
    TCP = "tcp"

    @classmethod
    def normalize(cls, source: str) -> str:
        """Standard source name normalization"""
        # ...existing normalize method...

class MessageValidator:
    """Централизованная валидация сообщений"""
    
    @classmethod
    def validate_message(cls, msg) -> bool:
        """Validate entire message structure"""
        if not cls.validate_source(msg.source):
            return False
            
        validators = {
            'LOG': cls.validate_log_content,
            'STATUS': cls.validate_status_content,
            'DATA': cls.validate_data_content,
            'COMMAND': cls.validate_command_content
        }
        
        return validators.get(msg.type.name, lambda x: True)(msg.content)

    # ...existing validation methods...
