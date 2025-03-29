from enum import Enum, auto
from typing import Dict, Any, Optional
import re

class MessageSource(Enum):
    SERVER = "server"
    SRC = "src"
    ASR = "asr"
    VAD = "vad"
    GUI = "gui"
    TEST = "test"
    SYSTEM = "system"

    @classmethod
    def normalize(cls, source: str) -> str:
        """Normalize source name to standard form"""
        source_map = {
            "srcstage": cls.SRC.value,
            "asrstage": cls.ASR.value,
            "system": cls.SERVER.value,
            "pipeline": cls.SERVER.value,
        }
        return source_map.get(source.lower(), source.lower())

class MessageValidator:
    @staticmethod
    def validate_log_content(content: Dict[str, Any]) -> bool:
        """Validate log message content"""
        required_fields = {'message', 'level'}
        valid_levels = {'debug', 'info', 'warning', 'error'}
        
        if not all(field in content for field in required_fields):
            return False
            
        if content['level'] not in valid_levels:
            return False
            
        if not isinstance(content['message'], str):
            return False
            
        return True

    @staticmethod
    def validate_status_content(content: Dict[str, Any]) -> bool:
        """Validate status message content"""
        if 'status' not in content:
            return False
            
        if not isinstance(content['status'], str):
            return False
            
        if 'details' in content and not isinstance(content['details'], dict):
            return False
            
        return True

    @staticmethod
    def validate_data_content(content: Dict[str, Any]) -> bool:
        """Validate data message content"""
        required_fields = {'data_type', 'payload'}
        
        if not all(field in content for field in required_fields):
            return False
            
        if not isinstance(content['data_type'], str):
            return False
            
        return True

    @staticmethod
    def validate_command_content(content: Dict[str, Any]) -> bool:
        """Validate command message content"""
        if 'command' not in content:
            return False
            
        if not isinstance(content['command'], str):
            return False
            
        if 'params' in content and not isinstance(content['params'], dict):
            return False
            
        return True

    @staticmethod
    def validate_source(source: str) -> bool:
        """Validate message source"""
        try:
            normalized = MessageSource.normalize(source)
            return normalized in [member.value for member in MessageSource]
        except:
            return False

    @staticmethod
    def validate_audio_level_data(level_data: Dict[str, Any]) -> bool:
        """Validate audio level data content"""
        required_fields = {'level', 'timestamp'}
        
        if not all(field in level_data for field in required_fields):
            return False
            
        if not isinstance(level_data['level'], (int, float)):
            return False
            
        if not isinstance(level_data['timestamp'], (int, float)):
            return False
            
        if level_data['level'] < 0 or level_data['level'] > 100:
            return False
            
        return True
