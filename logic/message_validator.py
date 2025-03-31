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
    TCP = "tcp"  # Добавляем TCP как валидный источник

    @classmethod
    def normalize(cls, source: str) -> str:
        """Normalize source name with component support"""
        # Handle compound sources (e.g. 'src.vad.info')
        parts = source.lower().split('.')
        base_source = parts[0]
        
        # Normalize base source
        source_map = {
            "srcstage": cls.SRC.value,
            "asrstage": cls.ASR.value,
            "system": cls.SERVER.value,
            "pipeline": cls.SERVER.value,
            "tcp": cls.SRC.value,
            "audio_device": cls.SRC.value,
            "vad": cls.SRC.value,
        }
        
        normalized_base = source_map.get(base_source, base_source)
        
        # Return either base source or compound source
        if len(parts) > 1:
            return f"{normalized_base}.{'.'.join(parts[1:])}"
        return normalized_base

class MessageValidator:
    @staticmethod
    def validate_log_content(content: Dict[str, Any]) -> bool:
        """Validate log message content"""
        # Базовые требования для всех лог-сообщений
        if not isinstance(content, dict):
            return False
            
        if 'message' not in content or not isinstance(content['message'], str):
            return False
            
        if 'level' not in content or content['level'] not in {'debug', 'info', 'warning', 'error', 'critical'}:
            return False
            
        # Валидация расширенного контекста
        if 'context' in content:
            ctx = content['context']
            if not isinstance(ctx, dict):
                return False
                
            # Проверяем process_info если есть
            if 'process_info' in ctx:
                p_info = ctx['process_info']
                if not isinstance(p_info, dict):
                    return False
                required_proc_fields = {'pid', 'thread_id', 'component', 'timestamp'}
                if not all(k in p_info for k in required_proc_fields):
                    return False
                    
            # Проверяем execution_context если есть
            if 'execution_context' in ctx:
                e_ctx = ctx['execution_context']
                if not isinstance(e_ctx, dict):
                    return False
                required_exec_fields = {'component', 'stage'}
                if not all(k in e_ctx for k in required_exec_fields):
                    return False
                    
            # Проверяем error_info для ошибок
            if 'error_info' in ctx:
                err_info = ctx['error_info']
                if not isinstance(err_info, dict):
                    return False
                if 'traceback' in err_info and not isinstance(err_info['traceback'], (list, str)):
                    return False
                
        # Проверяем error_details если есть (для обратной совместимости)
        if 'error_details' in content:
            err_details = content['error_details']
            if not isinstance(err_details, dict):
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
        """Validate message source with component support"""
        try:
            # Split compound source
            base_source = source.split('.')[0].lower()
            normalized = MessageSource.normalize(base_source)
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
