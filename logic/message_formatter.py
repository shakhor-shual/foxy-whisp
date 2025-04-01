from typing import Dict, Any, Optional, Tuple
import logging

class MessageFormatter:
    """Форматтер сообщений для GUI"""
    
    def __init__(self):
        self.source_mappings = {
            'srcstage': 'src',
            'asrstage': 'asr',
            'system': 'server',
            'audio_device': 'src',
            'tcp': 'src',
            'vad': 'src'
        }

    def format_message(self, text: str) -> Optional[Tuple[str, str, str, str]]:
        """
        Форматирует сообщение и возвращает кортеж (base_source, component, level, message)
        Returns None если сообщение не удалось разобрать
        """
        try:
            if text.startswith('[') and ']' in text:
                parts = text[1:].split(']', 1)
                if len(parts) == 2:
                    source_parts = parts[0].lower().split('.')
                    message = parts[1].strip()
                    
                    # Extract components
                    base_source = source_parts[0]
                    level = source_parts[-1] if len(source_parts) > 1 else 'info'
                    component = '.'.join(source_parts[1:-1]) if len(source_parts) > 2 else 'main'
                    
                    # Apply source mapping
                    base_source = self.source_mappings.get(base_source, base_source)
                    
                    return base_source, component, level, message
            return None
        except Exception as e:
            logging.error(f"Message parsing error: {e}, Text: {text}")
            return None

    def format_log_message(self, source: str, level: str, message: str, 
                          context: Optional[Dict[str, Any]] = None) -> str:
        """Форматирует лог сообщение с контекстом"""
        try:
            # Базовое сообщение
            formatted = f"[{source}.{level}] {message}"
            
            # Добавляем контекст выполнения
            if context:
                if exec_context := context.get('execution_context'):
                    context_str = ', '.join(f"{k}={v}" for k, v in exec_context.items() 
                                          if k not in ('component', 'stage'))
                    if context_str:
                        formatted += f" ({context_str})"
                
                # Для ошибок добавляем расширенную информацию
                if level in ('error', 'critical'):
                    if error_info := context.get('error_info'):
                        formatted += "\nTraceback (most recent call last):"
                        if traceback := error_info.get('traceback'):
                            formatted += '\n' + '\n'.join(f"  {line}" for line in traceback)
                        
                        if locals_info := error_info.get('locals'):
                            formatted += "\nLocal variables:"
                            for k, v in locals_info.items():
                                formatted += f"\n  {k} = {v}"
                                
            return formatted
            
        except Exception as e:
            logging.error(f"Error formatting log message: {e}")
            return f"[ERROR] Message formatting failed: {message}"

    def format_status_message(self, source: str, status: str, details: Dict[str, Any]) -> str:
        """Форматирует статусное сообщение"""
        try:
            formatted = f"[{source}.{status}] {status}"
            if isinstance(details, dict) and details:
                detail_str = ', '.join(f"{k}={v}" for k, v in details.items())
                if detail_str:
                    formatted += f" ({detail_str})"
            return formatted
        except Exception as e:
            logging.error(f"Error formatting status message: {e}")
            return f"[ERROR] Status formatting failed: {status}"
