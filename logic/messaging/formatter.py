from typing import Dict, Any, Optional, Tuple
from .validator import MessageSource

class MessageFormatter:
    """Централизованное форматирование сообщений"""
    
    def __init__(self):
        self.source_mappings = MessageSource.get_mappings()

    def format_message(self, text: str) -> Optional[Tuple[str, str, str, str]]:
        """Format message with standard normalization"""
        # ...existing format_message method...

    # ...existing formatting methods...
