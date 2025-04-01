from enum import Enum, auto

class MessageType(Enum):
    LOG = auto()
    STATUS = auto() 
    DATA = auto()
    COMMAND = auto()
    CONTROL = auto()

class MessageSource(Enum):
    SERVER = "server"
    SRC = "src"
    ASR = "asr"
    VAD = "vad"
    GUI = "gui"  
    TEST = "test"
    SYSTEM = "system"
    TCP = "tcp"

    @staticmethod
    def normalize(source: str) -> str:
        """Standard source name normalization"""
        parts = source.lower().split('.')
        base_source = parts[0]
        
        source_map = {
            "srcstage": "src",
            "asrstage": "asr", 
            "system": "server",
            "pipeline": "server",
            "tcp": "src",
            "audio_device": "src",
            "vad": "src"
        }
        
        normalized_base = source_map.get(base_source, base_source)
        return f"{normalized_base}.{'.'.join(parts[1:])}" if len(parts) > 1 else normalized_base
