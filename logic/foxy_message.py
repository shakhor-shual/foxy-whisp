from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Optional
import time
import logging
import os
from multiprocessing import Queue

class MessageType(Enum):
    LOG = auto()
    STATUS = auto()
    DATA = auto()
    COMMAND = auto()
    CONTROL = auto()

@dataclass
class PipelineMessage:
    source: str
    type: MessageType
    content: Any
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create_log(cls, source: str, message: str, level: str = "info", **kwargs):
        """Создание сообщения лога с унифицированным форматированием"""
        # Добавляем поддержку debug уровня
        if "PYTHONDEBUGLEVEL" in os.environ:
            if level == "debug" and os.environ["PYTHONDEBUGLEVEL"] != "1":
                return None

        # Важно: всегда используем нижний регистр для source и level
        source = source.lower().strip()
        level = level.lower().strip()
        
        # Убираем лишние уровни из source, если они есть
        if '.' in source:
            source = source.split('.')[0]
            
        return cls(
            source=source,
            type=MessageType.LOG,
            content={
                'message': message,
                'level': level,
                'raw_source': source,  # сохраняем оригинальный источник
                'formatted': f"[{source}.{level}] {message}"
            },
            metadata=kwargs
        )

    @classmethod
    def create_status(cls, source: str, status: str, **details):
        """Создание сообщения статуса"""
        return cls(
            source=source.lower(),
            type=MessageType.STATUS,
            content={
                'status': status,
                'details': details
            }
        )

    @classmethod
    def create_data(cls, source: str, data_type: str, data: Any, **metadata):
        """Создание сообщения с данными"""
        return cls(
            source=source.lower(),
            type=MessageType.DATA,
            content={
                'data_type': data_type,
                'payload': data
            },
            metadata=metadata
        )

    @classmethod
    def create_command(cls, source: str, command: str, **params):
        """Создание командного сообщения"""
        return cls(
            source=source.lower(),
            type=MessageType.COMMAND,
            content={
                'command': command,
                'params': params
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь для передачи через Queue"""
        return {
            'source': self.source,
            'type': self.type.name.lower(),
            'content': self.content,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Создание из словаря (из Queue)"""
        try:
            return cls(
                source=data['source'],
                type=MessageType[data['type'].upper()],
                content=data['content'],
                timestamp=data.get('timestamp', time.time()),
                metadata=data.get('metadata', {})
            )
        except (KeyError, AttributeError) as e:
            raise ValueError(f"Invalid message format: {e}")

    def send(self, queue: Queue):
        """Отправка сообщения в очередь"""
        if not queue:
            raise ValueError("Queue is not initialized")
        queue.put(self.to_dict())

    @staticmethod
    def receive(queue: Queue, timeout: float = 0.1) -> Optional['PipelineMessage']:
        """Получение сообщения из очереди"""
        try:
            if not queue.empty():
                return PipelineMessage.from_dict(queue.get(timeout=timeout))
        except Exception as e:
            logging.warning(f"Failed to receive message: {e}")
        return None

    def is_log(self) -> bool:
        return self.type == MessageType.LOG

    def is_status(self) -> bool:
        return self.type == MessageType.STATUS

    def is_data(self) -> bool:
        return self.type == MessageType.DATA

    def is_command(self) -> bool:
        return self.type == MessageType.COMMAND

    def is_control(self) -> bool:
        return self.type == MessageType.CONTROL

    def get_log_level(self) -> str:
        return self.content['level'] if self.is_log() else ''

    def get_status(self) -> str:
        return self.content['status'] if self.is_status() else ''

    def get_command(self) -> str:
        return self.content['command'] if self.is_command() else ''

    def get_data_type(self) -> str:
        return self.content['data_type'] if self.is_data() else ''