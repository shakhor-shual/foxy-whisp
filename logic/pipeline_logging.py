import logging
from typing import Optional, Any, Dict
from multiprocessing import Queue
import time

class QueueLoggerHandler(logging.Handler):
    def __init__(self, log_queue: Queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:
        try:
            log_entry = self.format(record)
            self.log_queue.put(log_entry)
        except Exception:
            self.handleError(record)

class LoggingMixin:
    def setup_logger(self, log_queue):
        self.log_queue = log_queue
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        
        # Очистка существующих обработчиков
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            
        # Добавление обработчика для очереди
        handler = QueueLoggerHandler(log_queue)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def log_message(self, level: str, message: str, extra: Optional[dict] = None):
        log_entry = {
            'source': self.__class__.__name__.lower(),
            'type': 'log',
            'level': level,
            'message': message,
            'timestamp': time.time()
        }
        if extra:
            log_entry['extra'] = extra
            
        if hasattr(self, 'logger'):
            getattr(self.logger, level)(log_entry)