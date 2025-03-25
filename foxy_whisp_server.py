# foxy_whisp_server.pu
import signal
import argparse
from multiprocessing import Process
from multiprocessing import Queue as MPQueue 
from multiprocessing import Event as MPEvent
from logic.foxy_config import *
from logic.foxy_utils import (
    set_logging,
    logger,
    add_shared_args
)
from logic.mqtt_handler import MQTTHandler
from logic.foxy_pipeline import SRCstage, ASRstage, ASRstage
from typing import Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass
import time  # Добавляем импорт time

import signal
import argparse
from multiprocessing import Process, Queue, Event
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union, Callable
import time
import logging
from enum import Enum, auto

logger = logging.getLogger(__name__)

import signal
import argparse
from multiprocessing import Process, Queue, Event
from dataclasses import dataclass
from typing import Optional, Dict, Any, Union, Callable
import time
import logging
from enum import Enum, auto
import sys
from logic.foxy_message import PipelineMessage, MessageType

logger = logging.getLogger(__name__)

# class MessageType(Enum):
#     LOG = auto()
#     STATUS = auto()
#     DATA = auto()
#     COMMAND = auto()
#     CONTROL = auto()

# @dataclass
# class PipelineMessage:
#     source: str  # 'gui', 'src', 'asr'
#     type: MessageType
#     content: Any
#     timestamp: float = field(default_factory=time.time)

import signal
import argparse
import logging
import sys
from multiprocessing import Process, Queue, Event
from typing import Optional, Dict, Any
from logic.foxy_message import PipelineMessage, MessageType
from logic.foxy_pipeline import SRCstage, ASRstage

logger = logging.getLogger(__name__)

class PipelineQueues:
    def __init__(self):
        self.src_2_asr = Queue(maxsize=100)  # Аудио SRC → ASR
        self.from_gui = Queue()  # GUI → Сервер
        self.to_gui = Queue()    # Сервер → GUI
        self.from_src = Queue()  # SRC → Сервер
        self.to_src = Queue()    # Сервер → SRC
        self.from_asr = Queue()  # ASR → Сервер
        self.to_asr = Queue()    # Сервер → ASR

class FoxyWhispServer:
    def __init__(self, from_gui: Optional[Queue] = None, 
                 to_gui: Optional[Queue] = None,
                 args: Optional[Dict[str, Any]] = None):
        self._shutdown_requested = False
        self.stop_event = Event()
        self.args = args or {}
        self.pipe_chunk = self.args.get('chunk_size', 512)
        
        # Инициализация очередей
        self.queues = PipelineQueues()
        if from_gui:
            self.queues.from_gui = from_gui
        if to_gui:
            self.queues.to_gui = to_gui

        # Инициализация процессов
        self.processes = {'src': None, 'asr': None}

        # Обработчики сигналов
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _init_processes(self):
        """Инициализация процессов конвейера"""
        self.processes['src'] = SRCstage(
            stop_event=self.stop_event,
            audio_out=self.queues.src_2_asr,
            out_queue=self.queues.from_src,
            in_queue=self.queues.to_src,
            args=self.args,
            pipe_chunk_size=self.pipe_chunk
        )
        
        self.processes['asr'] = ASRstage(
            stop_event=self.stop_event,
            audio_in=self.queues.src_2_asr,
            out_queue=self.queues.from_asr,
            in_queue=self.queues.to_asr,
            args=self.args,
            pipe_chunk_size=self.pipe_chunk
        )

    def process_messages(self):
        """Обработка входящих сообщений"""
        while not self.stop_event.is_set():
            # Проверяем все очереди сообщений
            for queue in [self.queues.from_src, self.queues.from_asr, self.queues.from_gui]:
                if msg := PipelineMessage.receive(queue):
                    self._handle_message(msg)
            
            time.sleep(0.01)

    def _handle_message(self, msg: PipelineMessage):
        """Обработка сообщения в зависимости от типа"""
        try:
            if msg.is_log():
                self._handle_log(msg)
            elif msg.is_status():
                self._handle_status(msg)
            elif msg.is_data():
                self._handle_data(msg)
            elif msg.is_command():
                self._handle_command(msg)
            elif msg.is_control():
                self._handle_control(msg)
        except Exception as e:
            logger.error(f"Error handling message: {e}\nMessage: {msg}")

    def _handle_log(self, msg: PipelineMessage):
        """Обработка лог-сообщения"""
        level = msg.content.get('level', 'info').upper()
        message = f"[{msg.source}] {msg.content.get('message', '')}"
        logger.log(getattr(logging, level, logging.INFO), message)

    def _handle_status(self, msg: PipelineMessage):
        """Пересылка статуса в GUI"""
        PipelineMessage.create_status(
            source=msg.source,
            status=msg.content.get('status', ''),
            **msg.content.get('details', {})
        ).send(self.queues.to_gui)

    def _handle_data(self, msg: PipelineMessage):
        """Обработка данных (например, результатов ASR)"""
        PipelineMessage.create_data(
            source=msg.source,
            data_type=msg.content.get('data_type', 'unknown'),
            data=msg.content.get('payload'),
            **msg.content.get('metadata', {})
        ).send(self.queues.to_gui)

    def _handle_command(self, msg: PipelineMessage):
        """Обработка команд от GUI"""
        if msg.source == 'gui':
            command = msg.content.get('command')
            if command == 'start':
                self.start_pipeline()
            elif command == 'stop':
                self.stop_pipeline()
            elif command == 'restart':
                self.restart_pipeline()

    def _handle_control(self, msg: PipelineMessage):
        """Обработка контрольных сообщений"""
        if msg.content.get('control') == 'restart':
            self.restart_pipeline()

    def start_pipeline(self):
        """Запуск конвейера обработки"""
        if any(p.is_alive() for p in self.processes.values() if p):
            logger.warning("Pipeline is already running")
            return

        try:
            self._init_processes()
            for name, proc in self.processes.items():
                if proc:
                    proc.start()
                    logger.info(f"Started {name} process")
            
            # Отправляем статус о запуске
            PipelineMessage.create_status(
                source='system',
                status='pipeline_started'
            ).send(self.queues.to_gui)

        except Exception as e:
            logger.error(f"Failed to start pipeline: {e}")

    def stop_pipeline(self):
        """Остановка конвейера"""
        if self.stop_event.is_set():
            return

        logger.info("Stopping pipeline...")
        self.stop_event.set()

        # Последовательная остановка процессов
        for name in ['asr', 'src']:  # Обратный порядок остановки
            if proc := self.processes.get(name):
                try:
                    # Отправляем команду на остановку
                    PipelineMessage.create_command(
                        source='server',
                        command='stop'
                    ).send(getattr(self.queues, f'to_{name}'))
                    
                    proc.join(timeout=2.0)
                    if proc.is_alive():
                        proc.terminate()
                except Exception as e:
                    logger.error(f"Error stopping {name}: {e}")

        logger.info("Pipeline stopped")

    def restart_pipeline(self):
        """Перезапуск конвейера"""
        self.stop_pipeline()
        self.stop_event.clear()
        self.start_pipeline()

    def run(self):
        """Основной цикл работы сервера"""
        logger.info("Starting FoxyWhisp server")
        
        try:
            self.start_pipeline()
            
            while not self._shutdown_requested:
                self.process_messages()
                time.sleep(0.1)
                
        except Exception as e:
            logger.critical(f"Server error: {e}", exc_info=True)
        finally:
            if not self._shutdown_requested:
                self.stop_pipeline()
            self._cleanup()
            logger.info("Server stopped")

    def _handle_signal(self, signum, frame):
        """Обработчик сигналов для graceful shutdown"""
        if self._shutdown_requested:
            return
            
        self._shutdown_requested = True
        sig_name = {signal.SIGINT: "SIGINT", signal.SIGTERM: "SIGTERM"}.get(signum, str(signum))
        logger.info(f"Received {sig_name}, shutting down...")
        
        # Уведомляем GUI о shutdown
        PipelineMessage.create_status(
            source='system',
            status='shutdown',
            reason=f"Signal {sig_name} received"
        ).send(self.queues.to_gui)

        self.stop_pipeline()

    def _cleanup(self):
        """Очистка ресурсов"""
        for queue in vars(self.queues).values():
            try:
                if hasattr(queue, 'close'):
                    queue.close()
                if hasattr(queue, 'join_thread'):
                    queue.join_thread()
            except Exception as e:
                logger.error(f"Error cleaning up queue: {e}")

def main():
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='small', help='ASR model size')
    parser.add_argument('--lan', default='ru', help='Language code')
    parser.add_argument('--chunk-size', type=int, default=512, help='Audio chunk size')
    args = parser.parse_args()
    
    # Запуск сервера
    server = FoxyWhispServer(
        from_gui=Queue(),
        to_gui=Queue(),
        args=vars(args))
    server.run()

if __name__ == "__main__":
    main()