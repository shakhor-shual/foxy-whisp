# foxy_whisp_server.py
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
from logic.foxy_pipeline import SRCstage, ASRstage
from typing import Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass
import time
import logging
import threading
from logic.foxy_message import PipelineMessage, MessageType
import os

################################################################################
class PipelineQueues:
    ####################
    def __init__(self):
        self.src_2_asr = MPQueue(maxsize=100)  # Аудио SRC → ASR
        self.from_gui = None  # GUI → Сервер (может быть None)
        self.to_gui = None    # Сервер → GUI (может быть None)
        self.from_src = MPQueue()  # SRC → Сервер
        self.to_src = MPQueue()    # Сервер → SRC
        self.from_asr = MPQueue()  # ASR → Сервер
        self.to_asr = MPQueue()    # Сервер → ASR

################################################################################
class QueueHandler(logging.Handler):
    ####################
    def __init__(self, queue):
        super().__init__()
        self.queue = queue

    ####################
    def emit(self, record):
        try:
            msg = self.format(record)
            PipelineMessage.create_log(
                source='server',
                message=msg,
                level=record.levelname.lower()
            ).send(self.queue)
        except ValueError as e:
            if "Queue is closed" in str(e):
                pass  # Игнорируем ошибку закрытой очереди
            else:
                raise

################################################################################
class FoxyWhispServer:
    ####################
    def __init__(self, from_gui: Optional[MPQueue] = None, 
                 to_gui: Optional[MPQueue] = None,
                 args: Optional[Dict[str, Any]] = None):
        self._shutdown_requested = False
        self.stop_event = MPEvent()
        self.args = args or {}
        self.pipe_chunk = self.args.get('chunk_size', 512)
        
        # Инициализация очередей
        self.queues = PipelineQueues()
        self.queues.from_gui = from_gui  # Может быть None
        self.queues.to_gui = to_gui      # Может быть None

        # Инициализация процессов
        self.processes = {'src': None, 'asr': None}

        # Настройка перенаправления логов только если есть очередь to_gui
        if self.queues.to_gui is not None:
            self._setup_log_redirection()
        
        # Обработчики сигналов
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    ####################
    def _setup_log_redirection(self):
        """Перенаправление логов в GUI (только если queue.to_gui существует)"""
        if self.queues.to_gui is None:
            return
            
        root_logger = logging.getLogger()
        # Удаляем ВСЕ существующие обработчики
        root_logger.handlers.clear()
        
        # Устанавливаем только QueueHandler
        queue_handler = QueueHandler(self.queues.to_gui)
        queue_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        queue_handler.setFormatter(formatter)
        root_logger.addHandler(queue_handler)

    ####################
    def _send_gui_message(self, msg: PipelineMessage):
        if self.queues.to_gui is not None:
            try:
                msg.send(self.queues.to_gui)
            except Exception as e:
                logger.error(f"GUI disconnected: {e}")
                self.queues.to_gui = None
                # Больше не вызываем _force_shutdown при потере связи с GUI
                self._handle_gui_disconnect()

    ####################
    def _handle_gui_disconnect(self):
        """Обработка отключения GUI"""
        logger.warning("GUI disconnected! Continuing in headless mode...")
        # Переключаемся в режим логирования в консоль
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    ####################
    def _force_shutdown(self):
        """Аварийное завершение всех процессов сервера"""
        logger.critical("Force shutdown initiated...")
        
        # Останавливаем все дочерние процессы (SRC, ASR)
        for name, proc in self.processes.items():
            if proc and proc.is_alive():
                proc.terminate()
                
        self._shutdown_requested = True
        self.stop_event.set()

    ####################
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

    ####################
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

    ####################
    def _handle_log(self, msg: PipelineMessage):
        """Обработка лог-сообщения"""
        level = msg.content.get('level', 'info').upper()
        message = f"[{msg.source}] {msg.content.get('message', '')}"
        logger.log(getattr(logging, level, logging.INFO), message)

    ####################
    def _handle_status(self, msg: PipelineMessage):
        """Пересылка статуса в GUI (если очередь существует)"""
        if self.queues.to_gui is not None:
            self._send_gui_message(
                PipelineMessage.create_status(
                    source=msg.source,
                    status=msg.content.get('status', ''),
                    **msg.content.get('details', {})
                )
            )

    ####################
    def _handle_data(self, msg: PipelineMessage):
        """Обработка данных (например, результатов ASR)"""
        if self.queues.to_gui is not None:
            self._send_gui_message(
                PipelineMessage.create_data(
                    source=msg.source,
                    data_type=msg.content.get('data_type', 'unknown'),
                    data=msg.content.get('payload'),
                    **msg.content.get('metadata', {})
                )
            )

    ####################
    def _handle_command(self, msg: PipelineMessage):
        """Обработка команд от GUI"""
        if msg.source == 'gui' and self.queues.from_gui is not None:
            command = msg.content.get('command')
            if command == 'start':
                self.start_pipeline()
            elif command == 'stop':
                self.stop_pipeline()
            elif command == 'restart':
                self.restart_pipeline()
            elif command == 'shutdown':  # Fixed indentation and changed to elif
                logger.info("Received shutdown command")
                self.stop_pipeline()
                self._shutdown_requested = True

    ####################
    def _handle_control(self, msg: PipelineMessage):
        """Обработка контрольных сообщений"""
        if msg.content.get('control') == 'restart':
            self.restart_pipeline()

    ####################
    def process_messages(self):
        """Обработка входящих сообщений с проверкой наличия очередей"""
        while not self.stop_event.is_set():
            # Проверяем очередь GUI только если она существует
            if self.queues.from_gui is not None and (msg := PipelineMessage.receive(self.queues.from_gui)):
                self._handle_message(msg)
                continue
                
            # Проверяем очереди от компонентов
            for queue in [self.queues.from_src, self.queues.from_asr]:
                if msg := PipelineMessage.receive(queue):
                    self._handle_message(msg)
                    break
                    
            time.sleep(0.01)

    ####################
    def start_pipeline(self):
        """Запуск конвейера обработки"""
        if any(p and p.is_alive() for p in self.processes.values()):
            logger.warning("Pipeline is already running")
            return

        try:
            self._init_processes()
            for name, proc in self.processes.items():
                if proc:
                    proc.start()
                    logger.info(f"Started {name} process")
            
            # Отправляем статус о запуске
            self._send_gui_message(
                PipelineMessage.create_status(
                    source='system',
                    status='pipeline_started'
                )
            )

        except Exception as e:
            logger.error(f"Failed to start pipeline: {e}")

    ####################
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

    ####################
    def restart_pipeline(self):
        """Перезапуск конвейера"""
        self.stop_pipeline()
        self.stop_event.clear()
        self.start_pipeline()

    ####################
    def start_test_messages(self, interval: float = 1.0):
        """Запускает тестовые сообщения только если есть очередь to_gui"""
        if self.queues.to_gui is None:
            logger.info("No GUI queue, test messages disabled")
            return

        def test_message_loop():
            counter = 0
            while not self.stop_event.is_set() and self.queues.to_gui is not None:
                try:
                    # Изменяем на создание лог-сообщения вместо data
                    self._send_gui_message(
                        PipelineMessage.create_log(
                            source='test',
                            message=f"Тестовое сообщение #{counter}",
                            level='info'
                        )
                    )
                    counter += 1
                    time.sleep(interval)
                    
                except Exception as e:
                    logger.error(f"Test message error: {e}")
                    self.queues.to_gui = None
                    self._setup_log_redirection()
                    break

        threading.Thread(target=test_message_loop, daemon=True).start()

    ####################
    def _start_test_logging(self):
        """Запускает периодическую отправку тестовых лог-сообщений"""
        def test_log_loop():
            counter = 0
            while not self.stop_event.is_set():
                logger.info(f"Test log message #{counter}")
                counter += 1
                time.sleep(5.0)
                
        threading.Thread(target=test_log_loop, daemon=True).start()

    ####################
    def run(self):
        """Основной цикл работы сервера"""
        self._send_gui_message(
            PipelineMessage.create_status(
                source='server',
                status='server_initialized'
            )
        )
        logger.info("Starting FoxyWhisp server")
        
        try:
            # Запускаем тестовые сообщения если есть GUI
            if self.queues.to_gui is not None:
                self._start_test_logging()
                
            self.start_pipeline()
            
            while not self._shutdown_requested:
                self.process_messages()
                time.sleep(0.1)
                
        except Exception as e:
            logger.critical(f"Server error: {e}", exc_info=True)
        finally:
            if not self._shutdown_requested:
                self.stop_pipeline()
            
            # Логируем перед очисткой
            logger.info("Server stopping...")
            self._cleanup()

    ####################
    def _handle_signal(self, signum, frame):
        """Обработчик сигналов для graceful shutdown"""
        if self._shutdown_requested:
            return
            
        self._shutdown_requested = True
        sig_name = {signal.SIGINT: "SIGINT", signal.SIGTERM: "SIGTERM"}.get(signum, str(signum))
        logger.info(f"Received {sig_name}, shutting down...")
        
        # Уведомляем GUI о shutdown
        self._send_gui_message(
            PipelineMessage.create_status(
                source='system',
                status='shutdown',
                reason=f"Signal {sig_name} received"
            )
        )

        self.stop_pipeline()

    ####################
    def _cleanup(self):
        """Очистка ресурсов с проверкой на None"""
        # Удаляем QueueHandler если он был установлен
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            if isinstance(handler, QueueHandler):
                root_logger.removeHandler(handler)
        
        # Закрываем очереди (кроме None)
        for name, queue in vars(self.queues).items():
            try:
                if queue is None:
                    continue
                    
                if hasattr(queue, 'close'):
                    queue.close()
                if hasattr(queue, 'join_thread'):
                    queue.join_thread()
            except Exception as e:
                logger.error(f"Error cleaning up queue {name}: {e}")

####################
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
        # from_gui=MPQueue(),
        # to_gui=MPQueue(),
        args=vars(args))
    server.start_test_messages() # Запуск тестовых сообщений
    server.run()

if __name__ == "__main__":
    main()