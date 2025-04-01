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
            # Создаем сообщение с правильным форматированием
            PipelineMessage.create_log(
                source='server',
                message=record.getMessage(),
                level=record.levelname.lower()
            ).send(self.queue)
        except ValueError as e:
            if "Queue is closed" in str(e):
                pass
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

        self.test_thread = None
        self.test_running = False

        self._shutdown_started = False
        self._shutdown_completed = threading.Event()
        self._cleanup_timeout = 5.0  # Timeout for cleanup in seconds

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
        # Fix typo: 'levellevel' -> 'levelname'
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        queue_handler.setFormatter(formatter)
        root_logger.addHandler(queue_handler)

    ####################
    def _send_gui_message(self, msg: PipelineMessage):
        """Send message to GUI with proper source handling"""
        if self.queues.to_gui is not None:
            # Convert system messages to server messages
            if msg.source == 'system':
                msg.source = 'server'
            try:
                msg.send(self.queues.to_gui)
            except Exception as e:
                logger.error(f"GUI disconnected: {e}")
                self.queues.to_gui = None
                self._handle_gui_disconnect()

    ####################
    def _handle_gui_disconnect(self):
        """Обработка отключения GUI"""
        logger.warning("GUI disconnected! Continuing in headless mode...")
        # Ensure a StreamHandler is added to the root logger
        root_logger = logging.getLogger()
        # Remove only QueueHandler instances to avoid clearing other handlers
        root_logger.handlers = [
            handler for handler in root_logger.handlers
            if not isinstance(handler, QueueHandler)
        ]
        
        # Explicitly add a StreamHandler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # Log a test message to verify the handler is working
        root_logger.info("StreamHandler added to logger")
        
        # Ensure the logger propagates messages
        root_logger.propagate = True

    ####################
    def _force_shutdown(self):
        """Force terminate all processes"""
        for name, proc in self.processes.items():
            if proc and proc.is_alive():
                try:
                    logger.warning(f"Force terminating {name} process")
                    proc.terminate()
                    proc.join(timeout=1.0)
                    
                    if proc.is_alive():
                        os.kill(proc.pid, signal.SIGKILL)
                        logger.critical(f"Had to SIGKILL {name} process")
                except Exception as e:
                    logger.error(f"Error terminating {name}: {e}")

    ####################
    def _init_processes(self):
        """Инициализация процессов конвейера"""
        self.processes['src'] = SRCstage(
            stop_event=self.stop_event,
            audio_out=self.queues.src_2_asr,  # Передача данных на следующий этап
            out_queue=self.queues.from_src,
            in_queue=self.queues.to_src,
            args=self.args,
            pipe_chunk_size=self.pipe_chunk
        )
        
        self.processes['asr'] = ASRstage(
            stop_event=self.stop_event,
            audio_in=self.queues.src_2_asr,  # Получение данных от предыдущего этапа
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
            import traceback
            error_context = {
                'message_type': msg.type,
                'source': msg.source,
                'content': msg.content,
                'traceback': traceback.format_exc()
            }
            logger.exception(f"Message handling error", extra=error_context)

    ####################
    def _handle_log(self, msg: PipelineMessage):
        """Обработка лог-сообщения"""
        if self.queues.to_gui is not None:
            # Просто пересылаем сообщение в GUI без изменений
            self._send_gui_message(msg)
        
        # Для локального логирования используем стандартный формат
        level = msg.content.get('level', 'info').upper()
        message = msg.content.get('message', '')
        logger.log(getattr(logging, level, logging.INFO), message)

    ####################
    def _handle_status(self, msg: PipelineMessage):
        """Handle status message"""
        if self.queues.to_gui is not None:
            # Convert system messages to server messages for status updates
            source = 'server' if msg.source == 'system' else msg.source
            self._send_gui_message(
                PipelineMessage.create_status(
                    source=source,
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
        """Enhanced command handling"""
        command = msg.content.get('command')
        
        if command == 'shutdown':
            # Start shutdown in a separate thread to not block message processing
            threading.Thread(
                target=self.initiate_shutdown, 
                daemon=True
            ).start()
            return
            
        if msg.source == 'gui' and self.queues.from_gui is not None:
            params = msg.content.get('params', {})
            
            print(f"[SERVER] Received command: {command}")
            
            if command == 'stop_stage':
                stage = params.get('stage')
                if stage and stage in self.processes:
                    self._stop_stage(stage)
            elif command == 'start_stage':
                stage = params.get('stage')
                if stage and stage in self.processes:
                    self._start_stage(stage)
            elif command == 'update_params':
                # Обновляем параметры
                if params:
                    self.args.update(params)
                    # Перезапускаем SRC если изменился тип источника
                    if 'listen' in params and self.processes['src']:
                        self._restart_stage('src')
            elif command == 'start':
                self.start_pipeline()
            elif command == 'stop':
                self.stop_pipeline()
            elif command == 'restart':
                self.restart_pipeline()
            elif command == 'start_recording':
                # Отправляем команду на запись в SRC
                if self.processes.get('src'):
                    PipelineMessage.create_command(
                        source='server',
                        command='start_recording'
                    ).send(self.queues.to_src)
                    self._send_gui_message(
                        PipelineMessage.create_status(
                            source='server',
                            status='recording_started'
                        )
                    )
            elif command == 'stop_recording':
                # Отправляем команду остановки записи в SRC
                if self.processes.get('src'):
                    PipelineMessage.create_command(
                        source='server',
                        command='stop_recording'
                    ).send(self.queues.to_src)
                    self._send_gui_message(
                        PipelineMessage.create_status(
                            source='server',
                            status='recording_stopped'
                        )
                    )
            elif command == 'shutdown':
                logger.info("Received shutdown command")
                self.stop_pipeline()
                self._shutdown_requested = True

    ####################
    def _stop_stage(self, stage_name: str):
        """Остановка отдельной стадии с улучшенной обработкой ошибок"""
        if proc := self.processes.get(stage_name):
            try:
                logger.info(f"Stopping {stage_name} stage...")
                
                # Отправляем команду остановки
                try:
                    PipelineMessage.create_command(
                        source='server',
                        command='stop'
                    ).send(getattr(self.queues, f'to_{stage_name}'))
                except Exception as e:
                    logger.error(f"Error sending stop command to {stage_name}: {e}", 
                               exc_info=True)
                
                # Проверяем что процесс действительно наш дочерний
                if proc.is_alive() and proc.parent_pid == os.getpid():
                    # Даем время на graceful shutdown
                    proc.join(timeout=2.0)
                    
                    if proc.is_alive():
                        logger.warning(f"Force terminating {stage_name} process")
                        proc.terminate()
                        proc.join(timeout=1.0)
                        
                        # Если все еще жив - SIGKILL
                        if proc.is_alive():
                            import signal
                            os.kill(proc.pid, signal.SIGKILL)
                            logger.warning(f"SIGKILL sent to {stage_name} process")
                else:
                    logger.warning(f"{stage_name} process is not our child or not alive")
                
                # Очищаем ссылку на процесс
                if hasattr(proc, 'close'):
                    proc.close()
                self.processes[stage_name] = None
                
                # Очищаем связанные очереди
                self._clear_stage_queues(stage_name)
                
                logger.info(f"{stage_name} stage stopped")
                
            except Exception as e:
                logger.error(f"Error during {stage_name} stage shutdown: {e}", 
                           exc_info=True)
                # Форсированная очистка в случае ошибки
                self.processes[stage_name] = None

    ####################
    def _start_stage(self, stage_name: str):
        """Запуск отдельной стадии"""
        if stage_name not in self.processes:
            return
            
        try:
            # Создаем новый процесс стадии
            if stage_name == 'src':
                self.processes['src'] = SRCstage(
                    stop_event=self.stop_event,
                    audio_out=self.queues.src_2_asr,
                    out_queue=self.queues.from_src,
                    in_queue=self.queues.to_src,
                    args=self.args,
                    pipe_chunk_size=self.pipe_chunk
                )
            elif stage_name == 'asr':
                self.processes['asr'] = ASRstage(
                    stop_event=self.stop_event,
                    audio_in=self.queues.src_2_asr,
                    out_queue=self.queues.from_asr,
                    in_queue=self.queues.to_asr,
                    args=self.args,
                    pipe_chunk_size=self.pipe_chunk
                )

            # Запускаем процесс
            if proc := self.processes[stage_name]:
                proc.start()
                time.sleep(0.5)  # Даем время на инициализацию
                
                if not proc.is_alive():
                    raise RuntimeError(f"Failed to start {stage_name} stage")
                    
                logger.info(f"{stage_name} stage started (PID: {proc.pid})")
                
        except Exception as e:
            logger.error(f"Error starting {stage_name} stage: {e}")
            self._stop_stage(stage_name)
            raise

    ####################
    def _restart_stage(self, stage_name: str):
        """Перезапуск отдельной стадии"""
        try:
            self._stop_stage(stage_name)
            time.sleep(0.5)  # Даем время на освобождение ресурсов
            self._start_stage(stage_name)
        except Exception as e:
            logger.error(f"Error restarting {stage_name} stage: {e}")
            # Отправляем статус ошибки в GUI
            self._send_gui_message(
                PipelineMessage.create_status(
                    source='system',
                    status='stage_error',
                    details={
                        'stage': stage_name,
                        'error': str(e)
                    }
                )
            )

    ####################
    def _clear_stage_queues(self, stage_name: str):
        """Очистка очередей связанных со стадией"""
        queues_to_clear = []
        if stage_name == 'src':
            queues_to_clear = ['src_2_asr', 'from_src', 'to_src']
        elif stage_name == 'asr':
            queues_to_clear = ['from_asr', 'to_asr']
            
        for queue_name in queues_to_clear:
            queue = getattr(self.queues, queue_name)
            try:
                while not queue.empty():
                    queue.get_nowait()
            except:
                pass

    ####################
    def start_pipeline(self):
        """Запуск конвейера обработки"""
        if any(p and p.is_alive() for p in self.processes.values()):
            logger.warning("Pipeline is already running")
            self._send_gui_message(
                PipelineMessage.create_status(
                    source='system',
                    status='pipeline_error',
                    details={'error': 'Pipeline is already running'}
                )
            )
            return

        # Ensure clean state
        self.stop_event.clear()
        
        # Clear old processes
        for name, proc in self.processes.items():
            if proc is not None:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=1.0)
                if hasattr(proc, 'close'):
                    proc.close()
                self.processes[name] = None

        try:
            self._init_processes()
            
            # Start processes sequentially with status checks
            for name, proc in self.processes.items():
                if proc:
                    proc.start()
                    time.sleep(0.5)  # Give time to initialize
                    if not proc.is_alive():
                        raise RuntimeError(f"Failed to start {name} process")
                    pid = proc.pid if hasattr(proc, 'pid') else 'unknown'
                    logger.info(f"Started {name} process (PID: {pid})")

            # Send explicit status update
            self._send_gui_message(
                PipelineMessage.create_status(
                    source='system',
                    status='pipeline_started',
                    details={'state': 'running'}
                )
            )
            logger.info("Pipeline started successfully")

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to start pipeline: {error_msg}")
            self.stop_pipeline()  # Ensure cleanup
            self._send_gui_message(
                PipelineMessage.create_status(
                    source='system',
                    status='pipeline_error',
                    details={'error': error_msg}
                )
            )

    ####################
    def stop_pipeline(self):
        """Остановка конвейера"""
        logger.info("Stopping pipeline...")
        
        # Set stop event first
        self.stop_event.set()
        
        # Send initial stopping status
        self._send_gui_message(
            PipelineMessage.create_status(
                source='system',
                status='pipeline_stopping'
            )
        )

        time.sleep(0.1)  # Даем время на обработку события

        # Последовательная остановка процессов
        for name in ['asr', 'src']:  # Обратный порядок остановки
            if proc := self.processes.get(name):
                try:
                    # Отправляем команду остановки
                    PipelineMessage.create_command(
                        source='server',
                        command='stop'
                    ).send(getattr(self.queues, f'to_{name}'))
                    
                    # Даем время на graceful shutdown
                    proc.join(timeout=2.0)
                    
                    if proc.is_alive():
                        logger.warning(f"Force terminating {name} process")
                        proc.terminate()
                        proc.join(timeout=1.0)
                    
                    if proc.is_alive():
                        logger.error(f"Failed to stop {name} process")
                        continue
                        
                    if hasattr(proc, 'close'):
                        proc.close()
                    self.processes[name] = None
                    
                except Exception as e:
                    logger.error(f"Error stopping {name}: {e}")

        # Очищаем все очереди
        for queue_name in ['src_2_asr', 'from_src', 'to_src', 'from_asr', 'to_asr']:
            queue = getattr(self.queues, queue_name)
            try:
                while not queue.empty():
                    queue.get_nowait()
            except:
                pass

        # Always send final stopped status
        self._send_gui_message(
            PipelineMessage.create_status(
                source='system',
                status='pipeline_stopped',
                details={'state': 'stopped'}
            )
        )
        
        logger.info("Pipeline stopped")

    ####################
    def restart_pipeline(self):
        """Перезапуск конвейера"""
        self.stop_pipeline()
        # Нет необходимости создавать новый Event, 
        # т.к. он уже сбрасывается в stop_pipeline
        self.start_pipeline()

    ####################
    def process_messages(self):
        """Process all incoming messages from queues"""
        try:
            # Check GUI queue
            if self.queues.from_gui is not None:
                while msg := PipelineMessage.receive(self.queues.from_gui, timeout=0.1):
                    print(f"[SERVER] Processing message from GUI: {msg.type}")  # Debug output
                    self._handle_message(msg)

            # Check other component queues
            for name, queue in [('src', self.queues.from_src), ('asr', self.queues.from_asr)]:
                while msg := PipelineMessage.receive(queue, timeout=0.1):
                    print(f"[SERVER] Processing message from {name}: {msg.type}")  # Debug output
                    self._handle_message(msg)
                    
            return True
        except Exception as e:
            import traceback
            error_context = {
                'traceback': traceback.format_exc(),
                'queues_state': {
                    name: not queue._closed if hasattr(queue, '_closed') else None
                    for name, queue in vars(self.queues).items()
                }
            }
            logger.exception(f"Message processing error", extra=error_context)
            return False

    ####################
    def run(self):
        """Enhanced run method with shutdown handling"""
        self._send_gui_message(
            PipelineMessage.create_status(
                source='server',
                status='server_initialized'
            )
        )
        logger.info("Starting FoxyWhisp server")
        
        try:
            if self.queues.to_gui is not None:
                logger.info("Starting test message sender...")
                self.test_running = True
                self.test_thread = threading.Thread(target=self.send_test_messages, daemon=True)
                self.test_thread.start()

            self.start_pipeline()

            while not self._shutdown_completed.is_set():
                if not self.process_messages():
                    logger.error("Message processing failed")
                    break
                time.sleep(0.1)
                
        except Exception as e:
            logger.critical(f"Server error: {e}", exc_info=True)
        finally:
            if not self._shutdown_completed.is_set():
                self.initiate_shutdown()

    ####################
    def _handle_signal(self, signum, frame):
        """Enhanced signal handler"""
        sig_name = {signal.SIGINT: "SIGINT", signal.SIGTERM: "SIGTERM"}.get(signum, str(signum))
        logger.info(f"Received {sig_name}, initiating shutdown...")
        
        try:
            # Start shutdown timer
            import threading
            timer = threading.Timer(5.0, self._force_shutdown)
            timer.start()
            
            # Try graceful shutdown
            self.stop_pipeline()
            
            # Cancel timer if successful
            timer.cancel()
            
        except Exception as e:
            logger.error(f"Error during signal handling: {e}")
            self._force_shutdown()
        
        finally:
            self._shutdown_requested = True
            self.stop_event.set()

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
    def send_test_messages(self):
        """Send periodic test messages to GUI"""
        counter = 0
        while self.test_running and not self._shutdown_requested and self.queues.to_gui is not None:
            try:
                logger.info(f"Sending test message #{counter}")  # Добавляем лог для отладки
                test_msg = PipelineMessage.create_log(  # Меняем на create_log
                    source='test',
                    message=f'Test message #{counter} from server',
                    level='info'
                )
                test_msg.send(self.queues.to_gui)
                print(f"[SERVER] Sent test message #{counter}")  # Добавляем прямой вывод
                counter += 1
                time.sleep(5.0)
            except Exception as e:
                logger.error(f"Error sending test message: {e}")
                break

    ####################
    def initiate_shutdown(self):
        """Start the shutdown sequence"""
        if self._shutdown_started:
            return
            
        self._shutdown_started = True
        logger.info("Server shutdown initiated")
        
        try:
            # 1. Stop pipeline first
            self.stop_pipeline()
            
            # 2. Wait for pipeline components to finish
            shutdown_start = time.time()
            while any(p and p.is_alive() for p in self.processes.values()):
                if time.time() - shutdown_start > self._cleanup_timeout:
                    logger.warning("Forcing process termination")
                    self._force_shutdown()
                    break
                time.sleep(0.1)
                
            # 3. Clean up queues
            self._cleanup_queues()
            
            # 4. Set shutdown completed event
            self._shutdown_completed.set()
            logger.info("Server shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            self._force_shutdown()

    ####################
    def _cleanup_queues(self):
        """Safe queue cleanup"""
        for name, queue in vars(self.queues).items():
            try:
                if queue is None or queue._closed:
                    continue
                    
                # Clear queue
                while not queue.empty():
                    try:
                        queue.get_nowait()
                    except:
                        break
                        
                queue.close()
                queue.join_thread()
                logger.debug(f"Queue {name} closed")
                
            except Exception as e:
                logger.warning(f"Error cleaning up queue {name}: {e}")

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
    server = FoxyWhispServer(args=vars(args))
    # Исправляем: вызываем правильный метод run() напрямую
    # Тестовые сообщения запускаются внутри run()
    server.run()

if __name__ == "__main__":
    main()