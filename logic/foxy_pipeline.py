from multiprocessing import Process, Queue, Event
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
import time
from logic.audio_sources import AudioDeviceSource, TCPSource
from logic.vad_filters import VADBase, create_vad, WebRTCVAD, AudioBuffer
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from logic.foxy_message import PipelineMessage, MessageType
from scipy.signal import resample_poly
import numpy as np
from logic.audio_utils import calculate_vu_meter
import traceback
import os
import threading


class PipelineElement(ABC):
    def __init__(self, 
                 stop_event: Event,
                 args: Dict[str, Any], 
                 audio_in: Optional[Queue] = None,
                 audio_out: Optional[Queue] = None,
                 in_queue: Optional[Queue] = None,
                 out_queue: Optional[Queue] = None,
                 pipe_chunk_size: int = 320):
        self.stop_event = stop_event
        self.args = args
        self.pipe_input = audio_in
        self.audio_out = audio_out
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.chunk_size = pipe_chunk_size
        self.pause_event = Event()
        self._process = None

    def send_log(self, message: str, level: str = "info", **kwargs):
        """Enhanced logging with detailed context"""
        source = self.__class__.__name__.lower()
        component = kwargs.pop('component', 'main')
        
        # Добавляем расширенный контекст для любого типа сообщения
        runtime_context = {
            'process_info': {
                'pid': os.getpid(),
                'thread_id': threading.get_ident(),
                'component': self.__class__.__name__,
                'timestamp': time.time()
            },
            'execution_context': {
                'component': component,
                'stage': source,
                **kwargs
            }
        }

        # Если это ошибка, добавляем стектрейс
        if level in ('error', 'critical'):
            import traceback
            runtime_context['error_info'] = {
                'traceback': traceback.format_stack(),
                'locals': {k: str(v) for k, v in locals().items() 
                          if not k.startswith('__')}
            }
        
        content = {
            'message': message,
            'level': level,
            'context': runtime_context
        }
        
        msg = PipelineMessage(
            source=f"{source}.{component}",
            type=MessageType.LOG,
            content=content
        )
        msg.send(self.out_queue)

    def send_status(self, status: str, **details):
        msg = PipelineMessage(
            source=self.__class__.__name__.lower(),
            type=MessageType.STATUS,
            content={'status': status, 'details': details}
        )
        msg.send(self.out_queue)

    def send_data(self, data_type: str, data: Any, **metadata):
        msg = PipelineMessage(
            source=self.__class__.__name__.lower(),
            type=MessageType.DATA,
            content={'data_type': data_type, 'payload': data, **metadata}
        )
        msg.send(self.out_queue)

    def process_control_commands(self):
        while msg := PipelineMessage.receive(self.in_queue):
            if msg.type == MessageType.COMMAND:
                self._handle_command(msg.content)

    def _handle_command(self, command: Dict[str, Any]):
        cmd = command.get('command')
        if cmd == 'stop':
            self.stop_event.set()
        elif cmd == 'pause':
            self.pause_event.set()
        elif cmd == 'resume':
            self.pause_event.clear()

    def audio_read(self) -> Optional[bytes]:
        if self.pipe_input:
            try:
                return self.pipe_input.get(timeout=0.1)
            except:
                return None
        return None

    def audio_write(self, data: bytes):
        if self.audio_out and data is not None and data.size > 0:
            self.audio_out.put(data)

    @abstractmethod
    def configure(self):
        pass

    @abstractmethod
    def process(self, audio_chunk: bytes) -> Optional[bytes]:
        pass

    def send_exception(self, e: Exception, message: str = None, level: str = "error", **kwargs):
        """Send exception with full context"""
        import traceback
        
        error_context = {
            'exception': {
                'type': type(e).__name__,
                'message': str(e),
                'traceback': traceback.format_exc().split('\n')
            },
            **kwargs
        }
        
        self.send_log(
            message=message or str(e),
            level=level,
            error_details=error_context
        )

    def _run(self):
        """Updated run method with enhanced error handling"""
        self.send_status('starting')
        try:
            self.configure()
            while not self.stop_event.is_set():
                self.process_control_commands()
                
                if self.pause_event.is_set():
                    time.sleep(0.1)
                    continue

                try:
                    if data := self.audio_read():
                        if processed := self.process(data):
                            self.audio_write(processed)
                    else:
                        time.sleep(0.01)
                except Exception as e:
                    # Отправляем детальную информацию об ошибке
                    self.send_exception(
                        e,
                        f"Error processing data in {self.__class__.__name__}",
                        component_info={
                            'state': 'processing',
                            'has_data': data is not None
                        }
                    )
                    # Продолжаем работу после ошибки
                    continue
                    
        except Exception as e:
            # Отправляем информацию о критической ошибке
            self.send_exception(
                e,
                f"Critical error in {self.__class__.__name__}",
                level="critical",
                component_info={
                    'state': 'failed'
                }
            )
            self.send_status('error', error=str(e))
        finally:
            self.send_status('stopped')

    def start(self):
        if self._process is None or not self._process.is_alive():
            self._process = Process(target=self._run)
            self._process.start()
            self.send_log(f"Process started (PID: {self._process.pid})")

    def stop(self):
        if self._process and self._process.is_alive():
            self.stop_event.set()
            self._process.join(timeout=2.0)
            if self._process.is_alive():
                self._process.terminate()

    def join(self, timeout=None):
        if self._process:
            self._process.join(timeout=timeout)

    def is_alive(self):
        return self._process is not None and self._process.is_alive()


class SRCstage(PipelineElement):
    def __init__(self, stop_event, audio_out, out_queue, in_queue, args, pipe_chunk_size=320):
        super().__init__(stop_event, args, None, audio_out, in_queue, out_queue, pipe_chunk_size)
        self.source = None
        self.vad = WebRTCVAD(aggressiveness=args.get("vad_aggressiveness", 3))
        self.vad.connect_control(self.out_queue)  # Connect VAD to pipeline messaging
        self.fifo_buffer = AudioBuffer(sample_rate=16000, max_duration=5)
        self.vu_meter_threshold = -60  # Минимальный уровень для логирования в дБ
        self.vad_chunks_counter = 0  # Счетчик обработанных VAD чанков
        self.vu_meter_update_interval = 3  # Уменьшаем интервал обновления
        self.last_level_msg_time = 0  # Время последней отправки уровня
        self.min_level_update_interval = 0.05  # Уменьшаем минимальный интервал между обновлениями в секундах
        self.stage_id = f"src_{id(self)}"  # Добавляем уникальный идентификатор стейджа
        self.send_log("SRC stage initialized", details={"buffer_size": self.fifo_buffer.max_size})

    def send_log(self, message: str, level: str = "info", **kwargs):
        """Enhanced logging with source identification"""
        content = {
            'message': message,
            'level': level,
            'details': {
                'stage_id': self.stage_id,
                'stage_type': 'src',
                'component_info': kwargs.get('component_info', {}),
                **kwargs
            }
        }
        
        msg = PipelineMessage(
            source=f"src.{kwargs.get('component', 'main')}",  # Добавляем компонент в источник
            type=MessageType.LOG,
            content=content
        )
        msg.send(self.out_queue)

    def _send_message(self, msg_type: MessageType, content: dict):
        """Enhanced message sending with component tracking"""
        if not self.out_queue:
            return False
            
        try:
            component = content.get('source_details', {}).get('component', 'main')
            source = f"src.{component}"
            
            msg = PipelineMessage(
                source=source,
                type=msg_type,
                content=content
            )
            return msg.send(self.out_queue)
        except Exception as e:
            self.send_log(f"Error sending message: {e}", 
                         level="error",
                         component="message_handler")
            return False

    def configure(self):
        """Initialize audio source based on configuration"""
        listen_type = self.args.get("listen", "tcp")
        self.send_log(f"Configuring {listen_type} source")
        
        try:
            if listen_type == "tcp":
                self.source = TCPSource(
                    args=self.args, 
                    stop_event=self.stop_event,
                    container=self
                )
                # Запускаем TCP source в отдельном потоке
                import threading
                self.tcp_thread = threading.Thread(target=self.source.run)
                self.tcp_thread.daemon = True
                self.tcp_thread.start()
            elif listen_type == "audio_device":
                self.source = AudioDeviceSource(
                    samplerate=self.args.get("sample_rate", 16000),
                    blocksize=self.args.get("chunk_size", 320),
                    device=self.args.get("audio_device"),
                    stop_event=self.stop_event,
                    container=self  # Pass self as container
                )
                # Запускаем AudioDeviceSource
                self.source.run()  # Добавляем явный запуск
            else:
                raise ValueError(f"Unknown source type: {listen_type}")
            
            self.send_status('configured')
        except Exception as e:
            self.send_exception(
                e,
                "Failed to configure source",
                level="error",
                component="configuration",
                component_info={
                    'stage_id': self.stage_id,
                    'source_type': self.args.get("listen", "tcp")
                }
            )
            raise

    def send_level_message(self, level: int, is_silence: bool = False):
        """Отправка сообщения об уровне сигнала"""
        current_time = time.time()
        if current_time - self.last_level_msg_time >= self.min_level_update_interval:
            # Нормализуем уровень в диапазон 0-100
            normalized_level = min(max(int(level), 0), 100)
            
            self.send_data('audio_level', {
                'level': normalized_level,
                'timestamp': current_time,
                'is_silence': is_silence
            })
            self.last_level_msg_time = current_time

    def process(self, audio_chunk):
        """Обработка аудиоданных."""
        try:
            # Проверка входных данных
            if audio_chunk is None or not isinstance(audio_chunk, np.ndarray):
                self.send_log("Invalid audio data received", level="warning")
                return None

            # Нормализация до float32 [-1, 1]
            if audio_chunk.dtype != np.float32:
                audio_chunk = audio_chunk.astype(np.float32)
            if np.max(np.abs(audio_chunk)) > 1.0:  # Исправляем проблемную строку
                audio_chunk = audio_chunk / 32768.0

            # Измерение уровня входного сигнала
            level = int(calculate_vu_meter(audio_chunk))
            
            # Отправка уровня сигнала с проверкой частоты обновления
            if self.vad_chunks_counter % self.vu_meter_update_interval == 0:
                self.send_level_message(level)

            # Ресэмплинг до 16 кГц если нужно
            audio_chunk = self._resample_to_16k(audio_chunk)

            # Добавление в буфер с проверкой размера
            if len(audio_chunk) > 0:
                self.fifo_buffer.add_data(audio_chunk)
                self.send_log("Buffer updated", level="debug", 
                            details={"buffer_size": len(self.fifo_buffer.buffer)})
            
            # Обработка данных из буфера
            while chunk := self.fifo_buffer.get_chunk(self.vad.frame_size):
                try:
                    if self.vad.detect_voice(chunk):
                        self.audio_write(chunk)
                except Exception as e:
                    self.send_exception(
                        e,
                        "VAD processing error",
                        component="vad",
                        component_info={
                            'frame_size': self.vad.frame_size,
                            'chunk_size': len(chunk)
                        }
                    )
                    break

            self.vad_chunks_counter += 1
            return None

        except Exception as e:
            self.send_exception(
                e,
                "Audio processing error",
                component="processor",
                component_info={
                    'chunk_size': len(audio_chunk) if audio_chunk is not None else 0,
                    'buffer_size': len(self.fifo_buffer.buffer) if hasattr(self, 'fifo_buffer') else 0
                }
            )
            return None

    def _resample_to_16k(self, audio_chunk):
        """Ресэмплинг аудиоданных до 16 кГц и одного канала."""
        input_sample_rate = self.args.get("sample_rate", 16000)
        if input_sample_rate != 16000:
            # Выполняем ресэмплинг
            audio_chunk = resample_poly(audio_chunk, 16000, input_sample_rate)
            self.send_log("Performed resampling", level="debug", 
                         details={
                             "from_rate": input_sample_rate,
                             "to_rate": 16000,
                             "input_size": len(audio_chunk)
                         })
        return audio_chunk.astype(np.float32)

    def _handle_command(self, command: Dict[str, Any]):
        """Handle incoming commands"""        
        cmd = command.get('command')
        if cmd == 'get_audio_devices':
            # Get devices list from AudioDeviceSource
            devices_info = AudioDeviceSource.get_audio_devices()
            self.send_data('audio_devices', devices_info)
        elif cmd == 'stop':
            self.stop_event.set()
        elif cmd == 'pause':
            self.pause_event.set()
        elif cmd == 'resume':
            self.pause_event.clear()

    def _run(self):
        self.send_status('starting')
        try:
            self.configure()
            self.send_log("SRC stage configured and running", level="info")
            
            # Добавляем проверку на запуск источника
            if hasattr(self.source, 'stream') and self.source.stream is None:
                self.source.run()
            
            while not self.stop_event.is_set():
                self.process_control_commands()
                if self.pause_event.is_set():
                    self.send_log("Processing paused", level="debug")
                    time.sleep(0.1)
                    continue
                if audio_data := self.source.receive_audio():
                    self.send_log("Received audio data", level="debug", 
                                  details={"data_size": len(audio_data)})
                    self.process(audio_data)
                else:
                    time.sleep(0.01)
        except Exception as e:
            self.send_exception(
                e,
                "SRC stage error",
                level="error",
                component="main",
                component_info={
                    'stage_id': self.stage_id,
                    'stage_type': 'src'
                }
            )
            self.send_status('error', error=str(e))
        finally:
            self.send_log("SRC stage stopping", level="info")
            self.send_status('stopped')


class ASRstage(PipelineElement):
    def __init__(self, stop_event, audio_in, out_queue, in_queue, args, pipe_chunk_size=320):
        super().__init__(stop_event, args, audio_in, None, in_queue, out_queue, pipe_chunk_size)
        self.chunk_counter = 0
        self.text_buffer = ""
        self.test_phrases = [
            "Тестовое распознавание текста",
            "Продолжаем тестирование системы",
            "Это имитация работы ASR"
        ]
        self.send_log("ASR stage initialized")

    def configure(self):
        self.send_status('configured')

    def process(self, audio_chunk):
        self.chunk_counter += 1
        # Уменьшаем интервал с 50 до 10 чанков
        if self.chunk_counter % 10 == 0:  # Каждые 10 чанков
            phrase = self.test_phrases[self.chunk_counter % len(self.test_phrases)]
            self.text_buffer += phrase + " "
            self.send_data('asr_result', {
                'text': phrase,
                'buffer': self.text_buffer,
                'is_final': False
            })
        
        return None  # ASR не передает аудио дальше

    def _run(self):
        self.send_status('starting')
        try:
            self.configure()
            while not self.stop_event.is_set():
                self.process_control_commands()
                if self.pause_event.is_set():
                    time.sleep(0.1)
                    continue
                if data := self.audio_read():
                    self.process(data)
                else:
                    time.sleep(0.01)
        except Exception as e:
            self.send_exception(
                e,
                "ASR stage error",
                level="error"
            )
            self.send_status('error', error=str(e))
        finally:
            if self.text_buffer:
                self.send_data('asr_final_result', {
                    'text': self.text_buffer,
                    'is_final': True
                })
            self.send_status('stopped')