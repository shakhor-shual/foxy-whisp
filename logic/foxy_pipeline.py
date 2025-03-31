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
import soundfile as sf


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

    def terminate(self):
        """Terminate the process with logging"""
        try:
            if self._process and self._process.is_alive():
                self._process.terminate()
                self.send_log(
                    "Process terminated",
                    level="warning",
                    details={
                        'pid': self._process.pid,
                        'name': self.__class__.__name__
                    }
                )
        except Exception as e:
            import traceback
            self.send_log(
                f"Error terminating process: {str(e)}", 
                level="error",
                details={
                    'traceback': traceback.format_exc(),
                    'pid': self._process.pid if self._process else None,
                    'name': self.__class__.__name__
                }
            )

    def stop(self):
        """Enhanced stop method with graceful shutdown"""
        try:
            if self._process and self._process.is_alive():
                # First try graceful shutdown
                self.stop_event.set()
                self._process.join(timeout=2.0)
                
                # If still alive, force terminate
                if self._process.is_alive():
                    self.terminate()
                    self._process.join(timeout=1.0)
                    
                # Clear process reference
                if hasattr(self._process, 'close'):
                    self._process.close()
                self._process = None
                
        except Exception as e:
            import traceback
            self.send_log(
                f"Error stopping process: {str(e)}", 
                level="error",
                details={
                    'traceback': traceback.format_exc(),
                    'pid': self._process.pid if self._process else None,
                    'name': self.__class__.__name__
                }
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
        self.fifo_buffer = AudioBuffer(sample_rate=16000, max_duration=1)  # Уменьшаем до 1 секунды
        self.vu_meter_threshold = -60  # Минимальный уровень для логирования в дБ
        self.vad_chunks_counter = 0  # Счетчик обработанных VAD чанков
        self.vu_meter_update_interval = 3  # Уменьшаем интервал обновления
        self.last_level_msg_time = 0  # Время последней отправки уровня
        self.min_level_update_interval = 0.05  # Уменьшаем минимальный интервал между обновлениями в секундах
        self.stage_id = f"src_{id(self)}"  # Добавляем уникальный идентификатор стейджа
        self.send_log("SRC stage initialized", details={"buffer_size": self.fifo_buffer.max_size})
        self._source_active = False  # Флаг активности источника
        self._recording_state = "stopped"  # состояние записи
        self._is_configured = False  # Новый флаг для отслеживания конфигурации
        self._current_device = None  # Добавляем явную инициализацию
        self._last_sent_state = None  # Добавляем отслеживание последнего отправленного состояния

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
        try:
            listen_type = self.args.get("listen", "tcp")
            self.send_log(f"Configuring {listen_type} source")
            
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
                from logic.foxy_utils import get_default_audio_device, initialize_audio_device
                
                # Get device info using existing utils
                if self.args.get("audio_device") is not None:
                    device_id, sample_rate = initialize_audio_device(self.args)
                    self._current_device = {
                        'device_id': device_id,
                        'samplerate': sample_rate,
                        'name': sd.query_devices(device_id)['name']
                    }
                else:
                    self._current_device = get_default_audio_device()
                
                self.source = AudioDeviceSource(
                    samplerate=self._current_device['default_samplerate'],
                    blocksize=self.args.get("chunk_size", 320),
                    device=self._current_device['device_id'],
                    stop_event=self.stop_event,
                    container=self
                )
                # Немедленно запускаем источник
                self.source.run()

            # После успешной конфигурации:
            self._is_configured = True
            self._source_active = True  # Источник сконфигурирован и активен
            self._recording_state = "recording"  # Меняем на recording вместо ready
            
            # Сразу активируем источник
            if hasattr(self.source, 'run'):
                self.source.run()
            
            # Отправляем обновленный статус с правильным состоянием
            self.send_source_status()
            self.send_status('configured')
            
            # Отправляем начальное состояние с активной записью
            self.send_initial_state()

        except Exception as e:
            self._is_configured = False
            self._source_active = False
            self._recording_state = "error"
            self.send_exception(e, "Failed to configure source")
            raise

    def send_initial_state(self):
        """Send complete initial state after configuration"""
        # Отправляем полное состояние источника с флагом активной записи
        current_state = {
            'active': True,
            'current_device': self._current_device,
            'recording_state': "recording",
            'is_configured': True,
            'available_devices': AudioDeviceSource.get_audio_devices() 
                               if hasattr(self.source, 'device') 
                               else None
        }
        self.send_data('source_status', current_state)
        self._last_sent_state = current_state.copy()

    def send_source_status(self):
        """Send current source status with state"""
        current_state = {
            'active': self._source_active,
            'current_device': self._current_device,
            'recording_state': self._recording_state,
            'is_configured': self._is_configured,
            'available_devices': AudioDeviceSource.get_audio_devices() 
                               if hasattr(self.source, 'device') 
                               else None
        }
        
        # Отправляем только если состояние изменилось
        if current_state != self._last_sent_state:
            self.send_data('source_status', current_state)
            self._last_sent_state = current_state.copy()

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
            if np.max(np.abs(audio_chunk)) > 1.0:
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
            
            # Обработка данных через VAD
            try:
                while True:
                    chunk = self.fifo_buffer.get_chunk(self.vad.frame_size)
                    if chunk is None or chunk.size == 0:
                        break
                        
                    if hasattr(self.vad, 'detect_voice'):
                        voice_detected = self.vad.detect_voice(chunk)
                        if voice_detected:
                            self.audio_write(chunk)
                    else:
                        self.send_log("VAD missing detect_voice method", level="error")
                        break

            except Exception as e:
                self.send_log(
                    f"VAD processing error: {str(e)}", 
                    level="error",
                    details={
                        'component': 'vad',
                        'frame_size': self.vad.frame_size,
                        'chunk_size': len(chunk) if chunk is not None else 0
                    }
                )

            self.vad_chunks_counter += 1
            return None

        except Exception as e:
            self.send_log(
                f"Audio processing error: {str(e)}", 
                level="error",
                details={
                    'component': 'processor',
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
            from logic.foxy_utils import get_default_audio_device
            # Use existing util function and format for UI
            default_dev = get_default_audio_device()
            devices_info = {
                'devices': [{
                    'index': default_dev['device_id'],
                    'name': default_dev['name'],
                    'channels': default_dev['max_input_channels'],
                    'default': True
                }]
            }
            self.send_data('audio_devices', devices_info)
        elif cmd == 'start_recording':
            if self._recording_state not in ["stopped", "ready"]:
                return
                
            self._recording_state = "starting"
            self.send_source_status()
            
            if not self._source_active:
                if hasattr(self.source, 'run'):
                    self.source.run()
                self._source_active = True
                
            self._recording_state = "recording"
            self.send_source_status()
            self.pause_event.clear()
            
        elif cmd == 'stop_recording':
            if self._recording_state != "recording":
                return
                
            self._recording_state = "stopping"
            self.send_source_status()
            
            if hasattr(self.source, 'stop'):
                self.source.stop()
                
            # Clear buffers
            self.fifo_buffer = AudioBuffer(sample_rate=16000, max_duration=1)
            self.vad_chunks_counter = 0
            
            # Reset indicators
            self.send_data('vad_status', {
                'active': False,
                'voice_detected': False
            })
            self.send_data('audio_level', {
                'level': 0,
                'timestamp': time.time(),
                'is_silence': True
            })
            
            self._source_active = False  # Источник остановлен
            self._recording_state = "stopped"
            self.send_source_status()
            self.pause_event.set()

    def _run(self):
        self.send_status('starting')
        try:
            self.configure()
            # После configure обязательно отправляем начальное состояние
            self.send_initial_state()
            
            while not self.stop_event.is_set():
                self.process_control_commands()
                if self.pause_event.is_set():
                    # Отправляем статус при паузе
                    self.send_source_status()
                    self.send_log("Processing paused", level="debug")
                    time.sleep(0.1)
                    continue
                
                # Исправляем проблемное место
                audio_data = self.source.receive_audio()
                if audio_data is not None and audio_data.size > 0:  # Правильная проверка numpy array
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
        self.debug_dir = os.path.join(os.getcwd(), "debug_audio")
        self.debug_file_path = os.path.join(self.debug_dir, "pipeline_debug.wav")
        self.debug_recording = False
        self.debug_file = None
        self.debug_sample_rate = 16000
        self.stop_recording_requested = False  # Флаг запроса на остановку записи
        self.recording_stopped = True  # Флаг состояния записи
        self._finalizing = False
        self.send_log("ASR stage initialized")

    def init_debug_recording(self):
        """Initialize debug recording"""
        try:
            # Создаем директорию если её нет
            os.makedirs(self.debug_dir, exist_ok=True)
            
            # Проверяем и удаляем существующий файл
            if os.path.exists(self.debug_file_path):
                try:
                    os.remove(self.debug_file_path)
                    self.send_log("Removed existing debug file", 
                                level="debug",
                                details={'file': self.debug_file_path})
                except Exception as e:
                    self.send_exception(
                        e,
                        "Failed to remove existing debug file",
                        component="debug_recorder"
                    )
            
            # Открываем файл для записи
            self.debug_file = sf.SoundFile(
                self.debug_file_path,
                mode='w',
                samplerate=self.debug_sample_rate,
                channels=1,
                format='WAV'
            )
            self.debug_recording = True
            self.recording_stopped = False
            self.send_log("Debug recording initialized", 
                         details={'file': self.debug_file_path})
            
        except Exception as e:
            self.send_exception(
                e,
                "Failed to initialize debug recording",
                component="debug_recorder"
            )
            self.debug_recording = False

    def stop_debug_recording(self):
        """Enhanced stop debug recording with safe finalization"""
        self._finalizing = True
        try:
            if self.debug_file:
                # Принудительно сбрасываем буферы на диск
                self.debug_file.flush()
                # Закрываем файл
                self.debug_file.close()
                self.debug_file = None
                
                # Проверяем что файл действительно записан
                if os.path.exists(self.debug_file_path):
                    file_size = os.path.getsize(self.debug_file_path)
                    self.send_log(
                        "Debug recording finalized", 
                        level="info",
                        details={'file': self.debug_file_path, 'size': file_size}
                    )
            self.debug_recording = False
            self.recording_stopped = True
            
        except Exception as e:
            self.send_exception(
                e,
                "Error finalizing debug recording",
                component="debug_recorder"
            )
        finally:
            self._finalizing = False

    def write_debug_audio(self, audio_chunk):
        """Write audio chunk to debug file"""
        try:
            if self.debug_recording and self.debug_file:
                self.debug_file.write(audio_chunk)
                
        except Exception as e:
            self.send_exception(
                e,
                "Error writing debug audio",
                component="debug_recorder",
                chunk_size=len(audio_chunk) if audio_chunk is not None else 0
            )
            self.stop_debug_recording()

    def process(self, audio_chunk):
        """Enhanced process with safe finalization"""
        try:
            if self.stop_recording_requested:
                # Даем время на обработку оставшихся данных
                if self.pipe_input:
                    while not self.pipe_input.empty():
                        try:
                            data = self.pipe_input.get(timeout=0.1)
                            if data is not None and len(data) > 0:
                                self.write_debug_audio(data)
                        except:
                            break
                    
                self.stop_debug_recording()
                self.stop_recording_requested = False
                return None

            # Записываем входящий аудиопоток если включена отладка
            if self.debug_recording and not self.recording_stopped:
                if audio_chunk is not None and len(audio_chunk) > 0:
                    self.write_debug_audio(audio_chunk)

            # Обработка чанков и генерация тестовых фраз
            self.chunk_counter += 1
            if self.chunk_counter % 10 == 0:  # Каждые 10 чанков
                phrase = self.test_phrases[self.chunk_counter % len(self.test_phrases)]
                self.text_buffer += phrase + " "
                self.send_data('asr_result', {
                    'text': phrase,
                    'buffer': self.text_buffer,
                    'is_final': False
                })
            
            return None  # ASR не передает аудио дальше
            
        except Exception as e:
            self.send_exception(
                e,
                "Error in ASR processing",
                level="error",
                details={
                    'debug_recording': self.debug_recording,
                    'chunk_size': len(audio_chunk) if audio_chunk is not None else 0,
                    'stop_requested': self.stop_recording_requested,
                    'recording_stopped': self.recording_stopped
                }
            )
            return None

    def configure(self):
        """Enhanced configure method with debug setup"""
        try:
            # Инициализируем отладочную запись по умолчанию
            self.init_debug_recording()
            self.send_status('configured')
            
        except Exception as e:
            self.send_exception(
                e,
                "Failed to configure ASR stage",
                level="error"
            )

    def stop(self):
        """Enhanced stop method"""
        try:
            # Ensure all data is written before stopping
            if self.stop_recording_requested and not self.recording_stopped:
                self.send_log("Flushing remaining audio data", level="debug")
                while not self.pipe_input.empty():
                    if data := self.audio_read():
                        self.process(data)
            else:
                # Close recording if still active
                self.stop_debug_recording()
            
        except Exception as e:
            self.send_exception(
                e,
                "Error in ASR stop",
                level="error"
            )

    def _handle_command(self, command: Dict[str, Any]):
        """Handle stage commands"""
        cmd = command.get('command')
        if cmd == 'stop_recording':
            self.stop_recording_requested = True
            self.send_log("Stop recording requested", level="debug")
        elif cmd == 'start_recording':
            # Если файл существует, закрываем его
            self.stop_debug_recording()
            # Инициализируем новую запись
            self.init_debug_recording()
            self.stop_recording_requested = False
            self.recording_stopped = False
        else:
            super()._handle_command(command)

    def _run(self):
        """Updated run method with safe cleanup"""
        import atexit
        # Регистрируем функцию очистки
        def cleanup():
            if self.debug_file and not self._finalizing:
                self.stop_debug_recording()
        atexit.register(cleanup)
        try:
            self.send_status('starting')
            self.configure()
            while not self.stop_event.is_set():
                self.process_control_commands()
                if self.pause_event.is_set():
                    time.sleep(0.1)
                    continue
                    
                # Fix: Правильная проверка numpy array
                audio_data = self.audio_read()
                if audio_data is not None and audio_data.size > 0:
                    self.process(audio_data)
                else:
                    time.sleep(0.01)
        except Exception as e:
            self.send_exception(
                e,
                "ASR stage error",
                level="error",
                component="main"
            )
            self.send_status('error', error=str(e))
        finally:
            cleanup()
            atexit.unregister(cleanup)
            if hasattr(self, 'text_buffer') and self.text_buffer:
                self.send_data('asr_final_result', {
                    'text': self.text_buffer,
                    'is_final': True
                })
            self.send_status('stopped')