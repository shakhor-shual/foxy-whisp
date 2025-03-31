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
        msg = PipelineMessage(
            source=self.__class__.__name__.lower(),
            type=MessageType.LOG,
            content={'message': message, 'level': level, **kwargs}
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
                    if processed := self.process(data):
                        self.audio_write(processed)
                else:
                    time.sleep(0.01)
                    
        except Exception as e:
            self.send_log(f"Error in {self.__class__.__name__}: {str(e)}", level="error")
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
        self.send_log("SRC stage initialized", details={"buffer_size": self.fifo_buffer.max_size})

    def send_log(self, message: str, level: str = "info", **kwargs):
        """Enhanced logging from components"""
        msg = PipelineMessage(
            source=self.__class__.__name__.lower(),
            type=MessageType.LOG,
            content={'message': message, 'level': level, **kwargs}
        )
        msg.send(self.out_queue)

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
            self.send_log(f"Failed to configure source: {e}", level="error")
            raise

    def send_level_message(self, level: int, is_silence: bool = False):
        """Отправка сообщения об уровне сигнала"""
        current_time = time.time()
        if current_time - self.last_level_msg_time >= self.min_level_update_interval:
            message = {
                'data_type': 'audio_level',
                'payload': {
                    'level': 0 if is_silence else level,
                    'timestamp': current_time,
                    'is_silence': is_silence
                }
            }
            if self.out_queue:
                self.send_data('audio_level', message['payload'])
            self.last_level_msg_time = current_time

    def process(self, audio_chunk):
        """Добавление данных в FIFO буфер и обработка через VAD."""
        try:
            # Сначала нормализуем входные данные до диапазона [-1, 1]
            if audio_chunk.dtype != np.float32:
                audio_chunk = audio_chunk.astype(np.float32)
                audio_chunk = audio_chunk / 32768.0  # нормализация из int16

            # Измеряем уровень входного сигнала
            level = calculate_vu_meter(audio_chunk)

            # Ресэмплинг аудиоданных до 16 кГц
            audio_chunk = self._resample_to_16k(audio_chunk)

            # Добавление данных в буфер
            self.fifo_buffer.add_data(audio_chunk)
            vad_chunk_size = self.vad.get_chunk_size()

            # Check VAD status messages and forward them
            vad_status = self.vad.get_status()
            if vad_status:
                self.send_status('vad_status', **vad_status)

            chunks_processed = 0
            voice_chunks_detected = 0
            have_data = False

            while True:
                chunk = self.fifo_buffer.get_chunk(vad_chunk_size)
                if chunk is None:
                    if not have_data:
                        # Если не смогли получить ни одного чанка, отправляем нулевой уровень
                        self.send_level_message(0.0, is_silence=True)
                    break

                have_data = True
                chunks_processed += 1
                self.vad_chunks_counter += 1
                
                # Измеряем уровень каждого чанка
                chunk_level = calculate_vu_meter(chunk)
                
                # Отправляем информацию об уровне чаще
                if self.vad_chunks_counter % self.vu_meter_update_interval == 0:
                    self.send_level_message(chunk_level)

                # Проверка на голос
                if self.vad.detect_voice(chunk):
                    voice_chunks_detected += 1                    
                    self.audio_write(chunk)
                    # Статистика только для голосовых чанков
                    self.send_data('audio_chunk', {
                        'size': len(chunk),
                        'voice_detected': True,
                        'stats': {
                            'total_chunks': chunks_processed,
                            'voice_chunks': voice_chunks_detected,
                            'level': float(chunk_level)
                        }
                    })

            # Forward level messages from source
            if hasattr(self.source, 'last_level'):
                self.send_data('audio_level', {'level': self.source.last_level})

            # Если данных больше нет в буфере, отправляем сообщение о тишине
            if not have_data:
                self.send_level_message(0.0, is_silence=True)

        except Exception as e:
            self.send_log(f"Process error: {e}", level="error")
            raise

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
            self.send_log(f"SRC error: {str(e)}", level="error", 
                         details={"exception_type": type(e).__name__})
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
            self.send_log(f"ASR error: {str(e)}", level="error")
            self.send_status('error', error=str(e))
        finally:
            if self.text_buffer:
                self.send_data('asr_final_result', {
                    'text': self.text_buffer,
                    'is_final': True
                })
            self.send_status('stopped')