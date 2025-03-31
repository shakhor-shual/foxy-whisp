from logic.foxy_config import *
from logic.audio_utils import calculate_vu_meter
from logic.foxy_message import PipelineMessage, MessageType
from typing import Any, Optional
import numpy as np
import sounddevice as sd
import socket
import time
import select
import errno
from collections import deque
from multiprocessing import Event
from scipy.signal import resample
from logic.foxy_utils import get_default_audio_device  # Используем существующую функцию

class AudioDeviceSource:
    def __init__(self, samplerate=None, blocksize=320, device=None, 
                 accumulation_buffer_size=0.5, stop_event=None, container=None):
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.device = device
        self.stream = None
        self.accumulation_buffer = deque(maxlen=int(16000 * accumulation_buffer_size))
        self.internal_queue = deque()
        self.stop_event = stop_event or Event()  # Default value if None
        self.container = container  # Reference to src-stage container
        self.last_level_time = time.time()  # Initialize with current time
        self.debug_enabled = True  # Добавляем флаг отладки
        self.source_id = f"audio_device_{id(self)}"  # Уникальный идентификатор источника

    @staticmethod
    def list_devices():
        """Get list of available audio input devices."""
        devices = []
        try:
            device_list = sd.query_devices()
            for i, device in enumerate(device_list):
                if device['max_input_channels'] > 0:
                    devices.append({
                        'index': i,
                        'name': device['name'],
                        'channels': device['max_input_channels'],
                        'default_samplerate': device['default_samplerate']
                    })
        except Exception as e:
            print(f"Error listing audio devices: {e}")
        return devices

    @staticmethod
    def get_audio_devices():
        """Get list of available audio input devices using sounddevice"""
        devices = []
        try:
            for i, device in enumerate(sd.query_devices()):
                if device['max_input_channels'] > 0:
                    devices.append({
                        'index': i,
                        'name': device['name'],
                        'channels': device['max_input_channels'],
                        'default': i == sd.default.device[0]
                    })
        except Exception as e:
            print(f"Error getting audio devices: {e}")
        return devices

    @staticmethod
    def get_current_device():
        """Get current active audio device info"""
        try:
            return get_default_audio_device()  # Используем существующую функцию
        except Exception as e:
            print(f"Error getting current device: {e}")
            return None

    def _send_message(self, msg_type: MessageType, content: dict):
        """Унифицированный метод отправки сообщений"""
        if not self.container or not hasattr(self.container, 'out_queue'):
            if self.debug_enabled:
                print(f"AudioDeviceSource[{self.source_id}]: Can't send message - no container queue")
            return False

        try:
            # Добавляем source_info в детали сообщения
            source_info = {
                'source_id': self.source_id,
                'source_type': 'audio_device',
                'device_info': {
                    'name': sd.query_devices(self.device)['name'] if self.device is not None else 'default',
                    'samplerate': self.samplerate,
                    'blocksize': self.blocksize
                }
            }
            
            if 'details' in content:
                content['details'].update(source_info)
            else:
                content['details'] = source_info

            msg = PipelineMessage(
                source='src.audio_device',  # Изменяем формат source
                type=msg_type,
                content=content
            )
            msg.send(self.container.out_queue)
            
            if self.debug_enabled:
                print(f"AudioDeviceSource[{self.source_id}]: Message sent: {msg_type}, Content: {content}")
            return True
        except Exception as e:
            if self.debug_enabled:
                print(f"AudioDeviceSource[{self.source_id}]: Failed to send message: {e}")
            return False

    def log(self, level: str, message: str, **details):
        """Улучшенное логирование"""
        content = {
            'message': message,
            'level': level.lower(),
        }
        if details:
            content['details'] = details

        if self._send_message(MessageType.LOG, content):
            if self.debug_enabled:
                print(f"AudioDeviceSource Log: [{level}] {message} {details if details else ''}")

    def send_data(self, data_type: str, data: Any, **metadata):
        """Send data through PipelineMessage"""
        content = {
            'data_type': data_type,
            'payload': data,
            **metadata
        }
        return self._send_message(MessageType.DATA, content)

    def send_status(self, status: str, **details):
        """Send status through PipelineMessage"""
        content = {
            'status': status,
            'details': details
        }
        return self._send_message(MessageType.STATUS, content)

    def send_exception(self, exception, message, **details):
        """Send exception details through PipelineMessage"""
        content = {
            'message': message,
            'exception': str(exception),
            **details
        }
        self._send_message(MessageType.EXCEPTION, content)

    def run(self):
        # Проверяем состояние stop_event перед запуском
        if self.stop_event and self.stop_event.is_set():
            self.log("INFO", "AudioDeviceSource: stop_event is set, not starting")
            return
            
        if self.stream:
            self.stop()

        device_info = sd.query_devices(self.device, 'input')
        default_samplerate = int(device_info['default_samplerate'])
        self.samplerate = self.samplerate or default_samplerate

        try:
            self.stream = sd.InputStream(
                samplerate=self.samplerate,
                blocksize=self.blocksize,
                device=self.device,
                channels=1,
                dtype=np.float32,
                callback=self.callback
            )
            self.stream.start()
            
            # Отправляем статусы о запуске
            self.send_status('starting')
            self.log("info", "Audio stream started",
                    samplerate=self.samplerate,
                    blocksize=self.blocksize,
                    device=self.device)
            self.send_status('configured')
            
        except Exception as e:
            self.send_exception(
                e,
                "Failed to start audio stream",
                level="error",
                component="audio_device",
                device_info={
                    'device': self.device,
                    'samplerate': self.samplerate,
                    'blocksize': self.blocksize
                }
            )
            raise

    def stop(self):
        """Stop audio capture and clear buffers"""
        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None
                
            # Clear buffers
            self.accumulation_buffer.clear()
            self.internal_queue.clear()
            
            # Send status updates
            self.send_status('stopped')
            self.send_data('audio_level', {
                'level': 0,
                'timestamp': time.time(),
                'is_silence': True
            })
            self.log("info", "Audio device stopped and buffers cleared")
        except Exception as e:
            self.send_exception(e, "Error stopping audio device")

    def callback(self, indata, frames, time_info, status):
        """Callback-функция для обработки аудиоданных."""
        if status:
            self.send_exception(
                status,
                "Audio callback status",
                level="warning",
                frames=frames,
                time_info=str(time_info)
            )
        
        try:
            # Convert to float32 and normalize for level calculation
            float_data = indata.astype(np.float32)
            if float_data.ndim > 1:
                float_data = np.mean(float_data, axis=1)

            # Calculate and send level every 100ms
            current_time = time.time()
            if current_time - self.last_level_time >= 0.1:
                level = calculate_vu_meter(float_data)
                self.send_data('audio_level', {
                    'level': level,
                    'timestamp': current_time,
                    'is_silence': level == 0
                })
                self.last_level_time = current_time

            # Process data for the pipeline
            processed_data = self._process_audio(indata)
            self.accumulation_buffer.extend(processed_data)
            
            if len(self.accumulation_buffer) >= self.accumulation_buffer.maxlen:
                self.internal_queue.append(np.array(self.accumulation_buffer))
                self.accumulation_buffer.clear()
                self.log("debug", "Buffer accumulated",
                        size=len(processed_data),
                        queue_size=len(self.internal_queue))

        except Exception as e:
            self.send_exception(
                e,
                "Audio processing error",
                level="error",
                details={
                    'frames': frames,
                    'buffer_size': len(self.accumulation_buffer),
                    'queue_size': len(self.internal_queue)
                }
            )

    def receive_audio(self):
        """Извлечение аудиоданных из внутренней очереди."""
        try:
            if len(self.internal_queue) > 0:
                data = self.internal_queue.popleft()
                # Проверяем что данные валидны
                if not isinstance(data, np.ndarray):
                    raise ValueError("Invalid audio data type")
                    
                # Приводим к одномерному массиву если нужно
                if data.ndim > 1:
                    data = np.mean(data, axis=1)
                    
                # Проверяем размер и возвращаем данные
                if data.size > 0:
                    return data.astype(np.float32)  # Явно приводим к float32
                return None
            return None
            
        except Exception as e:
            self.send_exception(
                e,
                "Error receiving audio data",
                level="error",
                details={
                    'queue_size': len(self.internal_queue) if self.internal_queue else 0,
                    'error_type': type(e).__name__,
                    'data_shape': data.shape if isinstance(data, np.ndarray) else None
                }
            )
            return None

    def _process_audio(self, indata):
        """Обработка аудиоданных: преобразование стерео в моно, ресемплинг и т.д."""
        if indata.ndim > 1 and indata.shape[1] > 1:
            indata = np.mean(indata, axis=1)  # Преобразуем стерео в моно

        indata = indata.astype(np.float32)

        if self.samplerate != 16000:
            target_len = int(len(indata) * (16000 / self.samplerate))
            indata = resample(indata, target_len)

        indata = np.clip(indata * 32767, -32768, 32767).astype(np.int16)
        return indata

##################################################################
class TCPSource:
    def __init__(self, args, stop_event=None, container=None):
        self.args = args
        self.host = args.get("host", "0.0.0.0")
        self.port = args.get("port", 43007)
        self.socket = None
        self.connection = None
        self.stop_event = stop_event or Event()  # Default value if None
        self.chunk_size = args.get("chunk_size", 320)
        self.sample_width = 2
        self.internal_queue = deque()
        self.container = container  # Reference to src-stage container
        self.name = "tcp"  # Добавляем имя для логов
        self.debug_enabled = True  # Включаем отладку
        self.source_id = f"tcp_{id(self)}"  # Уникальный идентификатор источника
        self.port_retry_count = 5  # Number of retries for port binding
        self.port_retry_delay = 1.0  # Delay between retries in seconds
        self.bind_timeout = 30.0  # Maximum time to wait for port
        self._closing = False  # Flag to track graceful shutdown

    def _send_message(self, msg_type: MessageType, content: dict):
        """Унифицированный метод отправки сообщений"""
        if not self.container or not hasattr(self.container, 'out_queue'):
            if self.debug_enabled:
                print(f"TCPSource[{self.source_id}]: Can't send message - no container queue")
            return False

        try:
            # Добавляем source_info в детали сообщения
            source_info = {
                'source_id': self.source_id,
                'source_type': 'tcp',
                'connection_info': {
                    'host': self.host,
                    'port': self.port
                }
            }
            
            if 'details' in content:
                content['details'].update(source_info)
            else:
                content['details'] = source_info

            msg = PipelineMessage(
                source='src.tcp',  # Изменяем формат source
                type=msg_type,
                content=content
            )
            msg.send(self.container.out_queue)
            
            if self.debug_enabled:
                print(f"TCPSource[{self.source_id}]: Message sent: {msg_type}, Content: {content}")
            return True
        except Exception as e:
            if self.debug_enabled:
                print(f"TCPSource[{self.source_id}]: Failed to send message: {e}")
            return False

    def log(self, level: str, message: str, **details):
        """Улучшенное логирование"""
        content = {
            'message': message,
            'level': level.lower(),
        }
        if details:
            content['details'] = details

        if self._send_message(MessageType.LOG, content):
            if self.debug_enabled:
                print(f"TCPSource Log: [{level}] {message} {details if details else ''}")

    def send_exception(self, exception, message, **details):
        """Send exception details through PipelineMessage"""
        content = {
            'message': message,
            'exception': str(exception),
            **details
        }
        self._send_message(MessageType.EXCEPTION, content)

    def _create_socket(self):
        """Create and configure TCP socket with proper options"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # Allow port reuse
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # Non-blocking mode
            self.socket.setblocking(False)
            return True
        except Exception as e:
            self.send_exception(e, "Failed to create socket")
            return False

    def _bind_socket(self):
        """Attempt to bind socket with retries"""
        start_time = time.time()
        retry_count = 0
        
        while retry_count < self.port_retry_count:
            try:
                self.socket.bind((self.host, self.port))
                self.socket.listen(1)
                self.log("info", f"TCP server listening on {self.host}:{self.port}")
                return True
            except socket.error as e:
                if e.errno == errno.EADDRINUSE:
                    retry_count += 1
                    self.log("warning", f"Port {self.port} is busy, retry {retry_count}/{self.port_retry_count}")
                    
                    # Check if we exceeded timeout
                    if time.time() - start_time > self.bind_timeout:
                        self.send_exception(e, f"Failed to bind port {self.port} after timeout")
                        return False
                        
                    time.sleep(self.port_retry_delay)
                    continue
                else:
                    self.send_exception(e, "Socket bind error")
                    return False
        
        self.log("error", f"Failed to bind port {self.port} after {retry_count} retries")
        return False

    def stop(self):
        """Enhanced stop with graceful shutdown"""
        self._closing = True
        try:
            if self.connection:
                try:
                    self.connection.shutdown(socket.SHUT_RDWR)
                except:
                    pass
                self.connection.close()
                self.connection = None
                
            if self.socket:
                try:
                    self.socket.shutdown(socket.SHUT_RDWR)
                except:
                    pass
                self.socket.close()
                self.socket = None
                
            self.log("info", "TCP server stopped and resources released")
            
        except Exception as e:
            self.send_exception(e, "Error during TCP shutdown")
        finally:
            self._closing = False

    def run(self):
        """Updated run method with proper error handling"""
        try:
            if not self._create_socket() or not self._bind_socket():
                return
                
            while not self.stop_event.is_set():
                try:
                    ready = select.select([self.socket], [], [], 1.0)
                    if ready[0]:
                        self.connection, addr = self.socket.accept()
                        self.log("info", f"Client connected from {addr}")
                        
                        while not self.stop_event.is_set():
                            if data := self._non_blocking_receive_audio():
                                self.process_audio_data(data)
                            else:
                                break
                                
                except socket.error as e:
                    if self._closing:
                        break
                    self.send_exception(e, "Socket error during operation")
                    break
                    
        except Exception as e:
            self.send_exception(e, "TCP server error")
        finally:
            self.stop()

    def _non_blocking_receive_audio(self):
        """ Receive audio data in a non-blocking manner. """
        cycles = 0

        while cycles < self.max_cycles:
            # Check if the socket has data ready to read
            ready = select.select([self.conn], [], [], CONNECTION_SELECT_DELAY)  # Short timeout for polling
            if ready[0]:
                try:
                    raw_bytes = self.conn.recv(PACKET_SIZE)
                    if raw_bytes != b"":
                        return raw_bytes  # Return immediately if data is received
                except BlockingIOError:
                    pass
                except ConnectionResetError:
                    self.socket_closed = True
                    return None  # Connection was reset
            
            cycles += 1 # Increment the cycle count if no data is received
        return None  # Return None after timeout

    def receive_audio(self):
        """Извлечение аудиоданных из внутренней очереди."""
        try:
            if self.internal_queue:
                data = self.internal_queue.popleft()
                # Проверяем что данные валидны
                if not isinstance(data, np.ndarray):
                    raise ValueError("Invalid audio data type")
                if data.size == 0:
                    raise ValueError("Empty audio data")
                return data if data.size > 0 else None  # Явно проверяем размер массива
            return None
        except Exception as e:
            self.log("error", f"Error receiving audio data: {e}")
            return None