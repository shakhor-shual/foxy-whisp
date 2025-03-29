from logic.foxy_config import *
from logic.audio_utils import calculate_vu_meter
from logic.foxy_message import PipelineMessage, MessageType
from typing import Any, Optional  # Add this import
import logging
import numpy as np
import sounddevice as sd
import socket
import time
import select
from logic.foxy_utils import logger, get_port_status
from collections import deque
from multiprocessing import Event
from scipy.signal import resample

logger = logging.getLogger(__name__)

###################################################################
class AudioDeviceSource:
    def __init__(self, samplerate=None, blocksize=32768, device=None, 
                 accumulation_buffer_size=1, stop_event=None, container=None):
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.device = device
        self.stream = None
        self.accumulation_buffer = deque(maxlen=int(16000 * accumulation_buffer_size))
        self.internal_queue = deque()
        self.stop_event = stop_event or Event()  # Default value if None
        self.container = container  # Reference to src-stage container
        self.last_level_time = time.time()  # Initialize with current time

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
            logger.error(f"Error listing audio devices: {e}")
        return devices

    @classmethod
    def get_audio_devices(cls):
        """Method that can be called from SRCstage to get devices list"""
        devices = cls.list_devices()
        return {
            "status": "ok",
            "devices": devices
        }

    def log(self, level, message, **details):
        """Enhanced logging with details support"""
        if self.container:
            try:
                self.container.send_log(message, level=level.lower(), **details)
            except Exception as e:
                print(f"AudioSource logging error: {e}, Message: {message}")

    def send_status(self, status: str, **details):
        """Send status updates through container"""
        if self.container:
            try:
                self.container.send_status(status, **details)
            except Exception as e:
                print(f"AudioSource status error: {e}, Status: {status}")

    def send_data(self, data_type: str, data: Any, **metadata):
        """Send data through container"""
        if self.container:
            try:
                self.container.send_data(data_type, data, **metadata)
            except Exception as e:
                print(f"AudioSource data error: {e}, Type: {data_type}")

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
            self.log("info", "Audio stream started",
                    samplerate=self.samplerate,
                    blocksize=self.blocksize,
                    device=self.device)
        except Exception as e:
            self.log("error", "Failed to start audio stream",
                    error=str(e),
                    device=self.device)
            raise

    def stop(self):
        """Остановка захвата аудио."""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.log("INFO", "AudioDeviceSource stopped")

    def callback(self, indata, frames, time_info, status):
        """Callback-функция для обработки аудиоданных."""
        if status:
            self.log("warning", "Audio callback status", 
                    status=str(status), frames=frames)
        
        try:
            # Convert to float32 and normalize for level calculation
            float_data = indata.astype(np.float32)
            if float_data.ndim > 1:
                float_data = np.mean(float_data, axis=1)

            # Calculate and send level every 100ms
            current_time = time.time()
            if current_time - self.last_level_time >= 0.1:
                level = calculate_vu_meter(float_data)
                self.send_data('audio_level', {'level': level, 'timestamp': current_time})
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
            self.log("error", "Audio processing error",
                    error=str(e),
                    frames=frames,
                    buffer_size=len(self.accumulation_buffer))

    def receive_audio(self):
        """Извлечение аудиоданных из внутренней очереди."""
        if self.internal_queue:
            return self.internal_queue.popleft()
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

    def log(self, level, message, **details):
        """Enhanced logging with details support"""
        if self.container:
            if details:
                self.container.send_log(message, level=level, **details)
            else:
                self.container.send_log(message, level=level)

    def send_status(self, status: str, **details):
        """Send status updates through container"""
        if self.container:
            self.container.send_status(status, **details)

    def run(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((self.host, self.port))
        self.socket.listen(1)
        self.socket.settimeout(1)
        self.log("info", f"TCP server listening", host=self.host, port=self.port)

        while not self.stop_event.is_set():
            try:
                self.connection, addr = self.socket.accept()
                self.log("INFO", f"Connected to client on {addr}")

                while not self.stop_event.is_set():
                    try:
                        data = self.connection.recv(self.chunk_size * self.sample_width)
                        if not data:
                            break

                        self.internal_queue.append(np.frombuffer(data, dtype=np.int16))
                    except socket.timeout:
                        continue
                    except BrokenPipeError:
                        self.log("ERROR", "Client disconnected unexpectedly.")
                        break
                    except Exception as e:
                        self.log("ERROR", f"Error receiving audio data: {e}")
                        break

                self.connection.close()
                self.log("INFO", "Connection to client closed")

            except socket.timeout:
                continue
            except Exception as e:
                self.log("ERROR", f"Error in server loop: {e}")
                break

        self.log("INFO", "TCP server stopped")


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

    def stop(self):
        """Остановка TCP-сервера."""
        if self.connection:
            self.connection.close()
        if self.socket:
            self.socket.close()
        self.log("INFO", "TCP server resources released")

    def receive_audio(self):
        """Извлечение аудиоданных из внутренней очереди."""
        if self.internal_queue:
            return self.internal_queue.popleft()
        return None