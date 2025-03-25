from  logic.foxy_config import *
import logging
import numpy as np
import soundfile
import librosa
import socket
import time
import select
from logic.local_audio_input import LocalAudioInput
from logic.foxy_utils import logger, get_port_status

import socket
import logging
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)

import sounddevice as sd  # Библиотека для работы с аудиоустройствами

import numpy as np
import sounddevice as sd
from collections import deque
import logging
from scipy.signal import resample
#
from multiprocessing import Event

logger = logging.getLogger(__name__)

###################################################################
class AudioDeviceSource:
    def __init__(self, samplerate=None, blocksize=32768, device=None, 
                 accumulation_buffer_size=1, stop_event=None):
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.device = device
        self.stream = None
        self.accumulation_buffer = deque(maxlen=int(16000 * accumulation_buffer_size))
        self.internal_queue = deque()
        self.stop_event = stop_event or Event()  # Дефолтное значение если None

    def run(self):
        if self.stop_event.is_set():
            return
            
        if self.stream:
            self.stop()

        device_info = sd.query_devices(self.device, 'input')
        default_samplerate = int(device_info['default_samplerate'])
        self.samplerate = self.samplerate or default_samplerate

        self.stream = sd.InputStream(
            samplerate=self.samplerate,
            blocksize=self.blocksize,
            device=self.device,
            channels=1,
            dtype=np.float32,
            callback=self.callback
        )
        self.stream.start()

    def stop(self):
        """Остановка захвата аудио."""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def callback(self, indata, frames, time, status):
        """Callback-функция для обработки аудиоданных."""
        if status:
            logger.warning(f"Audio input status: {status}")

        try:
            # Обработка данных
            processed_data = self._process_audio(indata)

            # Добавление данных в буфер для накопления
            self.accumulation_buffer.extend(processed_data)

            # Передача данных из буфера во внутреннюю очередь
            if len(self.accumulation_buffer) >= self.accumulation_buffer.maxlen:
                self.internal_queue.append(np.array(self.accumulation_buffer))
                self.accumulation_buffer.clear()

        except Exception as e:
            logger.error(f"Error processing audio data: {e}")

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
    def __init__(self, args, stop_event=None):
        self.args = args
        self.host = args.get("host", "0.0.0.0")
        self.port = args.get("port", 43007)
        self.socket = None
        self.connection = None
        self.stop_event = stop_event or Event()  # Дефолтное значение если None
        self.chunk_size = args.get("chunk_size", 320)
        self.sample_width = 2
        self.internal_queue = deque()

    def run(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((self.host, self.port))
        self.socket.listen(1)
        self.socket.settimeout(1)
        logger.info(f"Listening on {self.host}:{self.port}")

        while not self.stop_event.is_set():
            try:
                self.connection, addr = self.socket.accept()
                logger.info(f"Connected to client on {addr}")

                while not self.stop_event.is_set():
                    try:
                        data = self.connection.recv(self.chunk_size * self.sample_width)
                        if not data:
                            break

                        self.internal_queue.append(np.frombuffer(data, dtype=np.int16))
                    except socket.timeout:
                        continue
                    except BrokenPipeError:
                        logger.error("Client disconnected unexpectedly.")
                        break
                    except Exception as e:
                        logger.error(f"Error receiving audio data: {e}")
                        break

                self.connection.close()
                logger.info("Connection to client closed")

            except socket.timeout:
                continue
            except Exception as e:
                logger.error(f"Error in server loop: {e}")
                break

        logger.info("TCP server stopped")


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
        logger.info("TCP server resources released")

    def receive_audio(self):
        """Извлечение аудиоданных из внутренней очереди."""
        if self.internal_queue:
            return self.internal_queue.popleft()
        return None