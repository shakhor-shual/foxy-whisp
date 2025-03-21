from logic.foxy_config import *
from logic.foxy_utils import logger, send_one_line_tcp, receive_lines_tcp
import sounddevice as sd
import numpy as np

import sounddevice as sd
import numpy as np

from logic.foxy_config import *
from logic.foxy_utils import logger, send_one_line_tcp, receive_lines_tcp
import sounddevice as sd
import numpy as np
import librosa
import numpy as np
from scipy.signal import resample

class LocalAudioInput:
    def __init__(self, samplerate=None, blocksize=16000, device=None):
        """
        Инициализация захвата аудио с локального устройства.

        :param samplerate: Частота дискретизации (если None, используется частота устройства по умолчанию).
        :param blocksize: Размер блока аудиоданных.
        :param device: ID аудиоустройства (если None, используется устройство по умолчанию).
        """
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.device = device
        self.stream = None
        self.audio_callback = None

    def start(self):
        """Запуск захвата аудио."""
        if self.stream:
            self.stop()

        # Получаем параметры устройства по умолчанию
        device_info = sd.query_devices(self.device, 'input')
        default_samplerate = int(device_info['default_samplerate'])

        # Используем частоту дискретизации устройства по умолчанию
        self.samplerate = self.samplerate or default_samplerate

        self.stream = sd.InputStream(
            samplerate=self.samplerate,
            blocksize=self.blocksize,
            device=self.device,
            channels=1,  # Моно-аудио
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
        if status:
            logger.warning(f"Audio input status: {status}")

        try:
            if indata.ndim > 1 and indata.shape[1] > 1:
                indata = np.mean(indata, axis=1)  # Преобразуем стерео в моно

            indata = indata.astype(np.float32)

            # Логирование количества данных
            #logger.debug(f"Received {len(indata)} samples at {self.samplerate} Hz")

            # Проверка на минимальное количество сэмплов
            if len(indata) < 2:
                indata = np.pad(indata, (0, 2 - len(indata)), mode="constant")
                logger.warning("Padded input data to avoid resampling error.")

            # Используем Scipy вместо librosa для ресэмплинга
            if self.samplerate != 16000:
                target_len = int(len(indata) * (16000 / self.samplerate))
                indata = resample(indata, target_len)

            # Удаление тишины
            indata, _ = librosa.effects.trim(indata, top_db=30)

            # Преобразование в int16
            indata = np.clip(indata * 32767, -32768, 32767).astype(np.int16)

            #logger.info(f"Processed audio: {indata.shape}, dtype={indata.dtype}, sr=16000")

            if self.audio_callback:
                self.audio_callback(indata)

        except Exception as e:
            logger.error(f"Error processing audio data: {e}")





    def set_audio_callback(self, callback):
        """Установка callback-функции для обработки аудиоданных."""
        self.audio_callback = callback

    @staticmethod
    def list_devices():
        """Возвращает список доступных аудиоустройств."""
        return sd.query_devices()

    @staticmethod
    def get_default_input_device():
        """Возвращает ID устройства по умолчанию для ввода."""
        return sd.default.device[0]