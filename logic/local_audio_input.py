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

class LocalAudioInput:
    def __init__(self, samplerate=None, blocksize=8192, device=None):
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
        """
        Callback-функция, вызываемая при поступлении новых аудиоданных.
        Выполняет ресамплинг, преобразование в моно-формат, удаление тишины
        и преобразование в 16-битный знаковый целочисленный формат.
        """
        if status:
            logger.warning(f"Audio input status: {status}")

        try:
            # Преобразуем данные в моно-формат (если они стерео)
            if indata.ndim > 1 and indata.shape[1] > 1:
                indata = np.mean(indata, axis=1)  # Преобразуем стерео в моно

            # Ресамплинг данных до 16 кГц
            if self.samplerate != 16000:
                indata = librosa.resample(
                    indata.flatten(),
                    orig_sr=self.samplerate,
                    target_sr=16000,
                    res_type="kaiser_fast"  # Используем быстрый метод ресамплинга
                )

            # Удаление тишины (опционально, порог -30 dB)
            indata, _ = librosa.effects.trim(indata, top_db=30)

            # Преобразование в 16-битный знаковый целочисленный формат
            indata = (indata * 32767).astype(np.int16)  # Масштабирование и преобразование

            # Логирование для отладки
            logger.info(f"Processed audio data: shape={indata.shape}, dtype={indata.dtype}, sr=16000")

            # Передача данных в FoxyCore
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