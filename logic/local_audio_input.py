from logic.foxy_config import *
from logic.foxy_utils import logger, send_one_line_tcp, receive_lines_tcp
import sounddevice as sd
import numpy as np

import sounddevice as sd
import numpy as np

class LocalAudioInput:
    def __init__(self, samplerate=SAMPLING_RATE, blocksize=8192, device=None):
        """
        Инициализация захвата аудио с локального устройства.

        :param samplerate: Частота дискретизации (по умолчанию из конфига).
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
        """
        if status:
            logger.warning(f"Audio input status: {status}")

        # Передача данных в FoxyCore
        if self.audio_callback:
            self.audio_callback(indata.copy())

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