#logic/local_audio_input
from collections import deque
import numpy as np
import sounddevice as sd
from scipy.signal import resample
import librosa
import logging

logger = logging.getLogger(__name__)

class BaseVAD:
    """Базовый класс для детектора голоса (VAD)."""
    def __init__(self, fifo):
        self.fifo = fifo  # Ссылка на FIFO-буфер

    def analyze(self):
        """
        Анализирует данные из FIFO и возвращает состояние флага voice_detected.
        В базовой реализации всегда возвращает True.
        """
        # Извлекаем данные из FIFO для анализа
        analysis_data = np.array(self.fifo)
        if len(analysis_data) > 0:
            # Здесь можно добавить логику анализа данных
            return True  # Всегда возвращаем True для фейкового VAD
        return False

class LocalAudioInput:
    def __init__(self, samplerate=None, blocksize=32768, device=None, vad_frame_size=16000, accumulation_buffer_size=1):
        """
        Инициализация захвата аудио с локального устройства.

        :param samplerate: Частота дискретизации (если None, используется частота устройства по умолчанию).
        :param blocksize: Размер блока аудиоданных.
        :param device: ID аудиоустройства (если None, используется устройство по умолчанию).
        :param vad_frame_size: Минимальное количество данных для анализа ВАД (в сэмплах).
        :param accumulation_buffer_size: Размер буфера для накопления (в секундах).
        """
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.device = device
        self.stream = None
        self.audio_callback = None

        # FIFO для временного хранения данных
        self.fifo = deque()  # Динамический FIFO
        self.vad_frame_size = vad_frame_size  # Минимальный размер данных для анализа ВАД

        # Буфер для накопления
        self.accumulation_buffer = deque(maxlen=16000 * accumulation_buffer_size)

        # Инициализация базового VAD
        self.vad = BaseVAD(self.fifo)
        self.voice_detected = False  # Флаг детекции голоса

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
        """Callback-функция для обработки аудиоданных."""
        if status:
            logger.warning(f"Audio input status: {status}")

        try:
            # Обработка данных (преобразование стерео в моно, ресемплинг и т.д.)
            processed_data = self._process_audio(indata)

            # Добавление данных в FIFO
            self.fifo.extend(processed_data)

            # Проверяем, достаточно ли данных для анализа ВАД
            if len(self.fifo) >= self.vad_frame_size:
                # Анализ данных с помощью VAD
                self.voice_detected = self.vad.analyze()

                # Передача данных в буфер для накопления (если голос обнаружен)
                if self.voice_detected:
                    self.accumulation_buffer.extend(self.fifo)

                # Очистка FIFO (если данные переданы в буфер)
                if self.voice_detected:
                    self.fifo.clear()

            # Передача данных из буфера в callback (если буфер заполнен)
            if len(self.accumulation_buffer) >= self.accumulation_buffer.maxlen:
                if self.audio_callback:
                    self.audio_callback(np.array(self.accumulation_buffer))
                self.accumulation_buffer.clear()

        except Exception as e:
            logger.error(f"Error processing audio data: {e}")

    def _process_audio(self, indata):
        """Обработка аудиоданных: преобразование стерео в моно, ресемплинг и т.д."""
        if indata.ndim > 1 and indata.shape[1] > 1:
            indata = np.mean(indata, axis=1)  # Преобразуем стерео в моно

        indata = indata.astype(np.float32)

        # Ресемплинг
        if self.samplerate != 16000:
            target_len = int(len(indata) * (16000 / self.samplerate))
            indata = resample(indata, target_len)

        # Удаление тишины
        indata, _ = librosa.effects.trim(indata, top_db=30)

        # Преобразование в int16
        indata = np.clip(indata * 32767, -32768, 32767).astype(np.int16)

        return indata

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