from abc import ABC, abstractmethod
import logging
import numpy as np
import webrtcvad
import torch
from openvino.runtime import Core
from multiprocessing import Queue

logger = logging.getLogger(__name__)

class VADBase(ABC):
    def __init__(self):
        self.input_queue = None
        self.output_queue = None
        self.control_queue = None  # Очередь для отправки статусов
        self.current_state = None  # Текущее состояние VAD (речь/тишина)

    def connect_input(self, input_queue):
        """Подключение входной очереди."""
        self.input_queue = input_queue

    def connect_output(self, output_queue):
        """Подключение выходной очереди."""
        self.output_queue = output_queue

    def connect_control(self, control_queue):
        """Подключение управляющей очереди."""
        self.control_queue = control_queue

    @abstractmethod
    def detect_voice(self, audio_chunk):
        """Определение наличия голоса в аудиоданных."""
        pass

    def _update_state(self, new_state):
        """Обновление состояния VAD и отправка статуса при изменении."""
        if new_state != self.current_state:
            self.current_state = new_state
            if self.control_queue:
                self.control_queue.put({"vad_status": self.current_state})

    def process(self, audio_chunk):
        """Обработка аудиоданных."""
        voice_detected = self.detect_voice(audio_chunk)
        self._update_state("speech" if voice_detected else "silence")
        return audio_chunk if voice_detected else None

    def run(self):
        """Основной цикл обработки."""
        while True:
            try:
                audio_chunk = self.input_queue.get()
                if audio_chunk is None:  # Сигнал завершения
                    break

                processed_data = self.process(audio_chunk)
                if processed_data is not None:
                    self.output_queue.put(processed_data)

            except Exception as e:
                logger.error(f"Error in VAD: {e}")
                break


class NullVAD(VADBase):
    def detect_voice(self, audio_chunk):
        """Всегда возвращает True (речь обнаружена)."""
        return True


class WebRTCVAD(VADBase):
    def __init__(self, aggressiveness=3):
        super().__init__()
        self.vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = 16000  # WebRTC VAD поддерживает только 16 кГц
        self.frame_duration = 30  # Длительность фрейма в миллисекундах
        self.frame_size = int(self.sample_rate * self.frame_duration / 1000)

    def detect_voice(self, audio_chunk):
        """Анализ аудиоданных с использованием WebRTC VAD."""
        if len(audio_chunk) < self.frame_size:
            return False  # Недостаточно данных для анализа

        # Разделение данных на фреймы
        frames = [
            audio_chunk[i : i + self.frame_size]
            for i in range(0, len(audio_chunk), self.frame_size)
        ]

        # Анализ каждого фрейма
        return any(self.vad.is_speech(frame.tobytes(), self.sample_rate) for frame in frames)


class SileroVAD(VADBase):
    def __init__(self):
        super().__init__()
        self.model, self.utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=True,
        )
        self.get_speech_timestamps = self.utils[0]

    def detect_voice(self, audio_chunk):
        """Анализ аудиоданных с использованием Silero VAD."""
        audio_tensor = torch.from_numpy(audio_chunk).float()
        speech_timestamps = self.get_speech_timestamps(audio_tensor, self.model)
        return bool(speech_timestamps)  # True, если найдены сегменты с речью


class OpenVINOVAD(VADBase):
    def __init__(self, model_path):
        super().__init__()
        self.core = Core()
        self.model = self.core.read_model(model_path)
        self.compiled_model = self.core.compile_model(self.model, "CPU")
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)

    def detect_voice(self, audio_chunk):
        """Анализ аудиоданных с использованием OpenVINO VAD."""
        # Подготовка входных данных
        input_data = np.expand_dims(audio_chunk, axis=0)

        # Выполнение инференса
        result = self.compiled_model(input_data)[self.output_layer]

        # Возвращаем True, если модель обнаружила речь
        return result > 0.5


def create_vad(args):
    """Фабрика для создания VAD на основе аргументов."""
    vad_type = args.get("vad", "")#.lower()

    if vad_type == "webrtc":
        return WebRTCVAD(aggressiveness=args.get("vad_aggressiveness", 3))
    elif vad_type == "silero":
        return SileroVAD()
    elif vad_type == "openvino":
        model_path = args.get("vad_model_path", "vad_model.xml")
        return OpenVINOVAD(model_path)
    else:
        return NullVAD()  # По умолчанию используется нулевая имплементация
    
    