from abc import ABC, abstractmethod
import logging
import numpy as np
import webrtcvad
import torch
from openvino.runtime import Core
from multiprocessing import Queue
import time
from logic.foxy_message import PipelineMessage, MessageType

logger = logging.getLogger(__name__)

class AudioBuffer:
    """Буфер для хранения аудиоданных с возможностью чтения чанков переменной длины."""
    def __init__(self, sample_rate=16000, max_duration=5):
        self.sample_rate = sample_rate
        self.max_size = sample_rate * max_duration  # Максимальный размер буфера в сэмплах
        self.buffer = np.array([], dtype=np.float32)

    def add_data(self, audio_data):
        """Добавление новых аудиоданных в буфер с приведением размерности."""
        try:
            # Убедимся что входные данные одномерные
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Убедимся что буфер одномерный
            if self.buffer.ndim > 1:
                self.buffer = np.mean(self.buffer, axis=1)
                
            # Приводим типы к float32
            audio_data = audio_data.astype(np.float32)
            self.buffer = self.buffer.astype(np.float32)
            
            # Теперь конкатенируем
            self.buffer = np.concatenate((self.buffer, audio_data))
            
            # Обрезаем если превышен максимальный размер
            if len(self.buffer) > self.max_size:
                self.buffer = self.buffer[-self.max_size:]
                
        except Exception as e:
            print(f"Error in add_data: {e}")
            # В случае ошибки очищаем буфер чтобы избежать проблем
            self.buffer = np.array([], dtype=np.float32)
            raise

    def get_chunk(self, chunk_size):
        """Получение чанка данных из буфера."""
        # Fix: Правильная проверка размера буфера
        if len(self.buffer) < chunk_size or chunk_size <= 0:
            return None
            
        try:
            # Извлекаем чанк
            chunk = self.buffer[:chunk_size].copy()  # Используем copy() для безопасности
            # Обновляем буфер
            self.buffer = self.buffer[chunk_size:]
            return chunk
        except Exception as e:
            print(f"Error in get_chunk: {e}")
            return None

class VADBase(ABC):
    def __init__(self):
        self.input_queue = None
        self.output_queue = None
        self.control_queue = None  # Очередь для отправки статусов
        self.current_state = None  # Текущее состояние VAD (речь/тишина)
        self.audio_buffer = AudioBuffer()  # Добавляем буфер аудиоданных
        self.last_status_time = 0
        self.min_status_interval = 0.03  # Увеличиваем частоту до ~33 Hz
        self.vad_id = f"vad_{id(self)}"  # Уникальный идентификатор VAD
        self.vad_type = self.__class__.__name__
        # Добавляем параметры для плавного затухания
        self.fade_frames = 20  # Количество фреймов для затухания
        self.current_fade_counter = 0  # Текущий счетчик затухания
        self.is_fading = False  # Флаг состояния затухания
        self.last_voice_detected = False  # Последнее состояние детекции речи

        # Добавляем общие параметры обработки аудио
        self.sample_rate = 16000
        self.frame_duration = 20  # мс
        self.frame_size = int(self.sample_rate * self.frame_duration / 1000)
        
        # История детекции речи
        self.speech_history = []
        self.history_size = 3
        self.sensitivity = 0.2

    def connect_input(self, input_queue):
        """Подключение входной очереди."""
        self.input_queue = input_queue

    def connect_output(self, output_queue):
        """Подключение выходной очереди."""
        self.output_queue = output_queue

    def connect_control(self, control_queue):
        """Подключение управляющей очереди."""
        self.control_queue = control_queue

    def add_to_buffer(self, audio_data):
        """Добавление аудиоданных в буфер."""
        self.audio_buffer.add_data(audio_data)

    def preprocess_audio(self, audio_chunk):
        """Общая предобработка аудио для всех VAD"""
        try:
            if not isinstance(audio_chunk, np.ndarray):
                raise ValueError("Input is not numpy array")

            # Нормализация до float32 [-1, 1]
            if audio_chunk.dtype != np.float32:
                audio_chunk = audio_chunk.astype(np.float32)
            if np.max(np.abs(audio_chunk)) > 1.0:
                audio_chunk = audio_chunk / 32768.0

            # Преобразование в int16
            audio_int16 = (audio_chunk * 32767).astype(np.int16)
            return audio_int16
            
        except Exception as e:
            self.send_exception(e, "Audio preprocessing error")
            return None

    def split_into_frames(self, audio_data):
        """Разделение аудио на фреймы"""
        if len(audio_data) < self.frame_size:
            return []
            
        num_frames = len(audio_data) // self.frame_size
        frame_length = num_frames * self.frame_size
        return np.array_split(audio_data[:frame_length], num_frames)

    def update_speech_history(self, is_speech: bool) -> bool:
        """Обновление истории речи и проверка порога"""
        self.speech_history.append(is_speech)
        self.speech_history = self.speech_history[-self.history_size:]
        
        if len(self.speech_history) >= self.history_size:
            speech_ratio = sum(self.speech_history) / len(self.speech_history)
            return speech_ratio >= self.sensitivity
        return False

    def detect_voice(self, audio_chunk):
        """Общая логика детекции голоса с предобработкой"""
        start_time = time.time()
        try:
            # Предобработка аудио
            processed_audio = self.preprocess_audio(audio_chunk)
            if processed_audio is None:
                return False

            # Разделение на фреймы
            frames = self.split_into_frames(processed_audio)
            if not frames:
                return False

            # Анализ фреймов
            raw_voice_detected = False
            for frame in frames:
                if len(frame) == self.frame_size:
                    frame_has_speech = self.process_frame(frame)
                    if self.update_speech_history(frame_has_speech):
                        raw_voice_detected = True
                        break

            # Применяем логику затухания
            final_voice_detected = self.apply_fade_logic(raw_voice_detected)

            # Отправляем расширенный статус
            self.send_status(
                "processing",
                frames_total=len(frames),
                frames_with_speech=sum(self.speech_history),
                frame_size=self.frame_size,
                raw_voice_detected=raw_voice_detected,
                voice_detected=final_voice_detected,
                is_fading=self.is_fading,
                fade_counter=self.current_fade_counter,
                performance={'processing_time_ms': (time.time() - start_time) * 1000}
            )

            return final_voice_detected

        except Exception as e:
            self.send_exception(e, "VAD processing error")
            return False

    @abstractmethod
    def process_frame(self, frame):
        """Абстрактный метод обработки отдельного фрейма"""
        pass

    def send_status(self, status: str, **details):
        """Enhanced status sending with higher update rate"""
        current_time = time.time()
        if current_time - self.last_status_time >= self.min_status_interval:
            if self.control_queue:
                # Добавляем vad_info в детали
                vad_info = {
                    'vad_id': self.vad_id,
                    'vad_type': self.vad_type,
                    'vad_config': self.get_config()
                }
                
                if 'details' in details:
                    details['details'].update(vad_info)
                else:
                    details['details'] = vad_info

                msg = PipelineMessage.create_status(
                    source='src.vad',  # Изменяем формат source
                    status=status,
                    **details
                )
                msg.send(self.control_queue)
                self.last_status_time = current_time

    def get_config(self):
        """Получение конфигурации VAD для логирования"""
        return {
            'fade_frames': self.fade_frames,
            'is_fading': self.is_fading,
            'fade_counter': self.current_fade_counter
        }

    def apply_fade_logic(self, voice_detected: bool) -> bool:
        """Применение логики плавного затухания"""
        # При обнаружении речи сразу сбрасываем затухание
        if voice_detected:
            self.is_fading = False
            self.current_fade_counter = self.fade_frames
            self.last_voice_detected = True
            return True

        # Если речь не обнаружена, но есть активное затухание
        if self.current_fade_counter > 0:
            self.is_fading = True
            self.current_fade_counter -= 1
            return True
        
        # Речь не обнаружена и затухание завершено
        self.is_fading = False
        self.last_voice_detected = False
        return False

    def _update_state(self, new_state):
        """Обновление состояния VAD и отправка статуса при изменении."""
        if new_state != self.current_state:
            self.current_state = new_state
            self.send_status('state_changed', 
                           state=new_state, 
                           timestamp=time.time())

    def process(self):
        """Обработка аудиоданных из буфера."""
        chunk_size = self.get_chunk_size()
        audio_chunk = self.audio_buffer.get_chunk(chunk_size)
        if audio_chunk is None:
            return None  # Недостаточно данных для обработки

        try:
            voice_detected = self.detect_voice(audio_chunk)
            self._update_state("speech" if voice_detected else "silence")
            
            # Send detailed status
            self.send_status(
                "processing",
                voice_detected=voice_detected,
                chunk_size=chunk_size,
                buffer_size=len(self.audio_buffer.buffer)
            )
            
            return audio_chunk if voice_detected else None
            
        except Exception as e:
            self.send_status("error", error=str(e))
            return None

    def run(self):
        """Основной цикл обработки."""
        while True:
            try:
                audio_data = self.input_queue.get()
                if audio_data is None:  # Сигнал завершения
                    break

                self.add_to_buffer(audio_data)
                while True:
                    processed_data = self.process()
                    if processed_data is None:
                        break  # Недостаточно данных для обработки
                    self.output_queue.put(processed_data)

            except Exception as e:
                logger.error(f"Error in VAD: {e}")
                break


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

    def get_chunk_size(self):
        """Возвращает размер чанка для Silero VAD."""
        return 16000  # Пример: 1 секунда аудио при 16 кГц


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

    def get_chunk_size(self):
        """Возвращает размер чанка для OpenVINO VAD."""
        return 16000  # Пример: 1 секунда аудио при 16 кГц


class WebRTCVAD(VADBase):
    def __init__(self, aggressiveness=3):
        super().__init__()
        self.vad = webrtcvad.Vad(aggressiveness)
        self.aggressiveness = aggressiveness

    def process_frame(self, frame):
        """Обработка отдельного фрейма для WebRTC VAD"""
        try:
            return self.vad.is_speech(frame.tobytes(), self.sample_rate)
        except Exception as e:
            self.send_exception(e, "WebRTC frame processing error")
            return False

    def get_chunk_size(self):
        """Возвращает размер чанка для WebRTC VAD."""
        return self.frame_size

    def get_config(self):
        """Возвращает расширенную конфигурацию WebRTC VAD"""
        config = super().get_config()
        config.update({
            'aggressiveness': self.aggressiveness,
            'sample_rate': self.sample_rate,
            'frame_duration': self.frame_duration,
            'frame_size': self.frame_size
        })
        return config


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

