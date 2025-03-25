from multiprocessing import Process, Queue
from abc import ABC, abstractmethod
import logging
import time
import  multiprocessing 
from foxy_pipeline import VADunit, ASRunit, AudioSource

logger = logging.getLogger(__name__)

import multiprocessing
from multiprocessing import Queue
from logic.foxy_config import *
from logic.foxy_utils import logger
from logic.asr_backends import FasterWhisperASR, OpenaiApiASR, WhisperTimestampedASR
from logic.asr_processors import OnlineASRProcessor, VACOnlineASRProcessor
from logic.foxy_processor import LocalAudioSource, TCPAudioSource, FoxyProcessor
from logic.local_audio_input import LocalAudioInput

class FoxyEngine:
    def __init__(self, args, mqtt_handler=None, gui_callback=None):
        self.args = args
        self.stop_event = multiprocessing.Event()

        # Очереди для управления компонентами
        self.audio_source_control_queue = Queue()
        self.vad_control_queue = Queue()
        self.asr_control_queue = Queue()

        # Очереди для передачи данных между компонентами
        self.audio_to_vad_queue = Queue(maxsize=100)
        self.vad_to_asr_queue = Queue(maxsize=100)

        # Компоненты конвейера
        self.audio_source = None
        self.vad = None
        self.asr = None
        self.proc = None

        # Внешние зависимости
        self.mqtt_handler = mqtt_handler
        self.gui_callback = gui_callback  # Callback для отправки данных в GUI

    def start_pipeline(self):
        """Настраивает и запускает конвейер на основе аргументов из argparse."""
        # Настройка аудиоисточника
        if self.args.listen == "audio_device":
            if self.args.audio_device is None:
                self.args.audio_device = LocalAudioInput.get_default_input_device()
            self.audio_source = LocalAudioSource(self.args.audio_device, self.send_data_to_gui)
        elif self.args.listen == "tcp":
            self.audio_source = TCPAudioSource(timeout=1)
        else:
            raise ValueError(f"Unsupported listen mode: {self.args.listen}")

        # Настройка ASR
        self.asr = self._create_asr_instance()
        target_language = self._configure_asr(self.asr)

        # Создание токенизатора
        tokenizer = self._create_tokenizer(target_language)

        # Создание OnlineASRProcessor
        self.online = self._create_online_processor(self.asr, tokenizer)

        # Настройка FoxyProcessor
        self.proc = FoxyProcessor(
            audio_source=self.audio_source,
            mqtt_handler=self.mqtt_handler,
            asr_processor=self.online,
            minimal_chunk=self.args.min_chunk_size,
            callback=self.send_data_to_gui
        )

        # Запуск компонентов
        if self.args.listen == "audio_device":
            self.audio_source.local_audio_input.start()
        self.proc.start()

        logger.info("FoxyEngine pipeline started.")

    def stop_pipline(self):
        """Останавливает конвейер."""
        self.stop_event.set()
        if self.proc:
            self.proc.stop()
        if self.audio_source and self.args.listen == "audio_device":
            self.audio_source.local_audio_input.stop()
        logger.info("FoxyEngine pipeline stopped.")

    def _create_asr_instance(self):
        """Создает экземпляр ASR в зависимости от выбранного бэкенда."""
        backend = self.args.backend
        if backend == "openai-api":
            logger.debug("Using OpenAI API.")
            return OpenaiApiASR(lan=self.args.lan)
        else:
            asr_cls = FasterWhisperASR if backend == "faster-whisper" else WhisperTimestampedASR
            size = self.args.model
            logger.info(f"Loading Whisper {size} model for {self.args.lan}...")
            start_time = time.time()
            asr = asr_cls(modelsize=size, lan=self.args.lan, cache_dir=self.args.model_cache_dir, model_dir=self.args.model_dir)
            logger.info(f"Model loaded in {round(time.time() - start_time, 2)} seconds.")
            return asr

    def _configure_asr(self, asr):
        """Настраивает ASR (VAD, язык, задача перевода)."""
        if getattr(self.args, 'vad', False):
            logger.info("Setting VAD filter")
            asr.use_vad()

        if self.args.task == "translate":
            asr.set_translate_task()
            return "en"  # Whisper переводит на английский
        return self.args.lan  # Whisper транскрибирует на указанный язык

    def _create_tokenizer(self, language):
        """Создает токенизатор, если требуется."""
        if self.args.buffer_trimming == "sentence":
            return create_tokenizer(language)
        return None

    def _create_online_processor(self, asr, tokenizer):
        """Создает экземпляр OnlineASRProcessor."""
        if self.args.vac:
            return VACOnlineASRProcessor(
                self.args.min_chunk_size, asr, tokenizer, logfile=sys.stderr,
                buffer_trimming=(self.args.buffer_trimming, self.args.buffer_trimming_sec)
            )
        return OnlineASRProcessor(
            asr, tokenizer, logfile=sys.stderr,
            buffer_trimming=(self.args.buffer_trimming, self.args.buffer_trimming_sec)
        )

    def send_data_to_gui(self, data):
        """Отправка данных в GUI."""
        if self.gui_callback:
            self.gui_callback(data)