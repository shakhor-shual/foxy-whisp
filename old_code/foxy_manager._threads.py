#!/usr/bin/env python3

import sys
import os
import logging
import time
import socket

from logic.foxy_config import *
from logic.foxy_utils import (
    load_audio_chunk,
    set_logging,
    logger,
    get_port_status,
    create_tokenizer
)
from logic.asr_backends import (
    FasterWhisperASR,
    OpenaiApiASR,
    WhisperTimestampedASR
)
from logic.asr_processors import OnlineASRProcessor, VACOnlineASRProcessor
from logic.mqtt_handler import MQTTHandler
from logic.foxy_processor import LocalAudioSource, TCPAudioSource, FoxyProcessor
from logic.local_audio_input import LocalAudioInput


class FoxyManager:
    def __init__(self, args, stop_event=None, callback=None):
        self.args = args
        self.stop_event = stop_event
        self.callback = callback
        self.asr = None
        self.online = None
        self.mqtt_handler = None
        self.proc = None
        self.setup_logging()
        self.setup_asr()
        self.setup_mqtt()

    def setup_logging(self):
        """Настройка логирования."""
        set_logging(self.args, logger)

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

    def setup_asr(self):
        """Инициализация ASR (Automatic Speech Recognition)."""
        if not self.args.model:
            logger.error("Модель не может быть пустой. Установлено значение по умолчанию: large-v3-turbo.")
            self.args.model = "large-v3-turbo"

        # Создание и настройка ASR
        self.asr = self._create_asr_instance()
        target_language = self._configure_asr(self.asr)

        # Создание токенизатора
        tokenizer = self._create_tokenizer(target_language)

        # Создание OnlineASRProcessor
        self.online = self._create_online_processor(self.asr, tokenizer)

        # Прогрев ASR, если указан файл для прогрева
        if self.args.warmup_file and os.path.isfile(self.args.warmup_file):
            a = load_audio_chunk(self.args.warmup_file, 0, 1)
            self.asr.transcribe(a)
            logger.info("Whisper is warmed up.")
        else:
            logger.warning("Whisper is not warmed up. The first chunk processing may take longer.")

    def setup_mqtt(self):
        """Инициализация MQTT."""
        self.mqtt_handler = MQTTHandler()
        self.mqtt_handler.connect_to_external_broker()

        if not self.mqtt_handler.connected:
            self.mqtt_handler.start_embedded_broker()

        if self.mqtt_handler.connected:
            self.mqtt_handler.publish_message(CONNECTION_TOPIC, "<foxy:started>")
        else:
            logging.error("MQTT client is not connected. Unable to publish message.")


    def start_local_audio(self):
        """Запуск локального аудиоввода."""
        if self.args.audio_device is None:
            self.args.audio_device = LocalAudioInput.get_default_input_device()
        
        local_audio_source = LocalAudioSource(self.args.audio_device, self.callback)
        self.proc = FoxyProcessor(
            audio_source=local_audio_source,
            mqtt_handler=self.mqtt_handler,
            asr_processor=self.online,
            minimal_chunk=self.args.min_chunk_size,
            callback=self.callback
        )
        local_audio_source.local_audio_input.start()  # Запуск через audio_source

        while True:
            if self.stop_event and self.stop_event.is_set():
                logger.info("Server stopping due to stop event.")
                break
            try:
                if not self.proc.process():
                    break
            except Exception as e:
                logger.error(f"Error processing audio: {e}")
                break
        local_audio_source.local_audio_input.stop()



    def start_tcp_server(self):
        """Запуск TCP-сервера."""
        if get_port_status(self.args.port) == 0:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((self.args.host, self.args.port))
                s.listen(1)
                s.settimeout(1)
                logger.info('Listening on' + str((self.args.host, self.args.port)))
                
                while get_port_status(self.args.port) > 0:
                    if self.stop_event and self.stop_event.is_set():
                        logger.info("Server stopping due to stop event.")
                        break

                    try:
                        conn, addr = s.accept()
                        logger.info('Connected to client on {}'.format(addr))
                        
                        tcp_audio_source = TCPAudioSource(conn, timeout=1)
                        self.proc = FoxyProcessor(
                            audio_source=tcp_audio_source,
                            mqtt_handler=self.mqtt_handler,
                            asr_processor=self.online,
                            minimal_chunk=self.args.min_chunk_size,
                            callback=self.callback
                        )

                        while get_port_status(self.args.port) == 1:
                            if self.stop_event and self.stop_event.is_set():
                                logger.info("Server stopping due to stop event.")
                                break

                            try:
                                if not self.proc.process():
                                    break
                            except BrokenPipeError:
                                logger.error("Client disconnected unexpectedly.")
                                break
                            except Exception as e:
                                logger.error(f"Error processing audio: {e}")
                                break

                        conn.close()
                        logger.info('Connection to client closed')
                    except socket.timeout:
                        continue
                    except Exception as e:
                        logger.error(f"Error in server loop: {e}")
                        break

            logger.info('Connection closed, terminating.')
        else:
            logger.info(f'port {self.args.port} already IN USE, terminating.')

    def run(self):
        """Основной метод для запуска сервера."""
        if self.args.listen == "audio_device":
            self.start_local_audio()
        elif self.args.listen == "tcp":
            self.start_tcp_server()
        else:
            logger.error(f"Unsupported listen mode: {self.args.listen}")