from logic.foxy_config import *
from logic.foxy_utils import logger, send_one_line_tcp, receive_lines_tcp

from logic.asr_backends import FasterWhisperASR, OpenaiApiASR, WhisperTimestampedASR
from logic.asr_processors import OnlineASRProcessor, VACOnlineASRProcessor
from logic.mqtt_handler import MQTTHandler
from logic.local_audio_input import LocalAudioInput

import numpy as np
import select
import io
import soundfile #as sf
import librosa
import asyncio

import numpy as np
import soundfile
import librosa
import io
import time


import numpy as np
import soundfile
import librosa
import io
import time
import select
from logic.foxy_utils import logger, send_one_line_tcp, receive_lines_tcp
from logic.local_audio_input import LocalAudioInput

class FoxyProcessor:
    def __init__(self, conn=None, mqtt_handler=None, asr_processor=None, minimal_chunk=None, 
                 language="en", use_local_audio=False, audio_device=None, tcp_echo=True, timeout=1, callback=None):
        """
        Инициализация процессора для обработки аудио и взаимодействия с клиентом.

        :param conn: TCP-соединение (если используется TCP).
        :param mqtt_handler: Обработчик MQTT (опционально).
        :param asr_processor: Процессор ASR для транскрипции.
        :param minimal_chunk: Минимальный размер аудиофрагмента.
        :param language: Язык транскрипции (по умолчанию "en").
        :param use_local_audio: Флаг использования локального аудиоввода.
        :param audio_device: ID аудиоустройства (если используется локальный аудиоввод).
        :param tcp_echo: Флаг эхо-ответа через TCP (по умолчанию True).
        :param timeout: Таймаут для TCP-соединения (по умолчанию 1 секунда).
        :param callback: Функция обратного вызова для обработки результатов транскрипции.
        """
        self.conn = conn
        self.mqtt_handler = mqtt_handler
        self.asr_processor = asr_processor
        self.minimal_chunk = minimal_chunk
        self.language = language
        self.use_local_audio = use_local_audio
        self.audio_device = audio_device
        self.tcp_echo = tcp_echo
        self.timeout = timeout
        self.callback = callback

        self.audio_buffer = np.array([], dtype=np.float32)
        self.last_end = None
        self.is_first = True
        self.void_counter = 0
        self.max_void = 5

        # Инициализация локального аудиоввода
        if self.use_local_audio:
            self.local_audio_input = LocalAudioInput(device=self.audio_device)
            self.local_audio_input.set_audio_callback(self.process_audio_chunk)
            self.local_audio_input.start()

        # Настройка TCP-соединения
        if self.conn:
            self.conn.setblocking(False)
            self.max_cycles = int(timeout / 0.1)  # Таймаут для select

    def process_audio_chunk(self, audio_chunk):
        """
        Обработка аудиофрагмента, полученного от локального устройства.
        """
        if audio_chunk is not None:
            logger.info(f"Received local audio chunk of length: {len(audio_chunk)}")
            self.asr_processor.insert_audio_chunk(audio_chunk)
            outcome = self.asr_processor.process_iter()
            if outcome is not None:
                buff_tail = self.asr_processor.get_tail()
                self.audio_buffer = (
                    self.trim_at_sentence_boundary(self.audio_buffer, buff_tail)
                    if self.has_punctuation(buff_tail)
                    else self.trim_at_clause_boundary(self.audio_buffer, buff_tail)
                )
                self.send_result(outcome, buff_tail)

    def receive_audio_chunk(self):
        """Получает аудиофрагмент от TCP-порта или локального устройства."""
        if self.use_local_audio:
            # Локальный аудиоввод обрабатывается в process_audio_chunk
            return None
        else:
            # Оригинальный код для TCP
            out = []
            min_limit = self.minimal_chunk * SAMPLING_RATE
            while sum(len(x) for x in out) < min_limit:
                raw_bytes = self.non_blocking_receive_audio()
                if not raw_bytes:
                    self.void_counter += 1
                    break
                self.void_counter = 0
                with soundfile.SoundFile(
                    io.BytesIO(raw_bytes),
                    channels=1,
                    endian="LITTLE",
                    samplerate=SAMPLING_RATE,
                    subtype="PCM_16",
                    format="RAW",
                ) as sf:
                    audio, _ = librosa.load(sf, sr=SAMPLING_RATE, dtype=np.float32)
                    out.append(audio)
            if not out:
                return None
            conc = np.concatenate(out)
            if self.is_first and len(conc) < min_limit:
                return None
            self.is_first = False
            return conc

    def non_blocking_receive_audio(self):
        """Получает аудиоданные от TCP-клиента в неблокирующем режиме."""
        cycles = 0
        while cycles < self.max_cycles:
            ready = select.select([self.conn], [], [], 0.1)  # Короткий таймаут для select
            if ready[0]:
                try:
                    raw_bytes = self.conn.recv(4096)  # PACKET_SIZE = 4096
                    if raw_bytes:
                        return raw_bytes
                except (BlockingIOError, ConnectionResetError):
                    return None
            cycles += 1
        return None

    def has_punctuation(self, text):
        """Проверяет, содержит ли текст знаки препинания, завершающие предложение."""
        return any(punc in text for punc in [".", "!", "?"])

    def trim_at_sentence_boundary(self, buffer, text):
        """Обрезает буфер на границе предложения."""
        if self.has_punctuation(text):
            last_punc_end_pos = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
            if last_punc_end_pos != -1:
                return buffer[:last_punc_end_pos + 1]
        return buffer

    def trim_at_clause_boundary(self, buffer, text):
        """Обрезает буфер на границе клаузы."""
        last_punc_mid_pos = max(text.rfind(","), text.rfind(":"))
        if last_punc_mid_pos != -1:
            return buffer[:last_punc_mid_pos + 1]
        return buffer

    def send_result(self, outcome, buff_tail):
        """Отправляет результат транскрипции клиенту."""
        if outcome[0] is not None:
            beg, end = outcome[0] * 1000, outcome[1] * 1000
            if self.last_end is not None:
                beg = max(beg, self.last_end)
            self.last_end = end
            text_commit = outcome[2]
            msg = f"[ {beg:.0f}, {end:.0f}, {end - beg:.0f} ] {text_commit}"
            if self.callback:
                self.callback(text_commit + " ")
            if self.tcp_echo and self.conn:
                send_one_line_tcp(self.conn, msg)
            return True
        return False

    def process(self):
        """Обрабатывает аудиопоток от клиента с обрезкой по знакам препинания."""
        self.asr_processor.init()
        send_error = False
        outcome = None

        while not send_error and self.void_counter < self.max_void:
            if self.use_local_audio:
                # Локальный аудиоввод обрабатывается в process_audio_chunk
                time.sleep(0.1)  # Небольшая задержка для снижения нагрузки на CPU
                continue

            audio_chunk = self.receive_audio_chunk()
            if audio_chunk is None:
                continue

            logger.info(f"Received audio chunk of length: {len(audio_chunk)}")
            self.asr_processor.insert_audio_chunk(audio_chunk)

            if len(audio_chunk) <= 8192 and self.void_counter > 1:
                break

            outcome = self.asr_processor.process_iter()
            if outcome is None:
                break

            buff_tail = self.asr_processor.get_tail()
            self.audio_buffer = (
                self.trim_at_sentence_boundary(self.audio_buffer, buff_tail)
                if self.has_punctuation(buff_tail)
                else self.trim_at_clause_boundary(self.audio_buffer, buff_tail)
            )

            send_error = not self.send_result(outcome, buff_tail)

        final_text = self.asr_processor.finish()
        logger.info(f"Final transcription: {final_text}")

        if final_text and final_text[-1]:
            return self.send_result(final_text, "")
        return self.send_result((None, None, ''), "")
    

# class FoxyCore:
#     def __init__(self, sensory_object=None, asr_processor_object=None, minimal_chunk=None, language="en", use_local_audio=False, audio_device=None):
#         """
#         Инициализация FoxyCore с поддержкой локального аудиоввода и TCP.

#         :param sensory_object: Объект соединения с клиентом (если используется TCP).
#         :param asr_processor_object: Процессор ASR.
#         :param minimal_chunk: Минимальный размер аудиофрагмента.
#         :param language: Язык транскрипции.
#         :param use_local_audio: Флаг использования локального аудиоввода.
#         :param audio_device: ID аудиоустройства (если используется локальный аудиоввод).
#         """
#         self.sensory_object = sensory_object
#         self.asr_processor = asr_processor_object
#         self.minimal_chunk = minimal_chunk
#         self.language = language
#         self.use_local_audio = use_local_audio
#         self.audio_device = audio_device

#         self.audio_buffer = np.array([], dtype=np.float32)
#         self.last_end = None
#         self.is_first = True
#         self.void_counter = 0
#         self.max_void = 5

#         # Инициализация локального аудиоввода
#         if self.use_local_audio:
#             self.local_audio_input = LocalAudioInput(device=self.audio_device)
#             self.local_audio_input.set_audio_callback(self.process_audio_chunk)
#             self.local_audio_input.start()

#     def process_audio_chunk(self, audio_chunk):
#         """
#         Обработка аудиофрагмента, полученного от локального устройства.
#         """
#         if audio_chunk is not None:
#             logger.info(f"Received local audio chunk of length: {len(audio_chunk)}")
#             self.asr_processor.insert_audio_chunk(audio_chunk)
#             outcome = self.asr_processor.process_iter()
#             if outcome is not None:
#                 buff_tail = self.asr_processor.get_tail()
#                 self.audio_buffer = (
#                     self.trim_at_sentence_boundary(self.audio_buffer, buff_tail)
#                     if self.has_punctuation(buff_tail)
#                     else self.trim_at_clause_boundary(self.audio_buffer, buff_tail)
#                 )
#                 self.send_result(outcome, buff_tail)

#     def receive_audio_chunk(self):
#         """Получает аудиофрагмент от TCP-порта или локального устройства."""
#         if self.use_local_audio:
#             # Локальный аудиоввод обрабатывается в process_audio_chunk
#             return None
#         else:
#             # Оригинальный код для TCP
#             out = []
#             min_limit = self.minimal_chunk * SAMPLING_RATE
#             while sum(len(x) for x in out) < min_limit:
#                 raw_bytes = self.sensory_object.non_blocking_receive_audio()
#                 if not raw_bytes:
#                     self.void_counter += 1
#                     break
#                 self.void_counter = 0
#                 with soundfile.SoundFile(
#                     io.BytesIO(raw_bytes),
#                     channels=1,
#                     endian="LITTLE",
#                     samplerate=SAMPLING_RATE,
#                     subtype="PCM_16",
#                     format="RAW",
#                 ) as sf:
#                     audio, _ = librosa.load(sf, sr=SAMPLING_RATE, dtype=np.float32)
#                     out.append(audio)
#             if not out:
#                 return None
#             conc = np.concatenate(out)
#             if self.is_first and len(conc) < min_limit:
#                 return None
#             self.is_first = False
#             return conc

#     def has_punctuation(self, text):
#         """Проверяет, содержит ли текст знаки препинания, завершающие предложение."""
#         if self.language in ["en", "ru"]:
#             return any(punc in text for punc in [".", "!", "?"])
#         return "." in text

#     def trim_at_sentence_boundary(self, buffer, text):
#         """Обрезает буфер на границе предложения."""
#         if buffer is None:
#             logger.warning("Buffer is None, returning empty buffer")
#             return np.array([], dtype=np.float32)

#         if self.has_punctuation(text):
#             last_punc_end_pos = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
#             if last_punc_end_pos != -1:
#                 logger.debug(f"Trimming buffer at last punctuation position: {last_punc_end_pos}")
#                 return buffer[:last_punc_end_pos + 1]

#         return buffer

#     def trim_at_clause_boundary(self, buffer, text):
#         """Обрезает буфер на границе предложения или клаузы."""
#         if buffer is None:
#             logger.warning("Buffer is None, returning empty buffer")
#             return np.array([], dtype=np.float32)

#         last_punc_mid_pos = max(text.rfind(","), text.rfind(":"))
#         if last_punc_mid_pos != -1:
#             logger.debug(f"Trimming buffer at last punctuation position: {last_punc_mid_pos}")
#             return buffer[:last_punc_mid_pos + 1]

#         return buffer

#     def format_output_transcript(self, o, t):
#         """Форматирует результат транскрипции."""
#         if o[0] is not None:
#             beg, end = o[0] * 1000, o[1] * 1000
#             if self.last_end is not None:
#                 beg = max(beg, self.last_end)

#             self.last_end = end
#             text_commit = o[2]
#             return f"[ {beg:.0f}, {end:.0f}, {end - beg:.0f} ] {text_commit}"

#         logger.debug("No text in this segment")
#         return None

#     def send_result(self, o, t):
#         """Отправляет результат транскрипции клиенту."""
#         msg = self.format_output_transcript(o, t)
#         try:
#             if msg is not None:
#                 self.sensory_object.send(msg)
#             if t is not None:
#                 self.sensory_object.send(t)
#             return True
#         except BrokenPipeError:
#             logger.info("Broken pipe -- connection closed?")
#             return False

#     def process(self):
#         """Обрабатывает аудиопоток от клиента с обрезкой по знакам препинания."""
#         self.asr_processor.init()
#         send_error = False
#         outcome = None

#         while not send_error and self.void_counter < self.max_void:
#             if self.use_local_audio:
#                 # Локальный аудиоввод обрабатывается в process_audio_chunk
#                 time.sleep(0.1)  # Небольшая задержка для снижения нагрузки на CPU
#                 continue

#             audio_chunk = self.receive_audio_chunk()
#             if audio_chunk is None:
#                 continue

#             logger.info(f"Received audio chunk of length: {len(audio_chunk)}")
#             self.asr_processor.insert_audio_chunk(audio_chunk)

#             if len(audio_chunk) <= 8192 and self.void_counter > 1:
#                 break

#             outcome = self.asr_processor.process_iter()
#             if outcome is None:
#                 break

#             buff_tail = self.asr_processor.get_tail()
#             self.audio_buffer = (
#                 self.trim_at_sentence_boundary(self.audio_buffer, buff_tail)
#                 if self.has_punctuation(buff_tail)
#                 else self.trim_at_clause_boundary(self.audio_buffer, buff_tail)
#             )

#             send_error = not self.send_result(outcome, buff_tail)

#         final_text = self.asr_processor.finish()
#         logger.info(f"Final transcription: {final_text}")

#         if final_text and final_text[-1]:
#             return self.send_result(final_text, "")
#         return self.send_result((None, None, ''), "")
    

# #######################################################################################################
# class FoxySensory:
# #######################
#     def __init__(self, conn, mqtt_handler, tcp_echo =True, timeout=1, callback=None):
#         """
#         :param conn: socket connection object
#         :param timeout: timeout in seconds for data inactivity
#         """
#         self.tcp_echo=tcp_echo
#         self.mqtt_handler=mqtt_handler
#         self.conn = conn
#         self.last_line = ""
#         self.timeout = timeout
#         self.conn.setblocking(False)  # Set the socket to non-blocking mode
#         self.max_cycles = int(timeout / CONNECTION_SELECT_DELAY)  # Calculate max cycles based on timeout
#         self.callback = callback

# #######################
#     def send(self, line):
#         '''It doesn't send the same line twice, to avoid duplicate transmissions.'''
#         if line == self.last_line:
#             return
        
#         payload= line.split("]", 1)[1].strip() if line.startswith("[") else None
#         if payload and self.callback: 
#             self.callback(payload + " ")
#         #if payload and self.mqtt_handler.connected:
#          #   self.mqtt_handler.publish_message(CONNECTION_TOPIC, payload + " ")


#         if self.tcp_echo:
#             send_one_line_tcp(self.conn, line)
#         self.last_line = line

# #######################
#     def receive_lines(self):
#         '''Receive lines of text data.'''
#         return receive_lines_tcp(self.conn)

# #######################
#     def non_blocking_receive_audio(self):
#         """ Receive audio data in a non-blocking manner. """
#         cycles = 0

#         while cycles < self.max_cycles:
#             # Check if the socket has data ready to read
#             ready = select.select([self.conn], [], [], CONNECTION_SELECT_DELAY)  # Short timeout for polling
#             if ready[0]:
#                 try:
#                     raw_bytes = self.conn.recv(PACKET_SIZE)
#                     if raw_bytes != b"":
#                         return raw_bytes  # Return immediately if data is received
#                 except BlockingIOError:
#                     pass
#                 except ConnectionResetError:
#                     self.socket_closed = True
#                     return None  # Connection was reset
            
#             cycles += 1 # Increment the cycle count if no data is received
#         return None  # Return None after timeout

