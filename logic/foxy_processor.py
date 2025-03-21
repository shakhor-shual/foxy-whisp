import io
import time
import select
import numpy as np
import soundfile as sf
import librosa
from collections import deque
from abc import ABC, abstractmethod
from itertools import islice

from logic.foxy_config import *
from logic.foxy_utils import logger, send_one_line_tcp, receive_lines_tcp
from logic.local_audio_input import LocalAudioInput

SAMPLING_RATE = 16000  # Предполагаемая частота дискретизации
PACKET_SIZE = 4096

class AudioSource(ABC):
    @abstractmethod
    def receive_audio(self):
        pass

class TCPAudioSource(AudioSource):
    def __init__(self, conn, timeout=1):
        self.conn = conn
        self.conn.setblocking(False)
        self.timeout = timeout
        self.max_cycles = int(timeout / 0.1)
    
    def receive_audio(self):
        cycles = 0
        while cycles < self.max_cycles:
            ready = select.select([self.conn], [], [], 0.1)
            if ready[0]:
                try:
                    return self.conn.recv(PACKET_SIZE)
                except (BlockingIOError, ConnectionResetError):
                    return None
            cycles += 1
        return None

class LocalAudioSource(AudioSource):
    def __init__(self, device, callback=None):
        self.local_audio_input = LocalAudioInput(device=device)
        self.audio_buffer = deque(maxlen=1024)  # Буфер для хранения аудиоданных
        self.callback = callback

        # Запускаем захват аудио
        self.local_audio_input.set_audio_callback(self._audio_callback)
        self.local_audio_input.start()

    def _audio_callback(self, audio_data):
        """Callback для захвата аудиоданных."""
       
        if audio_data is not None:
            self.audio_buffer.append(audio_data)  # Добавляем данные в буфер

    def receive_audio(self):
        """Возвращает аудиоданные из буфера."""
        if len(self.audio_buffer) > 0:
            return self.audio_buffer.popleft()  # Возвращаем данные из буфера
        return None  # Если буфер пуст

class FoxyProcessor:
    def __init__(self, audio_source, mqtt_handler=None, asr_processor=None, minimal_chunk=None, 
                 language="en", tcp_echo=True, callback=None):
        self.audio_source = audio_source
        self.mqtt_handler = mqtt_handler
        self.asr_processor = asr_processor
        self.minimal_chunk = minimal_chunk
        self.language = language
        self.tcp_echo = tcp_echo
        self.gui_callback = callback
        self.void_counter = 0
        self.max_void = 500
        self.is_first = True
        self.audio_buffer = deque(maxlen=10240)
        self.last_end = None
        self.is_tcp_mode=isinstance(self.audio_source, TCPAudioSource)
        self.is_local_mode=isinstance(self.audio_source, LocalAudioSource)

        self.text_commited=""
        self.text_uncommited=""


    def trim_text(self, buffer, text, punctuation_end=None, punctuation_mid=None):
        punctuation_end = punctuation_end or [".", "!", "?"]
        punctuation_mid = punctuation_mid or [",", ":"]
        
        last_end = max((text.rfind(p) for p in punctuation_end), default=-1)
        last_mid = max((text.rfind(p) for p in punctuation_mid), default=-1)
        last_punc = last_end if last_end != -1 else last_mid
        
        return deque(islice(buffer, last_punc + 1)) if last_punc != -1 else buffer



    def send_result(self, asr_outcome=None, text_uncommited=None):
        tcp_echo_on =self.is_tcp_mode and self.tcp_echo

        if asr_outcome[0] is not None:
            beg, end = asr_outcome[0] * 1000, asr_outcome[1] * 1000
            beg = max(beg, self.last_end or 0)
            self.last_end = end
            text_commited = asr_outcome[2]
            if  self.text_commited != text_commited:  
                commit_report = f"[ {beg:.0f}, {end:.0f}, {end - beg:.0f} ] {text_commited}"
                if self.gui_callback:
                    self.gui_callback(text_commited + " ")
                if tcp_echo_on:
                    send_one_line_tcp(self.audio_source.conn, commit_report)
            self.text_commited = text_commited 
        
        if text_uncommited is not None:
            if tcp_echo_on and (self.text_uncommited != text_uncommited): 
                send_one_line_tcp(self.audio_source.conn, text_uncommited)
            self.text_uncommited != text_uncommited
            
    

    def show_audio_level(self, audio_chunk=None):
        if self.gui_callback:
            """Вычисление уровня аудиосигнала в dB."""
            if audio_chunk is None or len(audio_chunk) == 0:
                self.gui_callback(0)
                return   # Сброс в ноль при отсутствии данных

            # Вычисляем RMS (среднеквадратичное значение)
            rms = np.sqrt(np.mean(np.square(audio_chunk)))

            # Преобразуем RMS в dB
            min_rms = 1e-5  # Минимальное значение, чтобы избежать log(0)
            rms = max(rms, min_rms)
            db = 20 * np.log10(rms)

            # Нормализуем значение для Progressbar (например, от -60 dB до 0 dB)
            min_db = -60
            max_db = 0
            db = max(min_db, min(db, max_db))  # Ограничиваем диапазон
            normalized_level = int((db - min_db) / (max_db - min_db) * 100)  # Масштабируем до 0-100

            self.gui_callback(normalized_level)


    def receive_audio_chunk(self):
        """Получение аудиочанка и отслеживание простоев."""
        out = []
        min_limit = self.minimal_chunk * SAMPLING_RATE
        print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        while sum(len(x) for x in out) < min_limit:
            raw_bytes = self.audio_source.receive_audio()
            print(f"===VOID_COUNT========={self.void_counter}============")
            #if not raw_bytes:
            if raw_bytes is None or len(raw_bytes) == 0:
                # Увеличиваем счетчик циклов без данных
                self.void_counter += 1

                # Сбрасываем индикатор, если счетчик превысил порог
                if self.void_counter > self.max_void // 2:  # Порог — половина от max_void
                    if self.gui_callback:
                        self.gui_callback(0)  # Сброс индикатора в ноль
                break
            else:
                # Сбрасываем счетчик, если данные поступают
                self.void_counter = 0

            # Декодируем аудиоданные
            with sf.SoundFile(io.BytesIO(raw_bytes), channels=1, endian="LITTLE", samplerate=SAMPLING_RATE,
                            subtype="PCM_16", format="RAW") as sfile:
                audio, _ = librosa.load(sfile, sr=SAMPLING_RATE, dtype=np.float32)
                out.append(audio)

        return np.concatenate(out) if out else None



    def handle_audio_chunk(self):
        """Обработка аудиочанка и обновление индикатора."""

        audio_chunk = self.receive_audio_chunk()
        self.show_audio_level(audio_chunk)
        
        if audio_chunk is None or len(audio_chunk) == 0:
            return 

        # Обработка аудиочанка в ASR
        self.asr_processor.insert_audio_chunk(audio_chunk)
        outcome = self.asr_processor.process_iter()
        if outcome:
            buff_tail = self.asr_processor.get_tail()
            self.audio_buffer = self.trim_text(self.audio_buffer, buff_tail)
            self.send_result(asr_outcome=outcome, text_uncommited=buff_tail)



    def process(self):
        self.asr_processor.init()

        while self.void_counter < self.max_void:
            self.handle_audio_chunk()

        the_end=self.asr_processor.finish()
        self.send_result(the_end, None)