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
    def __init__(self, device, callback):
        self.local_audio_input = LocalAudioInput(device=device)
        self.local_audio_input.set_audio_callback(callback)
        self.local_audio_input.start()

    def receive_audio(self):
        # Локальный источник аудио может использовать callback для обработки данных
        pass

class FoxyProcessor:
    def __init__(self, audio_source, mqtt_handler=None, asr_processor=None, minimal_chunk=None, 
                 language="en", tcp_echo=True, callback=None):
        self.audio_source = audio_source
        self.mqtt_handler = mqtt_handler
        self.asr_processor = asr_processor
        self.minimal_chunk = minimal_chunk
        self.language = language
        self.tcp_echo = tcp_echo
        self.callback = callback
        self.void_counter = 0
        self.max_void = 5
        self.is_first = True
        self.audio_buffer = deque(maxlen=10240)
        self.last_end = None

    def process_audio_chunk(self, audio_chunk):
        if not audio_chunk:
            return
        logger.info(f"Received audio chunk of length: {len(audio_chunk)}")
        self.handle_audio_chunk(audio_chunk)
    
    def receive_audio_chunk(self):
        out = []
        min_limit = self.minimal_chunk * SAMPLING_RATE
        while sum(len(x) for x in out) < min_limit:
            raw_bytes = self.audio_source.receive_audio()
            if not raw_bytes:
                self.void_counter += 1
                break
            self.void_counter = 0
            with sf.SoundFile(io.BytesIO(raw_bytes), channels=1, endian="LITTLE", samplerate=SAMPLING_RATE,
                              subtype="PCM_16", format="RAW") as sfile:
                audio, _ = librosa.load(sfile, sr=SAMPLING_RATE, dtype=np.float32)
                out.append(audio)
        return np.concatenate(out) if out else None
    
    
    def trim_text(self, buffer, text, punctuation_end=None, punctuation_mid=None):
        punctuation_end = punctuation_end or [".", "!", "?"]
        punctuation_mid = punctuation_mid or [",", ":"]
        
        last_end = max((text.rfind(p) for p in punctuation_end), default=-1)
        last_mid = max((text.rfind(p) for p in punctuation_mid), default=-1)
        last_punc = last_end if last_end != -1 else last_mid
        
        return deque(islice(buffer, last_punc + 1)) if last_punc != -1 else buffer

    def send_result(self, outcome, buff_tail):
        if outcome[0] is None:
            return False
        beg, end = outcome[0] * 1000, outcome[1] * 1000
        beg = max(beg, self.last_end or 0)
        self.last_end = end
        text_commit = outcome[2]
        msg = f"[ {beg:.0f}, {end:.0f}, {end - beg:.0f} ] {text_commit}"
        if self.callback:
            self.callback(text_commit + " ")
        if self.tcp_echo and isinstance(self.audio_source, TCPAudioSource):
            send_one_line_tcp(self.audio_source.conn, msg)
        return True
    
    def handle_audio_chunk(self, audio_chunk):
        if audio_chunk is None or len(audio_chunk) == 0:  # Проверяем, что массив не пустой
            return
        
        self.asr_processor.insert_audio_chunk(audio_chunk)
        outcome = self.asr_processor.process_iter()
        if outcome:
            buff_tail = self.asr_processor.get_tail()
            self.audio_buffer = self.trim_text(self.audio_buffer, buff_tail)
            self.send_result(outcome, buff_tail)

    def process(self):
        self.asr_processor.init()
        send_error = False
        while not send_error and self.void_counter < self.max_void:
            audio_chunk = self.receive_audio_chunk()
            if audio_chunk is None or len(audio_chunk) == 0:  # Проверяем, что массив не пустой
                continue
            logger.info(f"Received audio chunk of length: {len(audio_chunk)}")
            self.handle_audio_chunk(audio_chunk)
        final_text = self.asr_processor.finish()