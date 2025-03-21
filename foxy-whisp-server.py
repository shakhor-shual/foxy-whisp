 #!/usr/bin/env python3
import sys
import argparse
import os
import logging
import numpy as np
import time
import select
import io
import soundfile #as sf
import librosa
import socket
import line_packet
import psutil
from functools import lru_cache
import math
from typing import List, Tuple, Optional, Any, IO
from abc import ABC, abstractmethod
import subprocess
import importlib
from mqtt_handler import MQTTHandler
import tkinter as tk
from tkinter import ttk
import threading


# Настройка логгера
logger = logging.getLogger(__name__)

# Константы
SAMPLING_RATE = 16000
PACKET_SIZE = 2 * SAMPLING_RATE * 5 * 60  # 5 minutes
MOSES = False
WHISPER_LANG_CODES = "af,am,ar,as,az,ba,be,bg,bn,bo,br,bs,ca,cs,cy,da,de,el,en,es,et,eu,fa,fi,fo,fr,gl,gu,ha,haw,he,hi,hr,ht,hu,hy,id,is,it,ja,jw,ka,kk,km,kn,ko,la,lb,ln,lo,lt,lv,mg,mi,mk,ml,mn,mr,ms,mt,my,ne,nl,nn,no,oc,pa,pl,ps,pt,ro,ru,sa,sd,si,sk,sl,sn,so,sq,sr,su,sv,sw,ta,te,tg,th,tk,tl,tr,tt,uk,ur,uz,vi,yi,yo,zh".split(",")
MAX_BUFFER_SECONDS = 15
MAX_PROMPT_SIZE = 180  # Максимальная длина подсказки для Whisper
MAX_INCOMPLETE_SIZE =200 # Максимальная длина непотвержденного фрагмента
MAX_TAIL_SIZE = 500  # Максимальная длина хвоста текста

CONNECTION_SELECT_DELAY = 0.1  # Delay for each select call in seconds
CONNECTION_TOPIC="foxy-whisp"

logger = logging.getLogger(__name__)

@lru_cache(10**6)
def load_audio(fname):
    a, _ = librosa.load(fname, sr=16000, dtype=np.float32)
    return a

def load_audio_chunk(fname, beg, end):
    audio = load_audio(fname)
    beg_s = int(beg*16000)
    end_s = int(end*16000)
    return audio[beg_s:end_s]


# Функция для настройки логирования
def set_logging(args, logger, other=""):
    """Настраивает уровень логирования."""
    logging.basicConfig(format='%(levelname)s\t%(message)s')
    logger.setLevel(args.log_level)
    logging.getLogger("whisper_online" + other).setLevel(args.log_level)


# Функция для добавления общих аргументов
def add_shared_args(parser):
    """Добавляет общие аргументы для настройки ASR."""
    parser.add_argument('--model', type=str, default='large-v3-turbo', choices=[
        "tiny.en", "tiny", "base.en", "base", "small.en", "small", "medium.en", "medium", "large-v1", "large-v2", "large-v3", "large-v3-turbo"
    ], help="Модель Whisper для использования (по умолчанию: large-v2).")
    parser.add_argument('--lan', '--language', type=str, default='auto', help="Язык исходного аудио (например, 'ru', 'en').")
    parser.add_argument('--min-chunk-size', type=float, default=1.0, help="Минимальный размер аудиофрагмента в секундах.")
    parser.add_argument('--vad', action="store_true", default=False, help="Использовать VAD (Voice Activity Detection).")
    parser.add_argument('--vac', action="store_true", default=False, help="Использовать VAC (Voice Activity Controller).")
    parser.add_argument('--buffer-trimming', type=str, default="segment", choices=["sentence", "segment"], help="Стратегия обрезки буфера.")
    parser.add_argument('--task', type=str, default='transcribe', choices=["transcribe", "translate"], help="Задача: транскрибирование или перевод.")
    parser.add_argument('--model-cache-dir', type=str, default=None, help="Директория для кэширования моделей.")
    parser.add_argument('--model-dir', type=str, default=None, help="Директория с моделью Whisper.")
    parser.add_argument('--backend', type=str, default="faster-whisper", choices=["faster-whisper", "whisper_timestamped", "openai-api"], help="Бэкенд для Whisper.")
    parser.add_argument('--vac-chunk-size', type=float, default=0.04, help="Размер фрагмента для VAC в секундах.")
    parser.add_argument('--buffer-trimming-sec', type=float, default=15, help="Порог обрезки буфера в секундах.")
    parser.add_argument("-l", "--log-level", dest="log_level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='DEBUG', help="Уровень логирования.")
    parser.add_argument("--gui", action="store_true", help="Запустить сервер с графическим интерфейсом.")

# Фуекция устанавливает пакет с помощью pip, если он не установлен.
def install_package(package_name: str):
    """Устанавливает пакет с помощью pip, если он не установлен."""
    try:
        importlib.import_module(package_name)
    except ImportError:
        logger.info(f"Установка пакета {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

################################################################################################
class OnlineASRProcessor:
    def __init__(self, asr, tokenizer=None, buffer_trimming=("segment", 10), logfile=sys.stderr):
        """asr: WhisperASR object
        tokenizer: sentence tokenizer object for the target language. Must have a method *split* that behaves like the one of MosesTokenizer. It can be None, if "segment" buffer trimming option is used, then tokenizer is not used at all.
        ("segment", 15)
        buffer_trimming: a pair of (option, seconds), where option is either "sentence" or "segment", and seconds is a number. Buffer is trimmed if it is longer than "seconds" threshold. Default is the most recommended option.
        logfile: where to store the log. 
        """
        self.asr = asr
        self.tokenizer = tokenizer
        self.logfile = logfile
        self.tail=""

        self.init()
        self.buffer_trimming_way, self.buffer_trimming_sec = buffer_trimming

    def init(self, offset=None):
        """run this when starting or restarting processing"""
        self.audio_buffer = np.array([],dtype=np.float32)
        self.transcript_buffer = HypothesisBuffer(logfile=self.logfile)
        self.buffer_time_offset = 0
        if offset is not None:
            self.buffer_time_offset = offset
        self.transcript_buffer.last_commited_time = self.buffer_time_offset
        self.commited = []

    def insert_audio_chunk(self, audio):
        """Добавляет новый фрагмент аудио в буфер."""
        self.audio_buffer = np.append(self.audio_buffer, audio)

    #---- NEW VERSION ----
    def prompt(self, max_prompt_size=None):
        """Возвращает кортеж: (prompt, context).
        - prompt: Последние max_prompt_size символов текста, выходящего за пределы аудиобуфера.
        - context: Текст внутри аудиобуфера (для отладки и логирования).
        """
        if max_prompt_size is None:
            max_prompt_size = MAX_PROMPT_SIZE

        # Проверка на пустой список
        if not self.commited:
            return "", ""

        # Находим индекс, где текст переходит в область аудиобуфера
        k = next(
            (i for i in range(len(self.commited) - 1, -1, -1)
            if self.commited[i][1] <= self.buffer_time_offset),
            0
        )

        # Разделяем текст на prompt и context
        prompt_texts = [t for _, _, t in self.commited[:k]] if k > 0 else []
        context_texts = [t for _, _, t in self.commited[k:]]

        # Если prompt_texts пуст, возвращаем пустой prompt
        if not prompt_texts:
            return "", self.asr.sep.join(context_texts)

        # Формируем prompt из последних max_prompt_size символов
        prompt = []
        current_length = 0
        for text in reversed(prompt_texts):
            if current_length + len(text) + 1 > max_prompt_size:
                break
            prompt.append(text)
            current_length += len(text) + 1

        # Собираем результат
        prompt_str = self.asr.sep.join(reversed(prompt)) if prompt else ""
        context_str = self.asr.sep.join(context_texts)

        return prompt_str, context_str


    #---- NEW VERSION ----
    def process_iter(self):
        """Обрабатывает текущий аудиобуфер.
        Возвращает: кортеж (начало, конец, "текст") или (None, None, "").
        Непустой текст — это подтвержденная частичная транскрипция.
        """
        # Формируем промпт и логируем его
        prompt, non_prompt = self.prompt()
        logger.debug(f"PROMPT (truncated to {len(prompt)} characters): {prompt}")
        logger.debug(f"CONTEXT: {non_prompt}")
        logger.debug(f"Transcribing {len(self.audio_buffer)/SAMPLING_RATE:.2f} seconds from {self.buffer_time_offset:.2f}")

        # Транскрибируем аудио
        res = self.asr.transcribe(self.audio_buffer, init_prompt=prompt)
        tsw = self.asr.ts_words(res)  # Преобразуем в список слов с временными метками

        # Вставляем слова в буфер и извлекаем подтвержденные данные
        self.transcript_buffer.insert(tsw, self.buffer_time_offset)
        committed = self.transcript_buffer.flush()
        self.commited.extend(committed)

        # Логируем завершенные и незавершенные части
        completed = self.to_flush(committed)
        the_rest = self.to_flush(self.transcript_buffer.complete())
        logger.debug(f">>>>COMPLETE NOW: {completed}")
        logger.debug(f"INCOMPLETE: {the_rest}")

        # Формируем хвост текста и ограничиваем его длину
        self.tail = f"{{{completed[-1]}}}{the_rest[-1]}" if completed and the_rest else ""
        if len(self.tail) > MAX_TAIL_SIZE:
            self.tail = self.tail[:MAX_TAIL_SIZE]

        # Обрезка буфера по предложениям
        if committed and self.buffer_trimming_way == "sentence":
            buffer_duration = len(self.audio_buffer) / SAMPLING_RATE
            if buffer_duration > self.buffer_trimming_sec:
                self.chunk_completed_sentence()

        # Обрезка буфера по сегментам
        buffer_duration = len(self.audio_buffer) / SAMPLING_RATE
        trim_threshold = self.buffer_trimming_sec if self.buffer_trimming_way == "segment" else 30
        if buffer_duration > trim_threshold:
            self.chunk_completed_segment(res)
            logger.debug("!!!! STANDARD chunking !!!!")

        # Принудительная финализация, если незавершенный текст слишком длинный
        if the_rest and len(the_rest[-1]) > MAX_INCOMPLETE_SIZE:
            logger.warning("!!!! FORCED FINALIZE !!!!")
            return None

        # Логируем текущую длину буфера и возвращаем результат
        logger.debug(f"len of buffer now: {buffer_duration:.2f}")
        return self.to_flush(committed)

    #------------------
    def get_tail(self):
        return self.tail

    #--- NEW VERSION---
    def chunk_completed_sentence(self):
        if self._is_committed_empty():
            return

        logger.debug("Committed words: %s", self.commited)
        sents = self.words_to_sentences(self.commited)

        for sent in sents:
            logger.debug("\t\tSENT: %s", sent)

        if len(sents) < 2:
            return

        sents = sents[-2:]
        chunk_at = sents[0][1]
        logger.debug("--- Sentence chunked at %.2f", chunk_at)
        self.chunk_at(chunk_at)

    #--- NEW VERSION---
    def chunk_completed_segment(self, res):
        if self._is_committed_empty():
            return

        ends = self.asr.segments_end_ts(res)
        last_committed_time = self.commited[-1][1]

        if len(ends) < 2:
            logger.debug("--- Not enough segments to chunk")
            return

        e = ends[-2] + self.buffer_time_offset

        while len(ends) > 2 and e > last_committed_time:
            ends.pop(-1)
            e = ends[-2] + self.buffer_time_offset

        if e <= last_committed_time:
            logger.debug("--- Segment chunked at %.2f", e)
            self.chunk_at(e)
        else:
            logger.debug("--- Last segment not within committed area")

    #--- NEW ADDED---
    def _is_committed_empty(self):
        return not self.commited

    #----------------
    def chunk_at(self, time):
        """trims the hypothesis and audio buffer at "time """
        self.transcript_buffer.pop_commited(time)
        cut_seconds = time - self.buffer_time_offset
        self.audio_buffer = self.audio_buffer[int(cut_seconds*SAMPLING_RATE):]
        self.buffer_time_offset = time

    
    #--- NEW VERSION---
    def words_to_sentences(self, words):
        text = " ".join(word[2] for word in words)
        sentences = self.tokenizer.split(text) if MOSES else self.tokenizer.tokenize(text)
        result, word_list = [], list(words)

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            start_time, end_time, remaining_sentence = None, None, sentence
            while word_list and remaining_sentence:
                word_start, word_end, word_text = word_list.pop(0)
                word_text = word_text.strip()

                if start_time is None and remaining_sentence.startswith(word_text):
                    start_time = word_start

                if remaining_sentence == word_text:
                    end_time = word_end
                    result.append((start_time, end_time, sentence))
                    break

                remaining_sentence = remaining_sentence[len(word_text):].strip()

        return result

    #--- NEW VERSION---
    def finish(self):
        """Завершает обработку, возвращая оставшийся неподтвержденный текст.
        Возвращает: кортеж (начало, конец, "текст") или (None, None, ""), если текст пуст.
        """
        buffer_duration = len(self.audio_buffer) / SAMPLING_RATE
        logger.debug("Buffer duration before final flush: %.2f seconds", buffer_duration)

        # Получаем оставшийся неподтвержденный текст
        remaining_text = self.transcript_buffer.complete()
        flushed_result = self.to_flush(remaining_text)

        logger.debug("Final non-committed text: %s", flushed_result)

        # Обновляем смещение буфера
        self.buffer_time_offset += buffer_duration

        return flushed_result

    #--- NEW VERSION---
    def to_flush(self, sentences, sep=None, offset=0):
        """Объединяет временно меткированные слова или предложения в одну строку.
        
        Аргументы:
            sentences: Список кортежей [(начало, конец, "текст"), ...] или пустой список.
            sep: Разделитель между предложениями (по умолчанию используется self.asr.sep).
            offset: Смещение временных меток.

        Возвращает:
            Кортеж (начало, конец, "текст") или (None, None, ""), если список пуст.
        """
        if sep is None:
            sep = self.asr.sep

        # Объединяем текст
        text = sep.join(sentence[2] for sentence in sentences)

        # Обрабатываем пустой список
        if not sentences:
            return (None, None, "")

        # Вычисляем начало и конец
        start_time = offset + sentences[0][0]
        end_time = offset + sentences[-1][1]

        return (start_time, end_time, text)


##########################################################################################
class VACOnlineASRProcessor(OnlineASRProcessor):
    '''Wraps OnlineASRProcessor with VAC (Voice Activity Controller). '''
    def __init__(self, online_chunk_size, *a, **kw):
        self.online_chunk_size = online_chunk_size

        self.online = OnlineASRProcessor(*a, **kw)

        # VAC:
        import torch
        model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad:v4.0',
            model='silero_vad'
        )
        from silero_vad import FixedVADIterator
        #FixedVADIterator
        self.vac = FixedVADIterator(model)  # we use all the default options: 500ms silence, etc.  

        self.logfile = self.online.logfile
        self.init()


    def init(self):
        self.online.init()
        self.vac.reset_states()
        self.current_online_chunk_buffer_size = 0

        self.is_currently_final = False

        self.status = None  # or "voice" or "nonvoice"
        self.audio_buffer = np.array([],dtype=np.float32)
        self.buffer_offset = 0  # in frames


    def clear_buffer(self):
        self.buffer_offset += len(self.audio_buffer)
        self.audio_buffer = np.array([],dtype=np.float32)


    def insert_audio_chunk(self, audio):
        res = self.vac(audio)
        self.audio_buffer = np.append(self.audio_buffer, audio)

        if res is not None:
            frame = list(res.values())[0]
            if 'start' in res and 'end' not in res:
                self.status = 'voice'
                send_audio = self.audio_buffer[frame-self.buffer_offset:]
                self.online.init(offset=frame/SAMPLING_RATE)
                self.online.insert_audio_chunk(send_audio)
                self.current_online_chunk_buffer_size += len(send_audio)
                self.clear_buffer()
            elif 'end' in res and 'start' not in res:
                self.status = 'nonvoice'
                send_audio = self.audio_buffer[:frame-self.buffer_offset]
                self.online.insert_audio_chunk(send_audio)
                self.current_online_chunk_buffer_size += len(send_audio)
                self.is_currently_final = True
                self.clear_buffer()
            else:
                # It doesn't happen in the current code.
                raise NotImplemented("both start and end of voice in one chunk!!!")
        else:
            if self.status == 'voice':
                self.online.insert_audio_chunk(self.audio_buffer)
                self.current_online_chunk_buffer_size += len(self.audio_buffer)
                self.clear_buffer()
            else:
                # We keep 1 second because VAD may later find start of voice in it.
                # But we trim it to prevent OOM. 
                self.buffer_offset += max(0,len(self.audio_buffer)-SAMPLING_RATE)
                self.audio_buffer = self.audio_buffer[-SAMPLING_RATE:]


    def get_tail(self):
        return self.online.get_tail()


    def process_iter(self):
        if self.is_currently_final:
            return self.finish()
        elif self.current_online_chunk_buffer_size > SAMPLING_RATE*self.online_chunk_size:
            self.current_online_chunk_buffer_size = 0
            ret = self.online.process_iter()
            return ret
        else:
            print("no online update, only VAD", self.status, file=self.logfile)
            return (None, None, "")


    def finish(self):
        ret = self.online.finish()
        self.current_online_chunk_buffer_size = 0
        self.is_currently_final = False
        return ret


######################### NEW VERSION ################################################
class ASRBase(ABC):
    sep: str = " "

    def __init__(
        self,
        lan: str,
        modelsize: Optional[str] = None,
        cache_dir: Optional[str] = None,
        model_dir: Optional[str] = None,
        logfile: IO = sys.stderr,
    ):
        self._logfile = logfile
        self._transcribe_kargs: dict = {}
        self._original_language: Optional[str] = None if lan == "auto" else lan
        logger.info(f"Initializing ASR for language: {self._original_language}")
        try:
            self._model = self.load_model(modelsize, cache_dir, model_dir)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    @abstractmethod
    def load_model(self, modelsize: Optional[str], cache_dir: Optional[str]) -> Any:
        pass

    def transcribe(self, audio: np.ndarray, init_prompt: str = "") -> Any:
        try:
            result = self._transcribe_impl(audio, init_prompt)
            logger.info("Transcription completed successfully")
            return result
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise

    @abstractmethod
    def _transcribe_impl(self, audio: np.ndarray, init_prompt: str) -> Any:
        pass

    def use_vad(self) -> None:
        """Включает VAD (Voice Activity Detection)."""
        logger.info("Enabling VAD")
        self._transcribe_kargs["vad"] = True

    def set_translate_task(self) -> None:
        """Устанавливает задачу перевода."""
        logger.info("Setting task to 'translate'")
        self._transcribe_kargs["task"] = "translate"

######################### NEW VERSION ################################################
class WhisperTimestampedASR(ASRBase):
    sep = " "

    def load_model(self, modelsize: Optional[str], cache_dir: Optional[str], model_dir: Optional[str] = None) -> Any:
        install_package("whisper")
        install_package("whisper-timestamped")

        import whisper
        import whisper_timestamped
        from whisper_timestamped import transcribe_timestamped

        self.transcribe_timestamped = transcribe_timestamped

        if model_dir is not None:
            logger.debug("Ignoring model_dir, not implemented")
        return whisper.load_model(modelsize, download_root=cache_dir)

    def _transcribe_impl(self, audio: np.ndarray, init_prompt: str) -> Any:
        result = self.transcribe_timestamped(
            self._model,
            audio,
            language=self._original_language,
            initial_prompt=init_prompt,
            verbose=None,
            condition_on_previous_text=True,
            **self._transcribe_kargs
        )
        return result

    def ts_words(self, r) -> List[Tuple[float, float, str]]:
        return [(w["start"], w["end"], w["text"]) for s in r["segments"] for w in s["words"]]

    def segments_end_ts(self, res) -> List[float]:
        return [s["end"] for s in res["segments"]]

######################### NEW VERSION ################################################
class FasterWhisperASR(ASRBase):
    sep = ""

    def load_model(self, modelsize: Optional[str], cache_dir: Optional[str], model_dir: Optional[str] = None) -> Any:
        install_package("faster-whisper")

        from faster_whisper import WhisperModel

        if model_dir is not None:
            logger.debug(f"Loading whisper model from model_dir {model_dir}.")
            model_size_or_path = model_dir
        elif modelsize is not None:
            model_size_or_path = modelsize
        else:
            raise ValueError("modelsize or model_dir parameter must be set")

        model = WhisperModel(model_size_or_path, device="cuda", compute_type="int8_float32", download_root=cache_dir)
        return model

    def _transcribe_impl(self, audio: np.ndarray, init_prompt: str) -> Any:
        segments, info = self._model.transcribe(
            audio,
            language=self._original_language,
            initial_prompt=init_prompt,
            beam_size=5,
            word_timestamps=True,
            condition_on_previous_text=True,
            **self._transcribe_kargs
        )
        return list(segments)

    def ts_words(self, segments) -> List[Tuple[float, float, str]]:
        return [(word.start, word.end, word.word) for segment in segments for word in segment.words]

    def segments_end_ts(self, res) -> List[float]:
        return [s.end for s in res]

######################### NEW VERSION ################################################
class OpenaiApiASR(ASRBase):
    def __init__(self, lan: Optional[str] = None, temperature: int = 0, logfile: IO = sys.stderr):
        super().__init__(lan, logfile=logfile)
        self.modelname = "whisper-1"
        self.temperature = temperature
        self.task = "transcribe"
        self.use_vad_opt = False
        self.transcribed_seconds = 0

    def load_model(self, *args, **kwargs) -> Any:
        install_package("openai")

        from openai import OpenAI

        self.client = OpenAI()

    def _transcribe_impl(self, audio_data: np.ndarray, prompt: Optional[str] = None) -> Any:
        buffer = io.BytesIO()
        buffer.name = "temp.wav"
        soundfile.write(buffer, audio_data, samplerate=SAMPLING_RATE, format='WAV', subtype='PCM_16')
        buffer.seek(0)

        self.transcribed_seconds += math.ceil(len(audio_data) / SAMPLING_RATE)

        params = {
            "model": self.modelname,
            "file": buffer,
            "response_format": "verbose_json",
            "temperature": self.temperature,
            "timestamp_granularities": ["word", "segment"]
        }
        if self.task != "translate" and self._original_language:
            params["language"] = self._original_language
        if prompt:
            params["prompt"] = prompt

        proc = self.client.audio.translations if self.task == "translate" else self.client.audio.transcriptions
        transcript = proc.create(**params)
        logger.debug(f"OpenAI API processed accumulated {self.transcribed_seconds} seconds")
        return transcript

    def use_vad(self) -> None:
        self.use_vad_opt = True


############################################################################
class HypothesisBuffer:
    def __init__(self, logfile=sys.stderr):
        """Инициализирует буфер гипотез.

        Args:
            logfile: Файл для логирования (по умолчанию sys.stderr).
        """
        self.commited_in_buffer = []  # Подтвержденные части текста
        self.buffer = []  # Неподтвержденные части текста
        self.new = []  # Новые части текста для обработки

        self.last_commited_time = 0  # Время последнего подтвержденного слова
        self.last_commited_word = None  # Последнее подтвержденное слово

        self.logfile = logfile

    def insert(self, new, offset):
        """Вставляет новые данные в буфер.

        Args:
            new: Список кортежей (начало, конец, текст).
            offset: Смещение временных меток.
        """
        # Применяем смещение к новым данным
        new = [(a + offset, b + offset, t) for a, b, t in new]

        # Фильтруем новые данные, оставляя только те, которые идут после последнего подтвержденного времени
        self.new = [(a, b, t) for a, b, t in new if a > self.last_commited_time - 0.1]

        if not self.new:
            return

        # Проверяем, есть ли пересечение с последним подтвержденным словом
        a, b, t = self.new[0]
        if abs(a - self.last_commited_time) < 1 and self.commited_in_buffer:
            self._remove_duplicate_ngrams()

    def _remove_duplicate_ngrams(self):
        """Удаляет n-граммы, которые уже есть в подтвержденных данных."""
        cn = len(self.commited_in_buffer)
        nn = len(self.new)
        for i in range(1, min(min(cn, nn), 5) + 1):  # Проверяем n-граммы длиной от 1 до 5
            # Сравниваем n-граммы из подтвержденных и новых данных
            committed_ngram = " ".join([self.commited_in_buffer[-j][2] for j in range(1, i + 1)][::-1])
            new_ngram = " ".join(self.new[j - 1][2] for j in range(1, i + 1))

            if committed_ngram == new_ngram:
                # Удаляем дубликаты
                words = [repr(self.new.pop(0)) for _ in range(i)]
                logger.debug(f"Removing last {i} words: {' '.join(words)}")
                break

    def flush(self):
        """Извлекает подтвержденные части текста.

        Returns:
            Список кортежей (начало, конец, текст).
        """
        commit = []
        while self.new:
            na, nb, nt = self.new[0]

            if not self.buffer:
                break

            if nt == self.buffer[0][2]:
                commit.append((na, nb, nt))
                self.last_commited_word = nt
                self.last_commited_time = nb
                self.buffer.pop(0)
                self.new.pop(0)
            else:
                break

        self.buffer = self.new
        self.new = []
        self.commited_in_buffer.extend(commit)
        return commit

    def pop_commited(self, time):
        """Удаляет подтвержденные части текста, которые завершились до указанного времени.

        Args:
            time: Временная метка.
        """
        while self.commited_in_buffer and self.commited_in_buffer[0][1] <= time:
            self.commited_in_buffer.pop(0)

    def complete(self):
        """Возвращает неподтвержденные части текста.

        Returns:
            Список кортежей (начало, конец, текст).
        """
        return self.buffer
    
#######################################################################################################
class Connection:
    '''It wraps conn object for non-blocking audio receiving.'''
    def __init__(self, conn, mqtt_handler, tcp_echo =True, timeout=1):
        """
        :param conn: socket connection object
        :param timeout: timeout in seconds for data inactivity
        """
        self.tcp_echo=tcp_echo
        self.mqtt_handler=mqtt_handler
        self.conn = conn
        self.last_line = ""
        self.timeout = timeout
        self.conn.setblocking(False)  # Set the socket to non-blocking mode
        self.max_cycles = int(timeout / CONNECTION_SELECT_DELAY)  # Calculate max cycles based on timeout

    def send(self, line):
        '''It doesn't send the same line twice, to avoid duplicate transmissions.'''
        if line == self.last_line:
            return
        
        payload= line.split("]", 1)[1].strip() if line.startswith("[") else None
        if payload and self.mqtt_handler.connected:
            self.mqtt_handler.publish_message(CONNECTION_TOPIC, payload)
        if self.tcp_echo:
            line_packet.send_one_line(self.conn, line)
        self.last_line = line

    def receive_lines(self):
        '''Receive lines of text data.'''
        return line_packet.receive_lines(self.conn)

    def non_blocking_receive_audio(self):
        """
        Receive audio data in a non-blocking manner.
        Wait for data until timeout or socket closure.
        Return data immediately if available, or None on timeout.
        """
        cycles = 0

        while cycles < self.max_cycles:
            # Check if the socket has data ready to read
            ready = select.select([self.conn], [], [], CONNECTION_SELECT_DELAY)  # Short timeout for polling
            if ready[0]:
                try:
                    raw_bytes = self.conn.recv(PACKET_SIZE)
                    if raw_bytes != b"":
                        return raw_bytes  # Return immediately if data is received
                except BlockingIOError:
                    pass
                except ConnectionResetError:
                    self.socket_closed = True
                    return None  # Connection was reset
            
            cycles += 1 # Increment the cycle count if no data is received
        return None  # Return None after timeout

##################################################################
class ServerProcessor:
    def __init__(self, connection,  online_asr_proc, min_chunk, language="en"):
        """Инициализирует процессор для обработки аудиопотока от клиента.

        Args:
            connection: Объект соединения с клиентом.
            online_asr_proc: Процессор онлайн-транскрипции.
            min_chunk: Минимальный размер аудиофрагмента.
            language: Язык транскрипции (по умолчанию "en").
        """
        self.connection = connection
        self.asr_proc = online_asr_proc
        self.min_chunk = min_chunk
        self.language = language

        self.audio_buffer = np.array([], dtype=np.float32)  # Буфер для аудиоданных
        self.last_end = None  # Время окончания последнего сегмента
        self.is_first = True  # Флаг первого фрагмента
        self.void_counter = 0  # Счетчик пустых фрагментов
        self.max_void = 5  # Максимальное количество пустых фрагментов перед завершением

    def receive_audio_chunk(self):
        """Получает и обрабатывает аудиофрагмент.

        Returns:
            np.ndarray: Аудиоданные или None, если данные отсутствуют.
        """
        out = []
        min_limit = self.min_chunk * SAMPLING_RATE

        while sum(len(x) for x in out) < min_limit:
            raw_bytes = self.connection.non_blocking_receive_audio()
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

    def has_punctuation(self, text):
        """Проверяет, содержит ли текст знаки препинания, завершающие предложение.

        Args:
            text: Текст для проверки.

        Returns:
            bool: True, если текст содержит знаки препинания, иначе False.
        """
        if self.language in ["en", "ru"]:
            return any(punc in text for punc in [".", "!", "?"])
        return "." in text

    def trim_at_sentence_boundary(self, buffer, text):
        """Обрезает буфер на границе предложения.

        Args:
            buffer: Аудиобуфер.
            text: Текст для анализа.

        Returns:
            np.ndarray: Обрезанный буфер.
        """
        if buffer is None:
            logger.warning("Buffer is None, returning empty buffer")
            return np.array([], dtype=np.float32)

        if self.has_punctuation(text):
            last_punc_end_pos = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
            if last_punc_end_pos != -1:
                logger.debug(f"Trimming buffer at last punctuation position: {last_punc_end_pos}")
                return buffer[:last_punc_end_pos + 1]

        return buffer

    def trim_at_clause_boundary(self, buffer, text):
        """Обрезает буфер на границе предложения или клаузы.

        Args:
            buffer: Аудиобуфер.
            text: Текст для анализа.

        Returns:
            np.ndarray: Обрезанный буфер.
        """
        if buffer is None:
            logger.warning("Buffer is None, returning empty buffer")
            return np.array([], dtype=np.float32)

        last_punc_mid_pos = max(text.rfind(","), text.rfind(":"))
        if last_punc_mid_pos != -1:
            logger.debug(f"Trimming buffer at last punctuation position: {last_punc_mid_pos}")
            return buffer[:last_punc_mid_pos + 1]

        return buffer

    def format_output_transcript(self, o, t):
        """Форматирует результат транскрипции.

        Args:
            o: Кортеж (начало, конец, текст).
            t: Текст для отправки.

        Returns:
            str: Отформатированный результат или None.
        """
        if o[0] is not None:
            beg, end = o[0] * 1000, o[1] * 1000
            if self.last_end is not None:
                beg = max(beg, self.last_end)

            self.last_end = end
            text_commit = o[2]
            return f"[ {beg:.0f}, {end:.0f}, {end - beg:.0f} ] {text_commit}"

        logger.debug("No text in this segment")
        return None

    def send_result(self, o, t):
        """Отправляет результат транскрипции клиенту.

        Args:
            o: Кортеж (начало, конец, текст).
            t: Текст для отправки.

        Returns:
            bool: True, если отправка успешна, иначе False.
        """
        msg = self.format_output_transcript(o, t)
        try:
            if msg is not None:
                self.connection.send(msg)
            if t is not None:
                self.connection.send(t)
            return True
        except BrokenPipeError:
            logger.info("Broken pipe -- connection closed?")
            return False

    def process(self):
        """Обрабатывает аудиопоток от клиента с обрезкой по знакам препинания."""
        self.asr_proc.init()
        send_error = False
        outcome = None

        while not send_error and self.void_counter < self.max_void:
            audio_chunk = self.receive_audio_chunk()
            if audio_chunk is None:
                continue

            logger.info(f"Received audio chunk of length: {len(audio_chunk)}")
            self.asr_proc.insert_audio_chunk(audio_chunk)

            if len(audio_chunk) <= 8192 and self.void_counter > 1:
                break

            outcome = self.asr_proc.process_iter()
            if outcome is None:
                break

            buff_tail = self.asr_proc.get_tail()
            self.audio_buffer = (
                self.trim_at_sentence_boundary(self.audio_buffer, buff_tail)
                if self.has_punctuation(buff_tail)
                else self.trim_at_clause_boundary(self.audio_buffer, buff_tail)
            )

            send_error = not self.send_result(outcome, buff_tail)

        logger.info("ASR process completed with send_error: {send_error}")
        final_text = self.asr_proc.finish()
        logger.info(f"Final transcription: {final_text}")

        if final_text and final_text[-1]:
            return self.send_result(final_text, "")
        return self.send_result((None, None, ''), "")

##########################
def get_port_status(port):
    listening = False
    has_connections = False

    for conn in psutil.net_connections(kind='inet'):
        if conn.laddr.port == port:
            if conn.status == "LISTEN":
                listening = True
            elif conn.status == "ESTABLISHED":
                has_connections = True

    if listening and has_connections:
        return 1  # Port is listening and has connections
    elif listening:
        return 2  # Port is listening but has no connections
    else:
        return 0  # Port is not listening
    
#########################
def create_tokenizer(lan):
    """returns an object that has split function that works like the one of MosesTokenizer"""
    assert lan in WHISPER_LANG_CODES, "language must be Whisper's supported lang code: " + " ".join(WHISPER_LANG_CODES)

    if lan == "uk":
        import tokenize_uk
        class UkrainianTokenizer:
            def split(self, text):
                return tokenize_uk.tokenize_sents(text)
        return UkrainianTokenizer()

    # supported by fast-mosestokenizer
    if lan in "as bn ca cs de el en es et fi fr ga gu hi hu is it kn lt lv ml mni mr nl or pa pl pt ro ru sk sl sv ta te yue zh".split():
        from mosestokenizer import MosesTokenizer
        global MOSES
        MOSES=True
        return MosesTokenizer(lan)

    # the following languages are in Whisper, but not in wtpsplit:
    if lan in "as ba bo br bs fo haw hr ht jw lb ln lo mi nn oc sa sd sn so su sw tk tl tt".split():
        logger.debug(f"{lan} code is not supported by wtpsplit. Going to use None lang_code option.")
        lan = None

    from wtpsplit import WtP
    # downloads the model from huggingface on the first use
    wtp = WtP("wtp-canine-s-12l-no-adapters")
    class WtPtok:
        def split(self, sent):
            return wtp.split(sent, lang_code=lan)
    return WtPtok()


#####################################
def asr_factory(args, logfile=sys.stderr):
    """ Creates and configures an ASR and ASR Online instance based on the specified backend and arguments. """
    backend = args.backend
    if backend == "openai-api":
        logger.debug("Using OpenAI API.")
        asr = OpenaiApiASR(lan=args.lan)
    else:
        if backend == "faster-whisper":
            asr_cls = FasterWhisperASR
        else:
            asr_cls = WhisperTimestampedASR

        # Only for FasterWhisperASR and WhisperTimestampedASR
        size = args.model
        t = time.time()
        logger.info(f"Loading Whisper {size} model for {args.lan}...")
        asr = asr_cls(modelsize=size, lan=args.lan, cache_dir=args.model_cache_dir, model_dir=args.model_dir)
        e = time.time()
        logger.info(f"done. It took {round(e-t,2)} seconds.")

    # Apply common configurations
    if getattr(args, 'vad', False):  # Checks if VAD argument is present and True
        logger.info("Setting VAD filter")
        asr.use_vad()

    language = args.lan
    if args.task == "translate":
        asr.set_translate_task()
        tgt_language = "en"  # Whisper translates into English
    else:
        tgt_language = language  # Whisper transcribes in this language

    # Create the tokenizer
    if args.buffer_trimming == "sentence":
        tokenizer = create_tokenizer(tgt_language)
    else:
        tokenizer = None

    # Create the OnlineASRProcessor
    if args.vac:
        online = VACOnlineASRProcessor(args.min_chunk_size, asr,tokenizer,logfile=logfile,buffer_trimming=(args.buffer_trimming, args.buffer_trimming_sec))
    else:
        online = OnlineASRProcessor(asr,tokenizer,logfile=logfile,buffer_trimming=(args.buffer_trimming, args.buffer_trimming_sec))

    return asr, online


# Модифицируем функцию main для поддержки GUI
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default='0.0.0.0')
    parser.add_argument("--port", type=int, default=43007)
    parser.add_argument("--warmup-file", type=str, dest="warmup_file",
                        help="Path to a speech audio file to warm up Whisper.")
    add_shared_args(parser)  # Добавляем общие аргументы
    #add_gui_arg(parser)  # Добавляем параметр --gui
    args = parser.parse_args()

    if args.gui:
        # Запуск GUI
        root = tk.Tk()
        app = ServerGUI(root, parser, args)  # Передаем parser и args
        root.mainloop()
    else:
        # Запуск сервера в консольном режиме
        run_server(args)

# Функция для запуска сервера
def run_server(args, stop_event=None):
    set_logging(args, logger)

    # Проверяем, что модель не пустая
    if not args.model:
        logger.error("Модель не может быть пустой. Установлено значение по умолчанию: large-v3-turbo.")
        args.model = "large-v3-turbo"

    asr, online = asr_factory(args)
    if args.warmup_file and os.path.isfile(args.warmup_file):
        a = load_audio_chunk(args.warmup_file, 0, 1)
        asr.transcribe(a)
        logger.info("Whisper is warmed up.")
    else:
        logger.warning("Whisper is not warmed up. The first chunk processing may take longer.")

    mqtt_handler = MQTTHandler()
    mqtt_handler.connect_to_external_broker()

    if not mqtt_handler.connected:
        mqtt_handler.start_embedded_broker()

    if mqtt_handler.connected:
        mqtt_handler.publish_message(CONNECTION_TOPIC, "<foxy:started>")
    else:
        logging.error("MQTT client is not connected. Unable to publish message.")

    # Server loop
    if get_port_status(args.port) == 0:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((args.host, args.port))
            s.listen(1)
            s.settimeout(1)  # Устанавливаем таймаут для accept(), чтобы не блокировать навсегда
            logger.info('Listening on' + str((args.host, args.port)))
            
            while get_port_status(args.port) > 0:
                if stop_event and stop_event.is_set():  # Проверка на остановку
                    logger.info("Server stopping due to stop event.")
                    break

                try:
                    conn, addr = s.accept()  # Неблокирующий accept()
                    logger.info('Connected to client on {}'.format(addr))
                    connection = Connection(conn, mqtt_handler, tcp_echo=True)

                    while get_port_status(args.port) == 1:
                        if stop_event and stop_event.is_set():  # Проверка на остановку
                            logger.info("Server stopping due to stop event.")
                            break

                        proc = ServerProcessor(connection, online, args.min_chunk_size)
                        if not proc.process():
                            break
                    conn.close()
                    logger.info('Connection to client closed')
                except socket.timeout:
                    # Таймаут accept(), продолжаем цикл
                    continue
                except Exception as e:
                    logger.error(f"Error in server loop: {e}")
                    break

        logger.info('Connection closed, terminating.')
    else:
        logger.info(f'port {args.port} already IN USE, terminating.')


class ServerGUI:
    def __init__(self, root, parser, args):
        self.root = root
        self.parser = parser  # Сохраняем объект parser
        self.args = args  # Сохраняем объект args
        self.server_thread = None
        self.server_running = False
        self.stop_event = threading.Event()  # Событие для остановки сервера

        self.root.title("Whisper Server GUI")

        # Кнопка запуска/остановки сервера
        self.start_stop_button = ttk.Button(root, text="Start Server", command=self.toggle_server)
        self.start_stop_button.pack(pady=10)

        # Кнопка "Advanced" для разворачивания дополнительных параметров
        self.advanced_button = ttk.Button(root, text="Advanced", command=self.toggle_advanced)
        self.advanced_button.pack(pady=10)

        # Фрейм для дополнительных параметров
        self.advanced_frame = ttk.Frame(root)
        self.advanced_options_visible = False

        # Кнопка Apply для применения изменений
        self.apply_button = ttk.Button(self.advanced_frame, text="Apply", command=self.apply_changes, state=tk.DISABLED)
        self.apply_button.pack(pady=10)

        # Словарь для хранения виджетов и их переменных
        self.widgets = {}

        # Добавляем элементы управления для всех параметров
        self.add_parameter_controls()

        # Обработка закрытия окна
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def add_parameter_controls(self):
        """Добавляет элементы управления для всех параметров из parser."""
        for action in self.parser._actions:
            if action.dest == "help":  # Пропускаем аргумент --help
                continue

            frame = ttk.Frame(self.advanced_frame)
            frame.pack(fill=tk.X, padx=5, pady=5)

            ttk.Label(frame, text=action.help or action.dest).pack(side=tk.LEFT)

            # Определяем тип аргумента и создаем соответствующий виджет
            if action.choices:
                # Параметры с choices (выпадающий список)
                var = tk.StringVar(value=getattr(self.args, action.dest, action.default))
                combobox = ttk.Combobox(frame, textvariable=var, values=action.choices)
                combobox.pack(side=tk.RIGHT, fill=tk.X, expand=True)
                combobox.bind("<<ComboboxSelected>>", self.on_parameter_change)
                self.widgets[action.dest] = (combobox, var)
            elif action.type == bool or isinstance(action.default, bool):
                # Параметры типа bool (чекбокс)
                var = tk.BooleanVar(value=getattr(self.args, action.dest, action.default))
                checkbutton = ttk.Checkbutton(frame, variable=var, command=self.on_parameter_change)
                checkbutton.pack(side=tk.RIGHT)
                self.widgets[action.dest] = (checkbutton, var)
            elif action.type in (float, int):
                # Параметры типа float или int (текстовое поле с валидацией)
                var = tk.StringVar(value=str(getattr(self.args, action.dest, action.default)))
                entry = ttk.Entry(frame, textvariable=var)
                entry.pack(side=tk.RIGHT, fill=tk.X, expand=True)
                entry.bind("<KeyRelease>", self.on_parameter_change)
                self.widgets[action.dest] = (entry, var)
            else:
                # Параметры типа str (текстовое поле)
                var = tk.StringVar(value=getattr(self.args, action.dest, action.default))
                entry = ttk.Entry(frame, textvariable=var)
                entry.pack(side=tk.RIGHT, fill=tk.X, expand=True)
                entry.bind("<KeyRelease>", self.on_parameter_change)
                self.widgets[action.dest] = (entry, var)

    def on_parameter_change(self, event=None):
        """Активирует кнопку Apply при изменении любого параметра."""
        self.apply_button.config(state=tk.NORMAL)

    def apply_changes(self):
        """Применяет изменения параметров и перезапускает сервер."""
        # Останавливаем сервер, если он запущен
        if self.server_running:
            self.stop_server()

        # Обновляем объект args текущими значениями из GUI
        for param, (widget, var) in self.widgets.items():
            value = var.get()  # Получаем текущее значение из виджета

            # Логируем значение для отладки
            logger.debug(f"Параметр: {param}, Значение из GUI: {value}")

            # Если значение — пустая строка, заменяем на None
            if isinstance(value, str) and value.strip() == "":
                value = None

            if isinstance(value, str) and value is not None:
                # Преобразуем строку в нужный тип
                action = next((a for a in self.parser._actions if a.dest == param), None)
                if action and action.type:
                    try:
                        # Преобразуем значение в нужный тип (int, float, bool и т.д.)
                        if action.type == bool:
                            value = bool(value)
                        else:
                            value = action.type(value)
                    except (ValueError, TypeError):
                        logger.error(f"Некорректное значение для параметра {param}: {value}")
                        continue

            # Устанавливаем значение в args
            setattr(self.args, param, value)

        # Проверяем, что модель не пустая
        if not self.args.model:
            logger.error("Модель не может быть пустой. Установлено значение по умолчанию: large-v3-turbo.")
            self.args.model = "large-v3-turbo"

        self.apply_button.config(state=tk.DISABLED)
        logger.info("Параметры успешно применены.")

        # Перезапускаем сервер, если он был запущен
        if self.server_running:
            self.start_server()


    def toggle_server(self):
        if self.server_running:
            self.stop_server()
        else:
            self.start_server()

    def start_server(self):
        """Запуск сервера в отдельном потоке."""
        if self.server_running:
            return

        # Применяем изменения перед запуском сервера
        self.apply_changes()

        self.stop_event.clear()  # Сбрасываем событие остановки
        self.server_running = True
        self.start_stop_button.config(text="Stop Server")

        # Запуск сервера в отдельном потоке
        self.server_thread = threading.Thread(target=self.run_server_wrapper, args=(self.args,))
        self.server_thread.start()

    def stop_server(self):
        """Остановка сервера."""
        if not self.server_running:
            return

        self.server_running = False
        self.start_stop_button.config(text="Start Server")
        self.stop_event.set()  # Устанавливаем событие остановки

        # Ожидаем завершения потока сервера
        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(timeout=5)  # Ожидаем завершения потока (максимум 5 секунд)

    def run_server_wrapper(self, args):
        """Обертка для запуска сервера с поддержкой остановки."""
        try:
            run_server(args, self.stop_event)
        except Exception as e:
            logger.error(f"Server error: {e}")
        finally:
            self.server_running = False
            self.start_stop_button.config(text="Start Server")

    def toggle_advanced(self):
        """Разворачивает/скрывает дополнительные параметры."""
        if self.advanced_options_visible:
            self.advanced_frame.pack_forget()
        else:
            self.advanced_frame.pack(pady=10)
        self.advanced_options_visible = not self.advanced_options_visible

    def on_close(self):
        """Обработка закрытия окна."""
        self.stop_server()  # Останавливаем сервер
        self.root.destroy()  # Закрываем окно

if __name__ == "__main__":
    main()