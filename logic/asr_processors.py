#/logic/asr_processor.py
from  logic.foxy_config import *
from logic.foxy_utils import load_audio_chunk,  install_package, set_logging, logger, send_one_line_tcp, receive_lines_tcp, get_port_status
from logic.outcomes_buffer import IdeasBuffer

import sys
import numpy as np

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
        self.transcript_buffer = IdeasBuffer(logfile=self.logfile)
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
