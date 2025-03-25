from logic.foxy_config import *
from logic.foxy_utils import install_package, logger

import sys
import numpy as np
import io
import soundfile 
from typing import List, Tuple, Optional, Any, IO
from abc import ABC, abstractmethod
import math


#############################################################
class ASRBase(ABC):
    sep: str = " "
#######################
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

#######################
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

#######################
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


#########################################################################
class WhisperTimestampedASR(ASRBase):
    sep = " "
#######################
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

#######################
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

#######################
    def ts_words(self, r) -> List[Tuple[float, float, str]]:
        return [(w["start"], w["end"], w["text"]) for s in r["segments"] for w in s["words"]]

#######################
    def segments_end_ts(self, res) -> List[float]:
        return [s["end"] for s in res["segments"]]



#########################################################################
class FasterWhisperASR(ASRBase):
    sep = ""
#######################
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
    
#######################
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


#########################################################################
class OpenaiApiASR(ASRBase):
    def __init__(self, lan: Optional[str] = None, temperature: int = 0, logfile: IO = sys.stderr):
        super().__init__(lan, logfile=logfile)
        self.modelname = "whisper-1"
        self.temperature = temperature
        self.task = "transcribe"
        self.use_vad_opt = False
        self.transcribed_seconds = 0

#######################
    def load_model(self, *args, **kwargs) -> Any:
        install_package("openai")
        from openai import OpenAI
        self.client = OpenAI()

#######################
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

#######################
    def use_vad(self) -> None:
        self.use_vad_opt = True