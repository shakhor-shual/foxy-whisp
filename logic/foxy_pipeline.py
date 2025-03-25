from multiprocessing import Process, Queue, Event
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
import time
from logic.audio_sources import AudioDeviceSource, TCPSource
from logic.vad_filters import VADBase, create_vad
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from logic.foxy_message import PipelineMessage, MessageType


class PipelineElement(ABC):
    def __init__(self, 
                 stop_event: Event,
                 args: Dict[str, Any], 
                 audio_in: Optional[Queue] = None,
                 audio_out: Optional[Queue] = None,
                 in_queue: Optional[Queue] = None,
                 out_queue: Optional[Queue] = None,
                 pipe_chunk_size: int = 320):
        self.stop_event = stop_event
        self.args = args
        self.pipe_input = audio_in
        self.audio_out = audio_out
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.chunk_size = pipe_chunk_size
        self.pause_event = Event()
        self._process = None

    def send_log(self, message: str, level: str = "info", **kwargs):
        msg = PipelineMessage(
            source=self.__class__.__name__.lower(),
            type=MessageType.LOG,
            content={'message': message, 'level': level, **kwargs}
        )
        msg.send(self.out_queue)

    def send_status(self, status: str, **details):
        msg = PipelineMessage(
            source=self.__class__.__name__.lower(),
            type=MessageType.STATUS,
            content={'status': status, 'details': details}
        )
        msg.send(self.out_queue)

    def send_data(self, data_type: str, data: Any, **metadata):
        msg = PipelineMessage(
            source=self.__class__.__name__.lower(),
            type=MessageType.DATA,
            content={'data_type': data_type, 'payload': data, **metadata}
        )
        msg.send(self.out_queue)

    def process_control_commands(self):
        while msg := PipelineMessage.receive(self.in_queue):
            if msg.type == MessageType.COMMAND:
                self._handle_command(msg.content)

    def _handle_command(self, command: Dict[str, Any]):
        cmd = command.get('command')
        if cmd == 'stop':
            self.stop_event.set()
        elif cmd == 'pause':
            self.pause_event.set()
        elif cmd == 'resume':
            self.pause_event.clear()

    def audio_read(self) -> Optional[bytes]:
        if self.pipe_input:
            try:
                return self.pipe_input.get(timeout=0.1)
            except:
                return None
        return None

    def audio_write(self, data: bytes):
        if self.audio_out and data:
            self.audio_out.put(data)

    @abstractmethod
    def configure(self):
        pass

    @abstractmethod
    def process(self, audio_chunk: bytes) -> Optional[bytes]:
        pass

    def _run(self):
        self.send_status('starting')
        try:
            self.configure()
            while not self.stop_event.is_set():
                self.process_control_commands()
                
                if self.pause_event.is_set():
                    time.sleep(0.1)
                    continue

                if data := self.audio_read():
                    if processed := self.process(data):
                        self.audio_write(processed)
                else:
                    time.sleep(0.01)
                    
        except Exception as e:
            self.send_log(f"Error in {self.__class__.__name__}: {str(e)}", level="error")
            self.send_status('error', error=str(e))
        finally:
            self.send_status('stopped')

    def start(self):
        if self._process is None or not self._process.is_alive():
            self._process = Process(target=self._run)
            self._process.start()
            self.send_log(f"Process started (PID: {self._process.pid})")

    def stop(self):
        if self._process and self._process.is_alive():
            self.stop_event.set()
            self._process.join(timeout=2.0)
            if self._process.is_alive():
                self._process.terminate()

class SRCstage(PipelineElement):
    def __init__(self, stop_event, audio_out, out_queue, in_queue, args, pipe_chunk_size=320):
        super().__init__(stop_event, args, None, audio_out, in_queue, out_queue, pipe_chunk_size)
        self.source = None
        self.send_log("SRC stage initialized")

    def configure(self):
        listen_type = self.args.get("listen", "tcp")
        self.send_log(f"Configuring {listen_type} source")
        
        if listen_type == "tcp":
            self.source = TCPSource(args=self.args, stop_event=self.stop_event)
        elif listen_type == "audio_device":
            self.source = AudioDeviceSource(
                samplerate=self.args.get("sample_rate", 16000),
                blocksize=self.args.get("chunk_size", 320),
                device=self.args.get("audio_device"),
                stop_event=self.stop_event
            )
        else:
            raise ValueError(f"Unknown source type: {listen_type}")
        
        self.send_status('configured')

    def process(self, audio_chunk):
        return audio_chunk  # SRC просто передает аудио дальше

    def _run(self):
        self.send_status('starting')
        try:
            self.configure()
            while not self.stop_event.is_set():
                self.process_control_commands()
                
                if self.pause_event.is_set():
                    time.sleep(0.1)
                    continue

                if audio_data := self.source.receive_audio():
                    self.audio_write(audio_data)
                    self.send_data('audio_chunk', {'size': len(audio_data)})
                else:
                    time.sleep(0.01)
                    
        except Exception as e:
            self.send_log(f"SRC error: {str(e)}", level="error")
            self.send_status('error', error=str(e))
        finally:
            self.send_status('stopped')

class ASRstage(PipelineElement):
    def __init__(self, stop_event, audio_in, out_queue, in_queue, args, pipe_chunk_size=320):
        super().__init__(stop_event, args, audio_in, None, in_queue, out_queue, pipe_chunk_size)
        self.chunk_counter = 0
        self.text_buffer = ""
        self.test_phrases = [
            "Тестовое распознавание текста",
            "Продолжаем тестирование системы",
            "Это имитация работы ASR"
        ]
        self.send_log("ASR stage initialized")

    def configure(self):
        self.send_status('configured')

    def process(self, audio_chunk):
        self.chunk_counter += 1
        
        if self.chunk_counter % 50 == 0:  # Каждые 50 чанков
            phrase = self.test_phrases[self.chunk_counter % len(self.test_phrases)]
            self.text_buffer += phrase + " "
            self.send_data('asr_result', {
                'text': phrase,
                'buffer': self.text_buffer,
                'is_final': False
            })
        
        return None  # ASR не передает аудио дальше

    def _run(self):
        self.send_status('starting')
        try:
            self.configure()
            while not self.stop_event.is_set():
                self.process_control_commands()
                
                if self.pause_event.is_set():
                    time.sleep(0.1)
                    continue

                if data := self.audio_read():
                    self.process(data)
                else:
                    time.sleep(0.01)
                    
        except Exception as e:
            self.send_log(f"ASR error: {str(e)}", level="error")
            self.send_status('error', error=str(e))
        finally:
            if self.text_buffer:
                self.send_data('asr_final_result', {
                    'text': self.text_buffer,
                    'is_final': True
                })
            self.send_status('stopped')