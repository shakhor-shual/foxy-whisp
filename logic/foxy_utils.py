from  logic.foxy_config import *

import sys
import logging
import numpy as np
import librosa
import psutil
from functools import lru_cache
import subprocess
import importlib
from wtpsplit import WtP

PACKET_SIZE = 65536

logger = logging.getLogger(__name__)
@lru_cache(10**6)

##########################
def load_audio(fname):
    a, _ = librosa.load(fname, sr=16000, dtype=np.float32)
    return a

##########################
def load_audio_chunk(fname, beg, end):
    audio = load_audio(fname)
    beg_s = int(beg*16000)
    end_s = int(end*16000)
    return audio[beg_s:end_s]


##########################
def set_logging(args, logger, other=""):
    """Настраивает уровень логирования."""
    logging.basicConfig(format='%(levelname)s\t%(message)s')
    logger.setLevel(args.log_level)
    logging.getLogger("whisper_online" + other).setLevel(args.log_level)


##########################
def add_shared_args(parser):
    """Adds shared arguments for ASR configuration."""
    
    parser.add_argument('--lan', '--language', type=str, default='auto', choices= WHISPER_LANG_CODES, help="Language of the input audio (e.g., 'ru', 'en' or 'auto' for autodetect).")
    parser.add_argument('--task', type=str, default='transcribe', choices=["transcribe", "translate"], help="Task: transcription or translation.")
    parser.add_argument('--vad', action="store_true", default=False, help="Enable VAD (Voice Activity Detection).")
    parser.add_argument('--vac', action="store_true", default=False, help="Enable VAC (Voice Activity Controller).")
    parser.add_argument('--vac-chunk-size', type=float, default=0.04, help="VAC segment size in seconds.")

    parser.add_argument('--backend', type=str, default="faster-whisper", choices=["faster-whisper", "whisper_timestamped", "openai-api"], help="Choose ASR-backend for Speech-To-Text.")

    parser.add_argument('--model', type=str, default='large-v3-turbo', choices=[
        "tiny.en", "tiny", "base.en", "base", "small.en", "small", "medium.en", "medium", "large-v1", "large-v2", "large-v3", "large-v3-turbo"
    ], help="Whisper model to use (default: large-v3-turbo).")
    parser.add_argument('--model-cache-dir', type=str, default=None, help="Directory for caching models.")
    parser.add_argument('--model-dir', type=str, default=None, help="Directory containing the Whisper model.")
    parser.add_argument("--warmup-file", type=str, dest="warmup_file", help="Path to a speech audio file to warm up Whisper.")

    parser.add_argument('--buffer-trimming', type=str, default="segment", choices=["sentence", "segment"], help="Buffer trimming strategy.(e.g., trim by completed senteces or by defined time segments)")
    parser.add_argument('--buffer-trimming-sec', type=float, default=15, help="Buffer trimming threshold in seconds.(for 'segment' trimming strategy) ")
    parser.add_argument('--min-chunk-size', type=float, default=1.0, help="Minimum audio segment size in seconds.")

    parser.add_argument("-l", "--log-level", dest="log_level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='DEBUG', help="Logging level.")
    parser.add_argument("--gui", action="store_true", help="Launch the server with a control GUI.")
    parser.add_argument("--host", type=str, default='0.0.0.0', help="Host address to bind the server to.")
    parser.add_argument("--port", type=int, default=43007, help="TCP Port number for the server income audio stream.")



##########################
def install_package(package_name: str):
    """Устанавливает пакет с помощью pip, если он не установлен."""
    try:
        importlib.import_module(package_name)
    except ImportError:
        logger.info(f"Установка пакета {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])


##########################
def send_one_line_tcp(socket, text, pad_zeros=False):
    text.replace('\0', '\n')
    lines = text.splitlines()
    first_line = '' if len(lines) == 0 else lines[0]
    # TODO Is there a better way of handling bad input than 'replace'?
    data = first_line.encode('utf-8', errors='replace') + b'\n' + (b'\0' if pad_zeros else b'')
    for offset in range(0, len(data), PACKET_SIZE):
        bytes_remaining = len(data) - offset
        if bytes_remaining < PACKET_SIZE:
            padding_length = PACKET_SIZE - bytes_remaining
            packet = data[offset:] + (b'\0' * padding_length if pad_zeros else b'')
        else:
            packet = data[offset:offset+PACKET_SIZE]
        socket.sendall(packet)


##########################
def receive_one_line_tcp(socket):
    data = b''
    while True:
        packet = socket.recv(PACKET_SIZE)
        if not packet:  # Connection has been closed.
            return None
        data += packet
        if b'\0' in packet:
            break
    # TODO Is there a better way of handling bad input than 'replace'?
    text = data.decode('utf-8', errors='replace').strip('\0')
    lines = text.split('\n')
    return lines[0] + '\n'


##########################
def receive_lines_tcp(socket):
    try:
        data = socket.recv(PACKET_SIZE)
    except BlockingIOError:
        return []
    if data is None:  # Connection has been closed.
        return None
    # TODO Is there a better way of handling bad input than 'replace'?
    text = data.decode('utf-8', errors='replace').strip('\0')
    lines = text.split('\n')
    if len(lines)==1 and not lines[0]:
        return None
    return lines


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
    

##########################
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


    # downloads the model from huggingface on the first use
    wtp = WtP("wtp-canine-s-12l-no-adapters")
    class WtPtok:
        def split(self, sent):
            return wtp.split(sent, lang_code=lan)
    return WtPtok()
