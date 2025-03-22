#logic/foxy_utils.py
from  logic.foxy_config import *

import sys
import logging
import numpy as np
import librosa
import psutil
from functools import lru_cache
import subprocess
import importlib
#from wtpsplit import WtP

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
