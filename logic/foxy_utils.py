#logic/foxy_utils.py
from  logic.foxy_config import *

import argparse 
import sys
import logging
import numpy as np
import librosa
import psutil
from functools import lru_cache
import subprocess
import importlib
import sounddevice as sd
#from wtpsplit import WtP

logger = logging.getLogger(__name__)
@lru_cache(10**6)

##########################
def get_default_audio_device():
    """Get default input device and its parameters"""
    try:
        devices = sd.query_devices()
        default_device = sd.query_devices(kind='input')
        device_id = None
        
        # Находим ID дефолтного устройства
        for i, dev in enumerate(devices):
            if dev.get('name') == default_device.get('name') and dev.get('max_input_channels') > 0:
                device_id = i
                break
        
        if device_id is None:
            logger.warning("No default input device found, using device 0")
            device_id = 0
            default_device = sd.query_devices(device_id, 'input')
            
        return {
            'device_id': device_id,
            'name': default_device.get('name'),
            'default_samplerate': int(default_device.get('default_samplerate')),
            'max_input_channels': default_device.get('max_input_channels'),
        }
    except Exception as e:
        logger.error(f"Error getting default audio device: {e}")
        return {'device_id': 0, 'default_samplerate': 44100, 'max_input_channels': 1}

##########################
def get_supported_sample_rates(device_info):
    """Returns list of supported sample rates for the audio device"""
    try:
        device_id = device_info['device_id']
        default_sr = device_info['default_samplerate']
        supported_srs = []
        
        # Проверяем стандартные частоты и дефолтную частоту устройства
        test_rates = sorted(list(set([8000, 16000, 22050, 44100, 48000, default_sr])))
        
        for sr in test_rates:
            try:
                sd.check_input_settings(device=device_id, samplerate=sr, channels=1)
                supported_srs.append(sr)
                logger.debug(f"Sample rate {sr} Hz is supported")
            except sd.PortAudioError:
                continue
                
        if not supported_srs:
            logger.warning(f"Using default sample rate: {default_sr}")
            return [default_sr]
            
        return supported_srs
    except Exception as e:
        logger.error(f"Error checking sample rates: {e}")
        return [device_info['default_samplerate']]

##########################
def initialize_audio_device(args):
    """Initialize audio device with optimal parameters"""
    try:
        # Получаем информацию о дефолтном устройстве
        device_info = get_default_audio_device()
        
        # Если указано конкретное устройство в аргументах
        if args.audio_device is not None:
            device_info['device_id'] = args.audio_device
        
        # Получаем поддерживаемые частоты
        supported_rates = get_supported_sample_rates(device_info)
        
        # Выбираем оптимальную частоту (ближайшую к 16000)
        target_sr = min(supported_rates, key=lambda x: abs(x - 16000))
        
        logger.info(f"Using audio device: {device_info['name']} (id: {device_info['device_id']})")
        logger.info(f"Sample rate: {target_sr} Hz")
        
        return device_info['device_id'], target_sr
        
    except Exception as e:
        logger.error(f"Error initializing audio device: {e}")
        raise

##########################
def add_shared_args(parser):
    """Adds shared arguments for ASR configuration."""
    parser.add_argument("--listen", type=str, default="audio_device", 
                       choices=["tcp", "audio_device"], 
                       help="Source of audio input: 'audio_device' (default) or 'tcp'.")
    parser.add_argument("--audio-device", type=int, default=None,
                       help="ID of the audio input device (if not specified, default device will be used).")

    parser.add_argument('--lan', '--language', type=str, default='auto', choices= WHISPER_LANG_CODES, help="Language of the input audio (e.g., 'ru', 'en' or 'auto' for autodetect).")
    parser.add_argument('--task', type=str, default='transcribe', choices=["transcribe", "translate"], help="Task: transcription or translation.")
    parser.add_argument('--vad', action="store_true", default=False, help="Enable VAD (Voice Activity Detection).")
    parser.add_argument('--vad-fade-time', type=int, default=500, help="VAD fade time in milliseconds.")
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
def load_audio(fname):
    """Modified to use the closest supported sample rate"""
    device_info = get_default_audio_device()
    supported_rates = get_supported_sample_rates(device_info)
    target_sr = min(supported_rates, key=lambda x: abs(x - 16000))
    logger.info(f"Using sample rate: {target_sr} Hz")
    a, _ = librosa.load(fname, sr=target_sr, dtype=np.float32)
    if target_sr != 16000:
        # Resample to required 16000 Hz
        a = librosa.resample(a, orig_sr=target_sr, target_sr=16000)
    return a

##########################
def load_audio_chunk(fname, beg, end):
    audio = load_audio(fname)
    beg_s = int(beg*16000)
    end_s = int(end*16000)
    return audio[beg_s:end_s]

##########################
def set_logging(args, logger):
    """Настройка системы логирования."""
    log_level = logging.DEBUG if args.get('debug', False) else logging.INFO
    logger.setLevel(log_level)
    
    # Очищаем все существующие обработчики
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Создаем новый консольный обработчик
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # Добавляем обработчик для файла, если нужно
    if args.get('log_file'):
        fh = logging.FileHandler(args['log_file'])
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)


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
