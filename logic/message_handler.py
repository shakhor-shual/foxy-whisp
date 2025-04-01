from multiprocessing import Queue as MPQueue
from typing import Optional, Dict, Any, Callable, Tuple
from .messaging.base_handler import BaseMessageHandler
from .foxy_message import PipelineMessage
from .messaging.types import MessageType, MessageSource
import logging
import queue

class MessageHandler(BaseMessageHandler):
    """Центральный обработчик сообщений и форматирования"""
    
    def __init__(self, gui_to_server: MPQueue, server_to_gui: MPQueue):
        super().__init__()
        self.gui_to_server = gui_to_server
        self.server_to_gui = server_to_gui
        self.callbacks = {
            'log': [],
            'status': [],
            'data': [],
            'vad': [],
            'audio_level': [],
            'button': [],
            'device_info': []
        }
        self.message_queue = queue.Queue()

    def register_callback(self, msg_type: str, callback: Callable):
        """Register callback for specific message type"""
        if msg_type in self.callbacks:
            self.callbacks[msg_type].append(callback)

    def send_command(self, command: str, params: dict = None) -> bool:
        """Send command to server with logging"""
        try:
            msg = PipelineMessage.create_command(
                source='gui',
                command=command,
                **(params or {})
            )
            return self._send_message(self.gui_to_server, msg)
        except Exception as e:
            self.logger.error(f"Error sending command: {e}")
            return False

    def handle_message(self, msg: PipelineMessage):
        """Handle incoming message from server"""
        try:
            if msg.is_status() and msg.source == 'src.vad':
                self._handle_vad_status(msg)
            elif msg.is_log():
                self._handle_log(msg)
            elif msg.is_status():
                self._handle_status(msg)
            elif msg.is_data():
                self._handle_data(msg)
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
            self._notify('log', f"[ERROR] Message handling failed: {e}")

    def _handle_vad_status(self, msg: PipelineMessage):
        """Handle VAD status updates"""
        if msg.content.get('status') == 'processing':
            voice_detected = msg.content.get('details', {}).get('voice_detected', False)
            self._notify('vad', voice_detected)

    def _handle_log(self, msg: PipelineMessage):
        """Handle log messages"""
        formatted = self.format_message(
            f"[{msg.source}.{msg.content.get('level', 'info')}] {msg.content.get('message', '')}"
        )
        if formatted:
            base_source, component, level, message = formatted
            self._notify('log', f"[{base_source}.{component}.{level}] {message}")

    def _handle_status(self, msg: PipelineMessage):
        """Handle status messages"""
        formatted = self.format_message(
            f"[{msg.source}.{msg.content.get('status', '')}] {msg.content.get('details', {})}"
        )
        if formatted:
            base_source, component, level, message = formatted
            self._notify('status', f"[{base_source}.{component}.{level}] {message}")
        self._process_status_updates(msg)

    def _handle_data(self, msg: PipelineMessage):
        """Handle data messages"""
        data_type = msg.content.get('data_type')
        if data_type == 'audio_level':
            level_data = msg.content.get('payload', {})
            if isinstance(level_data, dict) and 'level' in level_data:
                self._notify('audio_level', level_data)
        elif data_type == 'audio_devices':
            self._notify('device_info', msg.content.get('payload'))
        else:
            self._notify('data', msg.content)

    def _notify(self, msg_type: str, data: Any):
        """Notify registered callbacks"""
        for callback in self.callbacks.get(msg_type, []):
            try:
                callback(data)
            except Exception as e:
                self.logger.error(f"Callback error for {msg_type}: {e}")
                
    def _process_status_updates(self, msg: PipelineMessage):
        """Process status updates that affect UI state"""
        status = msg.content.get('status')
        if status in ('pipeline_started', 'pipeline_stopped', 'pipeline_error'):
            if status == 'pipeline_started':
                config = {'text': "Стоп", 'state': 'normal'}
            else:
                config = {'text': "Старт", 'state': 'normal'}
            self._notify('button', {'type': 'server_state', 
                                  'button': 'server_btn',
                                  'config': config})
                    
    def format_message(self, text: str) -> Optional[Tuple[str, str, str, str]]:
        """Перенесено из MessageFormatter"""
        try:
            if text.startswith('[') and ']' in text:
                parts = text[1:].split(']', 1)
                if len(parts) == 2:
                    source_parts = parts[0].lower().split('.')
                    message = parts[1].strip()
                    base_source = source_parts[0]
                    level = source_parts[-1] if len(source_parts) > 1 else 'info'
                    component = '.'.join(source_parts[1:-1]) if len(source_parts) > 2 else 'main'
                    base_source = self.source_mappings.get(base_source, base_source)
                    return base_source, component, level, message
            return None
        except Exception as e:
            self.logger.error(f"Message parsing error: {e}, Text: {text}")
            return None