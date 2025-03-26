from multiprocessing import Event, Queue
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)

class PipelineElement:
    def __init__(self, stop_event: Event, in_queue: Queue, out_queue: Queue):
        self.stop_event = stop_event
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.name = self.__class__.__name__

    def pipe_read(self) -> Optional[Any]:
        """Read data from input queue"""
        try:
            return self.in_queue.get()
        except Exception as e:
            logger.error(f"{self.name}: Error reading from queue: {e}")
            return None

    def pipe_write(self, data: Any) -> bool:
        """Write data to output queue"""
        try:
            self.out_queue.put(data)
            return True
        except Exception as e:
            logger.error(f"{self.name}: Error writing to queue: {e}")
            return False

    def process_control_commands(self, msg: Any) -> bool:
        """Process control commands"""
        try:
            if msg.content.get('command') == 'stop':
                self.stop_event.set()
                return True
            return False
        except Exception as e:
            logger.error(f"{self.name}: Error processing command: {e}")
            return False

    def process_data(self, data: Any) -> None:
        """Process data - to be implemented by subclasses"""
        raise NotImplementedError

    def send_data(self, data: Any) -> None:
        """Send data message"""
        self.pipe_write({
            'type': 'data',
            'payload': data
        })

    def send_status(self, status: str) -> None:
        """Send status message"""
        self.pipe_write({
            'type': 'status',
            'status': status
        })

    def run(self) -> None:
        """Main processing loop"""
        try:
            while not self.stop_event.is_set():
                data = self.pipe_read()
                if data is None:
                    continue
                    
                try:
                    self.process_data(data)
                except Exception as e:
                    logger.error(f"{self.name}: Error processing data: {e}")
                    self.stop_event.set()
                    break
                    
        except Exception as e:
            logger.error(f"{self.name}: Fatal error in run loop: {e}")
            self.stop_event.set()
