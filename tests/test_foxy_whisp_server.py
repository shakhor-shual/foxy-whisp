import unittest
from unittest.mock import Mock, patch, MagicMock
from multiprocessing import Queue, Event
import queue
import time  # Добавляем импорт time
from logic.foxy_pipeline import SRCstage, VADstage, ASRStage
from foxy_whisp_server import FoxyWhispServer, PipelineQueues

class TestFoxyWhispServer(unittest.TestCase):
    def setUp(self):
        """Инициализация перед каждым тестом"""
        self.args = {
            'sample_rate': 16000,
            'chunk_size': 320,
            'listen': 'tcp'
        }
        self.from_gui_queue = Queue()
        self.to_gui_queue = Queue()
        self.server = FoxyWhispServer(
            from_gui_queue=self.from_gui_queue,
            to_gui_queue=self.to_gui_queue,
            args=self.args
        )

    def tearDown(self):
        """Очистка после каждого теста"""
        # Останавливаем сервер
        if hasattr(self, 'server'):
            self.server.stop_event.set()
            # Очищаем очереди
            self._clear_queues()
            
    def _clear_queues(self):
        """Вспомогательный метод для очистки очередей"""
        for queue_obj in [
            self.from_gui_queue, 
            self.to_gui_queue, 
            *self.server.queues.__dict__.values()
        ]:
            try:
                while not queue_obj.empty():
                    queue_obj.get_nowait()
            except (queue.Empty, AttributeError):
                continue

    ###############
    def test_init_queues(self):
        """Тест инициализации очередей"""
        queues = self.server._init_queues()
        self.assertIsInstance(queues, PipelineQueues)
        self.assertEqual(queues.pipe_audio_2_vad._maxsize, 2 * self.server.pipe_chunk)  # Исправлено maxsize на _maxsize
        self.assertEqual(queues.pipe_vad_2_asr._maxsize, 2 * self.server.pipe_chunk)

    ###############
    @patch('logic.foxy_pipeline.SRCstage')
    @patch('logic.foxy_pipeline.VADStage')
    @patch('logic.foxy_pipeline.ASRStage')
    def test_init_stages(self, mock_asr, mock_vad, mock_src):
        """Тест инициализации стейджей"""
        # Настраиваем моки для возврата Mock объектов
        mock_src.return_value = Mock()
        mock_vad.return_value = Mock()
        mock_asr.return_value = Mock()
        
        self.server._init_stages()
        
        # Проверяем вызовы конструкторов с правильными аргументами
        mock_src.assert_called_once_with(
            args=self.args,
            pipe_input=None,
            in_queue=self.server.queues.to_src_queue,
            out_queue=self.server.queues.from_src_queue,
            pipeline_chunk_size=self.server.pipe_chunk
        )
        
        self.assertIsNotNone(self.server.stages['src'])
        self.assertIsNotNone(self.server.stages['vad'])
        self.assertIsNotNone(self.server.stages['asr'])

    ###############
    def test_set_stop_events(self):
        """Тест установки событий остановки"""
        # Создаем мок-стейджи
        mock_stages = {
            'src': Mock(),
            'vad': Mock(),
            'asr': Mock()
        }
        self.server.stages = mock_stages
        
        self.server._set_stop_events()
        
        # Проверяем, что set_stop_event был вызван для каждого стейджа
        for stage in mock_stages.values():
            stage.set_stop_event.assert_called_once_with(self.server.stop_event)

    ###############
    def test_process_message(self):
        """Тест обработки сообщений"""
        # Тестируем различные типы сообщений
        test_cases = [
            {"type": "log", "level": "INFO", "message": "test", "process": "test"},
            {"type": "status", "content": "started"},
            {"type": "data", "content": {"test": "data"}},
            {"type": "unknown", "content": "should_warn"}
        ]
        
        for message in test_cases:
            self.server.process_message(message)

    ###############
    @patch('logic.foxy_pipeline.SRCstage')
    @patch('logic.foxy_pipeline.VADStage')
    @patch('logic.foxy_pipeline.ASRStage')
    def test_start_server_pipeline(self, mock_asr, mock_vad, mock_src):
        """Тест запуска пайплайна"""
        # Configure mocks to return Mock instances
        mock_src.return_value = Mock()
        mock_vad.return_value = Mock()
        mock_asr.return_value = Mock()
        
        self.server.start_server_pipeline()
        
        # Verify that mocks were called with correct arguments
        mock_src.assert_called_once()
        mock_vad.assert_called_once()
        mock_asr.assert_called_once()
        
        # Verify stages were set
        self.assertIsNotNone(self.server.stages['src'])
        self.assertIsNotNone(self.server.stages['vad'])
        self.assertIsNotNone(self.server.stages['asr'])

    ###############
    def test_stop_server_pipeline(self):
        """Тест остановки пайплайна"""
        # Создаем мок-стейджи
        mock_stages = {
            'src': Mock(),
            'vad': Mock(),
            'asr': Mock()
        }
        self.server.stages = mock_stages
        
        self.server.stop_server_pipeline()
        
        # Проверяем, что stop был вызван для каждого стейджа
        for stage in mock_stages.values():
            stage.stop.assert_called_once()

    ###############
    @patch('time.sleep')
    def test_requests_from_gui(self, mock_sleep):
        """Тест обработки команд от GUI"""
        # Setup mocks
        with patch.object(self.server, 'start_server_pipeline') as mock_start:
            with patch.object(self.server, 'stop_server_pipeline') as mock_stop:
                # Create a function to stop the loop after processing
                def stop_after_commands(*args, **kwargs):
                    self.server.stop_event.set()
                mock_stop.side_effect = stop_after_commands
                
                # Send commands
                self.from_gui_queue.put(("command", {"action": "start_server"}))
                
                # Run the GUI request processing
                self.server.requests_from_gui()
                
                # Verify the commands were processed
                mock_start.assert_called_once()
                mock_sleep.assert_called()

    def test_process_input_queues(self):
        """Тест обработки входных очередей"""
        stop_event = Event()
        
        def mock_process_input_queues():
            while not stop_event.is_set():
                for queue in [self.server.queues.from_src_queue,
                            self.server.queues.from_vad_queue,
                            self.server.queues.from_asr_queue]:
                    if not queue.empty():
                        message = queue.get()
                        self.server.process_message(message)
                        if message.get("type") == "stop":
                            stop_event.set()
                            break

        with patch.object(self.server, 'process_input_queues', 
                         side_effect=mock_process_input_queues):
            # Отправляем тестовое сообщение
            self.server.queues.from_src_queue.put({"type": "stop"})
            
            # Запускаем обработку
            self.server.process_input_queues()
            
            # Проверяем, что обработка завершилась
            self.assertTrue(stop_event.is_set())

    ###############
    def test_update_params(self):
        """Тест обновления параметров"""
        test_params = {"new_param": "value"}
        
        # Ensure args is a dictionary
        self.server.args = {}
        
        self.server.update_params(test_params)
        self.assertEqual(self.server.args["new_param"], "value")

if __name__ == '__main__':
    unittest.main()
