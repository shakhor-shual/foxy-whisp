import unittest
from unittest.mock import Mock, patch, MagicMock
from multiprocessing import Queue, Event
import queue
from logic.foxy_pipeline import PipelineElement

class TestPipelineElement(unittest.TestCase):
    class ConcretePipelineElement(PipelineElement):
        def configure(self):
            pass
        
        def process(self, audio_chunk):
            return audio_chunk
            
        def start(self):
            pass
            
        def stop(self):
            pass

    def setUp(self):
        self.args = {'test': 'value'}
        self.pipe_input = Queue()
        self.pipe_output = Queue()
        self.in_queue = Queue()
        self.out_queue = Queue()
        self.element = self.ConcretePipelineElement(
            args=self.args,
            audio_in=self.pipe_input,
            audio_out=self.pipe_output,
            in_queue=self.in_queue,
            out_queue=self.out_queue
        )

    def tearDown(self):
        # Очищаем очереди перед закрытием
        self._clear_queue(self.pipe_input)
        self._clear_queue(self.pipe_output)
        self._clear_queue(self.in_queue)
        self._clear_queue(self.out_queue)
        
        # Останавливаем элемент если он запущен
        if hasattr(self, 'element'):
            self.element.stop_event.set()
            if hasattr(self.element, 'process'):
                self.element.process = lambda x: None
        
        # Закрываем очереди
        for q in [self.pipe_input, self.pipe_output, self.in_queue, self.out_queue]:
            try:
                q.close()
                q.join_thread()
            except:
                pass

    def _clear_queue(self, q):
        """Безопасная очистка очереди"""
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass

    def _get_with_timeout(self, q, timeout=1):
        """Получение данных из очереди с таймаутом"""
        try:
            return q.get(timeout=timeout)
        except queue.Empty:
            self.fail("Timeout waiting for queue data")

    def test_init(self):
        """Тест инициализации элемента пайплайна"""
        self.assertEqual(self.element.args, self.args)
        self.assertEqual(self.element.pipe_input, self.pipe_input)
        self.assertEqual(self.element.audio_out, self.pipe_output)
        self.assertIsInstance(self.element.stop_event, Event)
        self.assertIsInstance(self.element.pause_event, Event)

    def test_pipe_read(self):
        """Тест чтения данных из входной очереди"""
        test_data = b"test_audio"
        self.pipe_input.put(test_data)
        result = self.element.audio_read()
        self.assertEqual(result, test_data)
        
        # Проверка таймаута при пустой очереди
        result = self.element.audio_read()
        self.assertIsNone(result)

    def test_pipe_write(self):
        """Тест записи данных в выходную очередь"""
        test_data = b"test_audio"
        self.element.pipe_write(test_data)
        result = self.pipe_output.get()
        self.assertEqual(result, test_data)

    def test_process_control_commands(self):
        """Тест обработки управляющих команд"""
        # Тест команды stop
        self.in_queue.put({"type": "command", "content": "stop"})
        self.element.process_control_commands()
        self.assertTrue(self.element.stop_event.is_set())

        # Тест команды pause
        self.in_queue.put({"type": "command", "content": "pause"})
        self.element.process_control_commands()
        self.assertTrue(self.element.pause_event.is_set())

        # Тест команды resume
        self.in_queue.put({"type": "command", "content": "resume"})
        self.element.process_control_commands()
        self.assertFalse(self.element.pause_event.is_set())

    @patch('time.sleep')  # Мокаем sleep для ускорения тестов
    def test_run(self, mock_sleep):
        """Тест основного цикла обработки"""
        test_data = b"test_audio"
        self.pipe_input.put(test_data)
        
        # Устанавливаем событие остановки после одной итерации
        def stop_after_iteration(*args):
            self.element.stop_event.set()
            return test_data

        self.element.process = Mock(side_effect=stop_after_iteration)
        self.element.run()

        # Проверяем, что данные были обработаны
        self.element.process.assert_called_once_with(test_data)
        
    @patch('time.sleep')
    def test_run_with_empty_input(self, mock_sleep):
        """Тест работы с пустым входом"""
        def stop_after_delay(*args):
            self.element.stop_event.set()
        mock_sleep.side_effect = stop_after_delay
        
        self.element.run()
        mock_sleep.assert_called()

    @patch('time.sleep')
    def test_run_with_error(self, mock_sleep):
        """Тест обработки ошибок"""
        test_data = b"test_audio"
        self.pipe_input.put(test_data)
        
        def raise_error(data):
            raise Exception("Test error")
            
        self.element.process = Mock(side_effect=raise_error)
        
        def stop_after_iteration(*args):
            self.element.stop_event.set()
        mock_sleep.side_effect = stop_after_iteration
        
        self.element.run()
        
        # Проверяем статус ошибки
        result = self._get_with_timeout(self.out_queue)
        self.assertEqual(result.get("type"), "status")
        self.assertEqual(result.get("content"), "stopped")

    def test_process_control_commands_error(self):
        """Тест обработки ошибок в командах управления"""
        self.in_queue.put({"type": "invalid"})
        self.element.process_control_commands()
        # Не должно быть исключений

    def test_send_status(self):
        """Тест отправки статуса"""
        test_status = "test_status"
        self.element.send_status(test_status)
        result = self.out_queue.get()
        self.assertEqual(result["type"], "status")
        self.assertEqual(result["content"], test_status)

    def test_send_data(self):
        """Тест отправки данных"""
        test_data = {"test": "data"}
        self.element.send_data(test_data)
        result = self.out_queue.get()
        self.assertEqual(result["type"], "data")
        self.assertEqual(result["content"], test_data)

if __name__ == '__main__':
    unittest.main()
