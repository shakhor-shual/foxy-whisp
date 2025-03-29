import unittest
from multiprocessing import Event, Queue
import numpy as np
from logic.foxy_pipeline import SRCstage
from logic.vad_filters import WebRTCVAD

class TestSRCStage(unittest.TestCase):
    def setUp(self):
        self.stop_event = Event()
        self.audio_out = Queue()
        self.out_queue = Queue()
        self.in_queue = Queue()
        self.args = {
            "vad_aggressiveness": 3,
            "listen": "tcp",
            "sample_rate": 16000,
            "chunk_size": 320
        }
        self.src_stage = SRCstage(
            stop_event=self.stop_event,
            audio_out=self.audio_out,
            out_queue=self.out_queue,
            in_queue=self.in_queue,
            args=self.args
        )
        self.src_stage.vad = WebRTCVAD(aggressiveness=3)  # Используем WebRTC VAD для тестирования

    def test_add_data_to_buffer(self):
        """Test that audio data is correctly added to the buffer."""
        audio_data = np.zeros(16000, dtype=np.float32)  # 1 секунда тишины
        self.src_stage.fifo_buffer.add_data(audio_data)
        self.assertEqual(len(self.src_stage.fifo_buffer.buffer), 16000)

    def test_buffer_overflow(self):
        """Test that buffer handles overflow correctly."""
        max_duration = 5  # 5 секунд
        max_size = 16000 * max_duration
        self.src_stage.fifo_buffer.max_size = max_size

        # Добавляем данные, превышающие размер буфера
        audio_data = np.ones(max_size + 16000, dtype=np.float32)  # 6 секунд данных
        self.src_stage.fifo_buffer.add_data(audio_data)

        # Проверяем, что размер буфера не превышает max_size
        self.assertEqual(len(self.src_stage.fifo_buffer.buffer), max_size)

    def test_buffer_overflow_keeps_latest_data(self):
        """Test that buffer keeps latest data when overflow occurs."""
        max_duration = 5  # 5 секунд
        max_size = 16000 * max_duration
        self.src_stage.fifo_buffer.max_size = max_size

        # Создаем два разных сигнала
        first_data = np.ones(max_size, dtype=np.float32)  # Старые данные
        second_data = np.full(16000, 2, dtype=np.float32)  # Новые данные

        # Добавляем данные последовательно
        self.src_stage.fifo_buffer.add_data(first_data)
        self.src_stage.fifo_buffer.add_data(second_data)

        # Проверяем, что последние данные сохранены
        buffer_end = self.src_stage.fifo_buffer.buffer[-16000:]
        self.assertTrue(np.all(buffer_end == 2))

    def test_get_chunk_from_buffer(self):
        """Test that chunks are correctly read from the buffer."""
        audio_data = np.ones(16000, dtype=np.float32)  # 1 секунда данных
        self.src_stage.fifo_buffer.add_data(audio_data)

        # Читаем чанк фиксированной длины
        chunk_size = 320
        chunk = self.src_stage.fifo_buffer.get_chunk(chunk_size)
        self.assertEqual(len(chunk), chunk_size)

        # Проверяем, что оставшиеся данные в буфере корректны
        self.assertEqual(len(self.src_stage.fifo_buffer.buffer), 16000 - chunk_size)

    def test_get_chunk_insufficient_data(self):
        """Test behavior when buffer has insufficient data."""
        audio_data = np.ones(100, dtype=np.float32)  # Меньше, чем размер чанка
        self.src_stage.fifo_buffer.add_data(audio_data)

        # Пытаемся прочитать чанк большего размера
        chunk_size = 320
        chunk = self.src_stage.fifo_buffer.get_chunk(chunk_size)
        self.assertIsNone(chunk)  # Ожидаем, что данных недостаточно

    def test_vad_detects_voice(self):
        """Test that VAD detects voice and forwards data."""
        # Генерируем сигнал с голосом (синусоида 440 Гц)
        sample_rate = 16000
        duration = 1  # 1 секунда
        frequency = 440  # 440 Гц
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        audio_data = (0.5 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)

        # Добавляем данные в буфер и обрабатываем
        self.src_stage.fifo_buffer.add_data(audio_data)
        vad_chunk_size = self.src_stage.vad.get_chunk_size()

        while True:
            chunk = self.src_stage.fifo_buffer.get_chunk(vad_chunk_size)
            if chunk is None:
                break
            if self.src_stage.vad.detect_voice(chunk):
                self.src_stage.audio_write(chunk)

        # Проверяем, что данные с голосом были переданы в выходную очередь
        self.assertFalse(self.audio_out.empty())
        forwarded_chunk = self.audio_out.get()
        self.assertEqual(len(forwarded_chunk), vad_chunk_size)

    def test_vad_silence(self):
        """Test that VAD does not forward silence."""
        # Генерируем тишину
        audio_data = np.zeros(16000, dtype=np.float32)  # 1 секунда тишины

        # Добавляем данные в буфер и обрабатываем
        self.src_stage.fifo_buffer.add_data(audio_data)
        vad_chunk_size = self.src_stage.vad.get_chunk_size()

        while True:
            chunk = self.src_stage.fifo_buffer.get_chunk(vad_chunk_size)
            if chunk is None:
                break
            if self.src_stage.vad.detect_voice(chunk):
                self.src_stage.audio_write(chunk)

        # Проверяем, что данные не были переданы в выходную очередь
        self.assertTrue(self.audio_out.empty())

    def test_continuous_voice_detection(self):
        """Test that VAD correctly processes continuous speech."""
        # Генерируем длинный речеподобный сигнал
        sample_rate = 16000
        duration = 3  # 3 секунды
        frequency = 440  # 440 Hz
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        audio_data = (0.5 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)

        chunks_detected = 0
        self.src_stage.process(audio_data)

        # Подсчитываем количество обнаруженных чанков с речью
        while not self.audio_out.empty():
            self.audio_out.get()
            chunks_detected += 1

        # Проверяем, что обнаружено ожидаемое количество чанков
        expected_chunks = len(audio_data) // self.src_stage.vad.get_chunk_size()
        self.assertGreater(chunks_detected, 0)
        self.assertLessEqual(chunks_detected, expected_chunks)

    def test_resampling_with_different_rates(self):
        """Test resampling with different input sample rates."""
        # Тестируем различные частоты дискретизации
        test_rates = [8000, 22050, 44100]
        
        for rate in test_rates:
            with self.subTest(sample_rate=rate):
                # Генерируем тестовый сигнал
                duration = 1  # 1 секунда
                t = np.linspace(0, duration, int(rate * duration), endpoint=False)
                audio_data = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

                # Обновляем параметры
                self.src_stage.args["sample_rate"] = rate

                # Выполняем ресемплинг
                resampled = self.src_stage._resample_to_16k(audio_data)

                # Проверяем результат
                self.assertEqual(len(resampled), 16000)
                self.assertTrue(np.all(np.isfinite(resampled)))
                self.assertTrue(np.all(np.abs(resampled) <= 1.0))

    def test_buffer_underflow_recovery(self):
        """Test that processing continues after buffer underflow."""
        # Сначала создаем условие недостатка данных
        small_chunk = np.ones(100, dtype=np.float32)
        self.src_stage.process(small_chunk)
        
        # Проверяем, что очередь пуста (не хватило данных для обработки)
        self.assertTrue(self.audio_out.empty())
        
        # Теперь добавляем достаточно данных
        full_chunk = np.ones(16000, dtype=np.float32)
        self.src_stage.process(full_chunk)
        
        # Проверяем, что обработка возобновилась
        self.assertFalse(self.audio_out.empty())

    def test_end_to_end_processing(self):
        """Test end-to-end processing from source to output."""
        # Генерируем сигнал с голосом (синусоида 440 Гц)
        sample_rate = 16000
        duration = 1  # 1 секунда
        frequency = 440  # 440 Гц
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        audio_data = (0.5 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)

        # Обрабатываем данные через SRCstage
        self.src_stage.process(audio_data)

        # Проверяем, что данные с голосом были переданы в выходную очередь
        self.assertFalse(self.audio_out.empty())
        forwarded_chunk = self.audio_out.get()
        vad_chunk_size = self.src_stage.vad.get_chunk_size()
        self.assertEqual(len(forwarded_chunk), vad_chunk_size)

if __name__ == "__main__":
    unittest.main()
