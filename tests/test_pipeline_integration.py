import unittest
from multiprocessing import Event, Queue
import numpy as np
import time
from logic.foxy_pipeline import SRCstage, ASRstage
from logic.vad_filters import WebRTCVAD
from logic.foxy_message import MessageType, PipelineMessage
from queue import Empty

class TestPipelineIntegration(unittest.TestCase):
    def setUp(self):
        # Общие события и очереди
        self.stop_event = Event()
        self.src_2_asr = Queue()  # Очередь между SRC и ASR
        self.out_queue = Queue()  # Очередь для логов/статусов
        
        # Базовые аргументы
        self.args = {
            "vad_aggressiveness": 3,
            "listen": "tcp",
            "sample_rate": 16000,
            "chunk_size": 320
        }

        # Инициализация SRC stage
        self.src_stage = SRCstage(
            stop_event=self.stop_event,
            audio_out=self.src_2_asr,  # Выход SRC подключен к очереди src_2_asr
            out_queue=self.out_queue,
            in_queue=Queue(),
            args=self.args
        )

        # Инициализация ASR stage
        self.asr_stage = ASRstage(
            stop_event=self.stop_event,
            audio_in=self.src_2_asr,  # Вход ASR подключен к очереди src_2_asr
            out_queue=self.out_queue,
            in_queue=Queue(),
            args=self.args
        )

        # Установка VAD для тестирования
        self.src_stage.vad = WebRTCVAD(aggressiveness=3)

    def generate_test_signal(self, duration=3.0, sample_rate=16000):
        """Генерация тестового сигнала с паузами."""
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        signal = np.zeros(len(t), dtype=np.float32)
        
        # Первая секунда - громкий сигнал (0.9 амплитуды)
        signal[:sample_rate] = 0.9 * np.sin(2 * np.pi * 440 * t[:sample_rate])
        
        # Вторая секунда - тишина (пауза)
        # signal[sample_rate:2*sample_rate] остается нулевой
        
        # Третья секунда - сигнал средней громкости (0.3 амплитуды)
        signal[2*sample_rate:] = 0.3 * np.sin(2 * np.pi * 440 * t[2*sample_rate:])
        
        return signal

    def collect_messages(self, timeout=1.0):
        """Сбор всех сообщений из очереди с таймаутом."""
        messages = []
        end_time = time.time() + timeout
        
        while time.time() < end_time:
            try:
                msg = self.out_queue.get_nowait()
                if isinstance(msg, dict):
                    print(f"Received dict message: {msg}")  # Debug output
                    messages.append(msg)
                elif isinstance(msg, PipelineMessage):
                    print(f"Received pipeline message: {msg.content}")  # Debug output
                    messages.append(msg)
            except Empty:
                time.sleep(0.01)
                continue
                
        return messages

    def test_end_to_end_pipeline(self):
        """Тест сквозной обработки данных через весь пайплайн."""
        try:
            # Генерируем тестовый аудиосигнал с большей амплитудой
            sample_rate = 16000
            duration = 30
            frequency = 440
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            # Увеличиваем амплитуду сигнала для лучшего детектирования VAD
            audio_data = (0.9 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)

            # Запускаем процессы
            self.src_stage.start()
            self.asr_stage.start()

            # Проверяем, что процессы запустились
            self.assertTrue(self.src_stage.is_alive(), "SRC stage не запустился")
            self.assertTrue(self.asr_stage.is_alive(), "ASR stage не запустился")

            # Используем меньшие чанки для более частой обработки
            chunk_size = 480  # 30мс при 16кГц
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                self.src_stage.process(chunk)
                # Увеличиваем задержку для обработки
                time.sleep(0.02)

            # Увеличиваем время ожидания обработки
            time.sleep(2.0)

            # Собираем все сообщения из очереди с таймаутом
            messages = []
            timeout = time.time() + 5.0  # 5 секунд таймаут
            while time.time() < timeout:
                try:
                    msg = self.out_queue.get_nowait()
                    messages.append(msg)
                    if isinstance(msg, PipelineMessage) and msg.type == MessageType.DATA:
                        print(f"Received message: {msg.content}")  # Отладочный вывод
                except:
                    time.sleep(0.1)
                    continue

            # Фильтруем и проверяем сообщения
            asr_messages = [
                msg for msg in messages 
                if isinstance(msg, PipelineMessage) 
                and msg.type == MessageType.DATA 
                and msg.content.get('data_type') == 'asr_result'
            ]

            # Добавляем отладочную информацию
            print(f"Total messages: {len(messages)}")
            print(f"ASR messages: {len(asr_messages)}")

            # Проверки с подробной информацией
            self.assertGreater(
                len(asr_messages), 0, 
                f"ASR должен произвести хотя бы один результат. "
                f"Всего сообщений: {len(messages)}"
            )

            # Проверяем каждое сообщение
            for msg in asr_messages:
                with self.subTest(msg=msg):
                    payload = msg.content.get('payload', {})
                    self.assertIn('text', payload, "Отсутствует поле 'text'")
                    self.assertIn('buffer', payload, "Отсутствует поле 'buffer'")
                    self.assertIn('is_final', payload, "Отсутствует поле 'is_final'")
                    self.assertIsInstance(payload['text'], str, "Поле 'text' должно быть строкой")
                    self.assertIsInstance(payload['buffer'], str, "Поле 'buffer' должно быть строкой")
                    self.assertIsInstance(payload['is_final'], bool, "Поле 'is_final' должно быть булевым")

        except Exception as e:
            self.fail(f"Тест завершился с ошибкой: {str(e)}")

    def test_end_to_end_pipeline_with_vu_meter(self):
        """Тест сквозной обработки с проверкой VU-метра и пауз."""
        try:
            # Генерируем тестовый сигнал с паузами
            sample_rate = 16000
            duration = 3  # 3 секунды
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            
            # Создаем сигнал с чередующимися активными участками и паузами
            audio_data = np.zeros(len(t), dtype=np.float32)
            audio_data[:sample_rate] = 0.9 * np.sin(2 * np.pi * 440 * t[:sample_rate])  # Громкий
            audio_data[2*sample_rate:] = 0.3 * np.sin(2 * np.pi * 440 * t[2*sample_rate:])  # Средний

            # Запускаем процессы
            self.src_stage.start()
            self.asr_stage.start()

            # Обрабатываем данные небольшими чанками
            chunk_size = 480  # 30мс при 16кГц
            messages = []
            
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                self.src_stage.process(chunk)
                time.sleep(0.02)  # Даем время на обработку

            # Увеличиваем время ожидания для сбора сообщений
            time.sleep(2.0)  # Даем время на обработку всех данных

            # Собираем сообщения об уровне сигнала
            messages = []
            timeout = time.time() + 5.0
            while time.time() < timeout:
                try:
                    data = self.out_queue.get_nowait()
                    if isinstance(data, dict) and data.get('data_type') == 'audio_level':
                        print(f"Level message: {data}")
                        messages.append(data)
                except Empty:
                    if messages:  # Выходим если уже собрали сообщения
                        break
                    time.sleep(0.1)

            # Проверяем наличие сообщений
            self.assertGreater(len(messages), 0, "Должны быть сообщения с уровнем сигнала")

            # Анализируем уровни сигнала
            levels = []
            for msg in messages:
                if isinstance(msg, dict):
                    if msg.get('data_type') == 'audio_level':
                        levels.append(msg['payload']['level'])
                elif isinstance(msg, PipelineMessage):
                    if msg.content.get('data_type') == 'audio_level':
                        levels.append(msg.content['payload']['level'])

            # Разделяем сообщения на три части (по секундам)
            msg_per_sec = max(len(levels) // 3, 1)  # Защита от деления на 0
            
            # Первая секунда - громкий сигнал (>80%)
            high_levels = [l for l in levels[:msg_per_sec] if l > 80]
            self.assertGreater(len(high_levels), 0, "Должны быть высокие уровни в первой секунде")

            # Вторая секунда - тишина (<1)
            silence = [l for l in levels[msg_per_sec:2*msg_per_sec] if l == 0]
            self.assertGreater(len(silence), 0, "Должна быть тишина во второй секунде")

            # Третья секунда - средний уровень
            mid_levels = [l for l in levels[2*msg_per_sec:] if 77 <= l <= 78]
            self.assertGreater(len(mid_levels), 0, "Должны быть средние уровни в третьей секунде")

            # Выводим статистику для отладки
            print(f"\nСтатистика обработки:")
            print(f"Всего сообщений: {len(messages)}")
            print(f"Высокие уровни: {len(high_levels)}")
            print(f"Периоды тишины: {len(silence)}")
            print(f"Средние уровни: {len(mid_levels)}")

        except Exception as e:
            print(f"Test error details: {str(e)}")
            self.fail(f"Тест завершился с ошибкой: {str(e)}")

    def test_vad_with_pauses(self):
        """Тест обработки VAD с паузами в сигнале."""
        try:
            # Генерируем тестовый сигнал с паузами
            audio_data = self.generate_test_signal()
            
            # Запускаем процессы
            self.src_stage.start()
            self.asr_stage.start()
            
            # Обрабатываем данные чанками
            chunk_size = 480  # 30мс при 16кГц
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                self.src_stage.process(chunk)
                time.sleep(0.02)

            # Собираем сообщения
            messages = self.collect_messages(timeout=2.0)
            
            # Анализируем сообщения
            voice_chunks = [msg for msg in messages 
                          if msg.type == MessageType.DATA 
                          and msg.content.get('data_type') == 'audio_chunk']
            
            # Проверяем распределение голосовых чанков
            chunks_per_second = len(voice_chunks) / 3  # Общее количество / длительность
            first_second = [msg for msg in voice_chunks 
                          if msg.content.get('timestamp', 0) < 1.0]
            second_second = [msg for msg in voice_chunks 
                           if 1.0 <= msg.content.get('timestamp', 0) < 2.0]
            third_second = [msg for msg in voice_chunks 
                          if msg.content.get('timestamp', 0) >= 2.0]
            
            # Проверки
            self.assertGreater(len(first_second), 0, "Должны быть обнаружены голосовые чанки в первой секунде")
            self.assertEqual(len(second_second), 0, "Не должно быть голосовых чанков во время паузы")
            self.assertGreater(len(third_second), 0, "Должны быть обнаружены голосовые чанки в третьей секунде")

        except Exception as e:
            self.fail(f"Тест завершился с ошибкой: {str(e)}")

    def test_level_meter_normalization(self):
        """Тест нормализации уровня сигнала."""
        try:
            # Генерируем тестовые сигналы разной амплитуды
            duration = 1.0
            sample_rate = 16000
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            
            test_amplitudes = {
                'full': 1.0,    # Полная шкала
                'half': 0.5,    # -6 дБ
                'quarter': 0.25,  # -12 дБ
                'silence': 0.0    # Тишина
            }
            
            for name, amplitude in test_amplitudes.items():
                audio_data = amplitude * np.sin(2 * np.pi * 440 * t).astype(np.float32)
                
                # Обрабатываем сигнал
                self.src_stage.process(audio_data)
                
                # Собираем сообщения
                messages = self.collect_messages(timeout=0.5)
                
                # Фильтруем сообщения об уровне
                level_messages = [
                    msg for msg in messages 
                    if msg.type == MessageType.DATA 
                    and msg.content.get('data_type') == 'audio_level'
                ]
                
                # Проверяем уровни
                if level_messages:
                    levels = [msg.content['payload']['level'] for msg in level_messages]
                    avg_level = np.mean(levels)
                    
                    if name == 'full':
                        self.assertGreater(avg_level, 90.0, "Полный уровень должен быть близок к 100")
                    elif name == 'silence':
                        self.assertEqual(avg_level, 0.0, "Тишина должна давать 0")
                    else:
                        self.assertGreater(avg_level, 0.0, f"Уровень для {name} должен быть больше 0")
                        self.assertLess(avg_level, 100.0, f"Уровень для {name} должен быть меньше 100")

        except Exception as e:
            self.fail(f"Тест завершился с ошибкой: {str(e)}")

    def test_level_update_rate(self):
        """Тест частоты обновления уровня сигнала."""
        try:
            # Генерируем длительный сигнал
            audio_data = self.generate_test_signal(duration=5.0)
            
            # Запускаем обработку
            start_time = time.time()
            chunk_size = 480
            
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                self.src_stage.process(chunk)
                time.sleep(0.01)
            
            # Собираем сообщения
            messages = self.collect_messages(timeout=1.0)
            
            # Фильтруем сообщения об уровне
            level_messages = [
                msg for msg in messages 
                if msg.type == MessageType.DATA 
                and msg.content.get('data_type') == 'audio_level'
            ]
            
            # Проверяем интервалы между сообщениями
            if len(level_messages) > 1:
                timestamps = [msg.content['payload']['timestamp'] for msg in level_messages]
                intervals = np.diff(timestamps)
                
                min_interval = np.min(intervals)
                self.assertGreaterEqual(
                    min_interval,
                    self.src_stage.min_level_update_interval,
                    "Минимальный интервал обновления уровня не соблюдается"
                )

        except Exception as e:
            self.fail(f"Тест завершился с ошибкой: {str(e)}")

    def test_vu_meter_message_rate(self):
        """Test that VU meter messages are sent at correct intervals."""
        try:
            # Генерируем тестовый сигнал
            sample_rate = 16000
            duration = 2  # 2 секунды
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            audio_data = 0.9 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

            # Запускаем процессы
            self.src_stage.start()
            self.asr_stage.start()

            # Обрабатываем данные небольшими чанками
            chunk_size = 480  # 30мс при 16кГц
            messages = []
            
            # Отправляем все чанки
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                self.src_stage.process(chunk)
                time.sleep(0.01)

            # Собираем сообщения
            time.sleep(1.0)
            while not self.out_queue.empty():
                msg = self.out_queue.get_nowait()
                if isinstance(msg, PipelineMessage) and msg.type == MessageType.DATA:
                    messages.append(msg)

            # Фильтруем сообщения об уровне
            level_messages = [
                msg for msg in messages 
                if msg.content.get('data_type') == 'audio_level'
            ]

            # Проверяем интервал между сообщениями
            if len(level_messages) > 1:
                timestamps = [
                    msg.content['payload']['timestamp'] 
                    for msg in level_messages
                ]
                intervals = np.diff(timestamps)
                min_interval = min(intervals)
                
                self.assertGreaterEqual(
                    min_interval,
                    self.src_stage.min_level_update_interval,
                    "Интервал между сообщениями слишком мал"
                )

            # Проверяем наличие сообщения о тишине в конце
            if level_messages:
                last_message = level_messages[-1]
                self.assertTrue(
                    last_message.content['payload'].get('is_silence', False),
                    "Последнее сообщение должно быть о тишине"
                )

        except Exception as e:
            self.fail(f"Тест завершился с ошибкой: {str(e)}")

    def tearDown(self):
        """Корректное завершение тестов."""
        try:
            # Останавливаем процессы
            self.stop_event.set()
            
            # Ожидаем завершения процессов
            for stage in [self.src_stage, self.asr_stage]:
                if hasattr(stage, '_process') and stage._process:
                    try:
                        stage._process.terminate()
                        stage._process.join(timeout=1.0)
                    except:
                        pass  # Игнорируем ошибки при остановке

            # Очищаем очереди
            for queue in [self.src_2_asr, self.out_queue]:
                try:
                    while not queue.empty():
                        queue.get_nowait()
                except:
                    pass  # Игнорируем ошибки при очистке очередей
                    
        except Exception as e:
            print(f"Ошибка при очистке ресурсов: {e}")

if __name__ == '__main__':
    unittest.main()
