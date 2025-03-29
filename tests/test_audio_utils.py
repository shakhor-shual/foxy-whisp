import unittest
import numpy as np
from logic.audio_utils import calculate_vu_meter

class TestAudioUtils(unittest.TestCase):
    def test_silence(self):
        """Test silence returns 0."""
        silence = np.zeros(1000, dtype=np.float32)
        level = calculate_vu_meter(silence)
        self.assertEqual(level, 0.0)

    def test_full_scale(self):
        """Test full scale signal."""
        full_scale = np.ones(1000, dtype=np.float32)
        level = calculate_vu_meter(full_scale)
        self.assertEqual(level, 100.0)

    def test_minus_20db(self):
        """Test signal at -20 dB."""
        amplitude = 0.1  # -20 dB относительно полной шкалы
        signal = np.full(1000, amplitude, dtype=np.float32)
        level = calculate_vu_meter(signal)
        # Должно быть около 66.67% (для -20 дБ при шкале -60 дБ -> 0 дБ)
        self.assertAlmostEqual(level, 66.67, places=1)

    def test_very_quiet(self):
        """Test very quiet signal returns 0."""
        # Сигнал ниже -60 дБ
        very_quiet = np.full(1000, 0.0005, dtype=np.float32)
        level = calculate_vu_meter(very_quiet)
        self.assertEqual(level, 0.0)

    def test_empty_input(self):
        """Test empty input returns 0."""
        empty = np.array([], dtype=np.float32)
        level = calculate_vu_meter(empty)
        self.assertEqual(level, 0.0)

    def test_sine_wave(self):
        """Test typical sine wave."""
        t = np.linspace(0, 1, 1000)
        sine = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)  # -6 dB amplitude
        level = calculate_vu_meter(sine)
        # Для сигнала -6 дБ ожидаем около 90% шкалы
        self.assertGreater(level, 85.0)
        self.assertLess(level, 95.0)

if __name__ == '__main__':
    unittest.main()
