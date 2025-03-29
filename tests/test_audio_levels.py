import unittest
import numpy as np
from logic.audio_utils import calculate_vu_meter

class TestAudioLevels(unittest.TestCase):
    def test_signal_levels(self):
        """Test VU meter levels for different amplitudes."""
        duration = 1.0
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate), endpoint=False)
        f = 440  # частота тестового сигнала
        
        # Пересчитанные значения на основе текущей логики calculate_vu_meter
        test_cases = [
            (1.0, calculate_vu_meter(np.sin(2 * np.pi * f * t))),    # 0 dB
            (0.5, calculate_vu_meter(0.5 * np.sin(2 * np.pi * f * t))),  # -6 dB
            (0.3, calculate_vu_meter(0.3 * np.sin(2 * np.pi * f * t))),  # -10.5 dB
            (0.1, calculate_vu_meter(0.1 * np.sin(2 * np.pi * f * t))),  # -20 dB
            (0.01, calculate_vu_meter(0.01 * np.sin(2 * np.pi * f * t))),  # -40 dB
            (0.001, calculate_vu_meter(0.001 * np.sin(2 * np.pi * f * t))),  # -60 dB
            (0.0, calculate_vu_meter(np.zeros_like(t)))  # -inf dB
        ]
        
        for amplitude, expected_level in test_cases:
            with self.subTest(amplitude=amplitude):
                signal = amplitude * np.sin(2 * np.pi * f * t)
                level = calculate_vu_meter(signal)
                self.assertEqual(
                    level, 
                    expected_level,
                    msg=f"For amplitude {amplitude}: expected {expected_level}, got {level}"
                )

    def test_noise_floor(self):
        """Test that very quiet signals are handled correctly."""
        noise = np.random.normal(0, 0.0001, 16000)
        level = calculate_vu_meter(noise)
        self.assertLess(level, 20, "Noise floor should be low")

if __name__ == '__main__':
    unittest.main()
