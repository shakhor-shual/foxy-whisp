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
        self.src_stage.vad = WebRTCVAD(aggressiveness=3)  # Use WebRTC VAD for testing

    def test_fifo_buffer_integration(self):
        """Test that audio data is correctly added to the FIFO buffer and processed by VAD."""
        # Simulate audio data (1 second of silence at 16 kHz)
        audio_data = np.zeros(16000, dtype=np.float32)

        # Process the audio data
        self.src_stage.process(audio_data)

        # Check that no audio chunks are forwarded (silence is not detected as speech)
        self.assertTrue(self.audio_out.empty())

    def test_vad_voice_detection(self):
        """Test that VAD detects voice and forwards audio chunks."""
        # Simulate audio data with a simple sine wave (voice-like signal)
        sample_rate = 16000
        duration = 1  # 1 second
        frequency = 440  # 440 Hz (A4 note)
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        audio_data = (0.5 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)

        # Ensure the data length is a multiple of the frame size
        vad_frame_size = self.src_stage.vad.get_chunk_size()
        audio_data = audio_data[: len(audio_data) // vad_frame_size * vad_frame_size]

        # Process the audio data
        self.src_stage.process(audio_data)

        # Check that audio chunks are forwarded (voice is detected)
        self.assertFalse(self.audio_out.empty())
        forwarded_chunk = self.audio_out.get()
        self.assertEqual(len(forwarded_chunk), vad_frame_size)

    def test_stop_event(self):
        """Test that the SRC stage stops processing when the stop event is set."""
        self.stop_event.set()  # Trigger the stop event
        self.src_stage._run()  # Run the SRC stage

        # Ensure no audio data is processed or forwarded
        self.assertTrue(self.audio_out.empty())

    def test_resample_to_16k(self):
        """Test that audio data is correctly resampled to 16 kHz."""
        # Simulate audio data at 8 kHz
        input_sample_rate = 8000
        duration = 1  # 1 second
        t = np.linspace(0, duration, int(input_sample_rate * duration), endpoint=False)
        audio_data = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)  # 440 Hz sine wave

        # Update args to simulate 8 kHz input
        self.src_stage.args["sample_rate"] = input_sample_rate

        # Perform resampling
        resampled_data = self.src_stage._resample_to_16k(audio_data)

        # Check the length of the resampled data
        self.assertEqual(len(resampled_data), 16000)  # 1 second at 16 kHz
        self.assertTrue(np.all(np.isfinite(resampled_data)))  # Ensure no NaNs or Infs

if __name__ == "__main__":
    unittest.main()
