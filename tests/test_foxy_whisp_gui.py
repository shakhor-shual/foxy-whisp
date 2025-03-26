import unittest
from unittest.mock import Mock, patch, MagicMock
import tkinter as tk
import argparse
from foxy_whisp_gui import FoxyWhispGUI
from logic.foxy_message import PipelineMessage

class TestFoxyWhispGUI(unittest.TestCase):
    def setUp(self):
        # Use MagicMock instead of real Queue
        self.gui_to_server = MagicMock()
        self.server_to_gui = MagicMock()
        self.parser = argparse.ArgumentParser()
        self.args = argparse.Namespace(
            listen="audio_device",
            model="large-v3",
            audio_device=0
        )
        
        # Patch tkinter components
        self.root_patcher = patch('tkinter.Tk')
        self.mock_tk = self.root_patcher.start()
        
        self.gui = FoxyWhispGUI(
            self.gui_to_server,
            self.server_to_gui,
            self.args,
            self.parser
        )

    def tearDown(self):
        self.root_patcher.stop()

    def test_init(self):
        """Test initialization of GUI"""
        self.assertFalse(self.gui.server_running)
        self.assertFalse(self.gui.recording)
        self.assertFalse(self.gui.advanced_options_visible)

    def test_toggle_server(self):
        """Test server toggle functionality"""
        # Test starting server
        self.gui.toggle_server()
        # Verify command was sent
        self.gui_to_server.put.assert_called_once()
        sent_msg = self.gui_to_server.put.call_args[0][0]
        self.assertEqual(sent_msg['type'], 'command')
        self.assertEqual(sent_msg['content']['command'], "start")
        
        # Reset mock and test stopping
        self.gui_to_server.reset_mock()
        self.gui.server_running = True
        self.gui.toggle_server()
        self.gui_to_server.put.assert_called_once()
        sent_msg = self.gui_to_server.put.call_args[0][0]
        self.assertEqual(sent_msg['type'], 'command')
        self.assertEqual(sent_msg['content']['command'], "stop")

    def test_toggle_audio_source(self):
        """Test audio source toggle"""
        initial_source = self.args.listen
        self.gui.toggle_audio_source()
        self.assertNotEqual(self.args.listen, initial_source)
        # Verify update_params command
        self.gui_to_server.put.assert_called_once()
        sent_msg = self.gui_to_server.put.call_args[0][0]
        self.assertEqual(sent_msg['type'], 'command')
        self.assertEqual(sent_msg['content']['command'], "update_params")

    def test_handle_server_message(self):
        """Test server message handling"""
        # Test status message
        status_msg = PipelineMessage.create_status("server", "server_started")
        self.gui.handle_server_message(status_msg)
        self.assertTrue(self.gui.server_running)

        # Test data message
        data_msg = PipelineMessage.create_data(
            "whisper",
            "transcription",
            "test transcription"
        )
        with patch.object(self.gui, 'append_text') as mock_append:
            self.gui.handle_server_message(data_msg)
            mock_append.assert_called_with("TRANSCRIPT: test transcription")

    def test_toggle_recording(self):
        """Test recording toggle"""
        self.gui.toggle_recording()
        # Verify start_recording command
        self.gui_to_server.put.assert_called_once()
        sent_msg = self.gui_to_server.put.call_args[0][0]
        self.assertEqual(sent_msg['type'], 'command')
        self.assertEqual(sent_msg['content']['command'], "start_recording")
        self.assertTrue(self.gui.recording)

        # Reset mock and test stop
        self.gui_to_server.reset_mock()
        self.gui.toggle_recording()
        self.gui_to_server.put.assert_called_once()
        sent_msg = self.gui_to_server.put.call_args[0][0]
        self.assertEqual(sent_msg['type'], 'command')
        self.assertEqual(sent_msg['content']['command'], "stop_recording")
        self.assertFalse(self.gui.recording)

    def test_append_text(self):
        """Test text append functionality"""
        test_text = "Test message"
        with patch.object(self.gui.text, 'insert') as mock_insert:
            self.gui.append_text(test_text)
            mock_insert.assert_called_with(tk.END, test_text + "\n")

    def test_clear_text(self):
        """Test text clear functionality"""
        with patch.object(self.gui.text, 'delete') as mock_delete:
            self.gui.clear_text()
            mock_delete.assert_called_with(1.0, tk.END)

if __name__ == '__main__':
    unittest.main()
