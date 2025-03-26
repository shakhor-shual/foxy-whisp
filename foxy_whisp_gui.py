# foxy_whisp_gui.py
from multiprocessing import Process
from multiprocessing import Queue as MPQueue
import tkinter as tk
from tkinter import ttk, filedialog
import threading
import argparse
from logic.foxy_utils import add_shared_args, logger
from logic.local_audio_input import LocalAudioInput
from foxy_whisp_server import FoxyWhispServer
from logic.foxy_message import PipelineMessage

class FoxyWhispGUI:
    def __init__(self, gui_to_server: MPQueue, server_to_gui: MPQueue, args, parser):
        self.gui_to_server = gui_to_server
        self.server_to_gui = server_to_gui
        self.args = args
        self.parser = parser

        self.root = tk.Tk()
        self.root.geometry("300x900")
        self.root.title("Foxy-Whisp")

        self.server_running = False
        self.recording = False
        self.advanced_options_visible = False
        self.widgets = {}
        self.audio_input = None

        self.setup_gui()
        self.start_queue_listener()

    def setup_gui(self):
        """Настройка основного интерфейса."""
        self.create_main_frame()
        self.create_control_buttons()
        self.create_audio_level_indicator()
        self.create_text_area()
        self.create_advanced_frame()
        self.create_apply_button()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.update_controls_activity()

    def create_main_frame(self):
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_control_buttons(self):
        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.pack(fill=tk.X, padx=5, pady=5)

        # Server control button
        self.server_btn = ttk.Button(
            self.control_frame,
            text="Start Server",
            command=self.toggle_server
        )
        self.server_btn.pack(side=tk.LEFT, fill=tk.X, padx=5, pady=5)

        # Audio source toggle
        self.source_btn = ttk.Button(
            self.control_frame,
            text="TCP" if self.args.listen == "tcp" else "Audio Device",
            command=self.toggle_audio_source
        )
        self.source_btn.pack(side=tk.LEFT, fill=tk.X, padx=5, pady=5)

        # Recording control
        self.record_btn = ttk.Button(
            self.control_frame,
            text="Start Recording",
            command=self.toggle_recording
        )
        self.record_btn.pack(side=tk.LEFT, fill=tk.X, padx=5, pady=5)

        # Audio device selection
        self.device_cb = ttk.Combobox(self.control_frame, state="readonly")
        self.device_cb.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        self.device_cb.bind("<<ComboboxSelected>>", self.on_device_change)
        self.update_audio_devices()

        # Advanced options toggle
        self.advanced_btn = ttk.Button(
            self.control_frame,
            text="Advanced",
            command=self.toggle_advanced
        )
        self.advanced_btn.pack(side=tk.RIGHT, fill=tk.X, padx=5, pady=5)

    def update_audio_devices(self):
        devices = LocalAudioInput.list_devices()
        input_devices = [f"{d['name']} (ID: {d['index']})" for d in devices if d["max_input_channels"] > 0]
        self.device_cb["values"] = input_devices

        if input_devices:
            default_device = LocalAudioInput.get_default_input_device()
            self.device_cb.set(f"{devices[default_device]['name']} (ID: {default_device})")
            self.args.audio_device = default_device

    def toggle_server(self):
        """Toggle server state and update UI"""
        if self.server_running:
            self.send_command("stop")
            # Не меняем текст кнопки здесь - дождемся ответа от сервера
        else:
            self.send_command("start", vars(self.args))
            # Не меняем текст кнопки здесь - дождемся ответа от сервера

    def toggle_audio_source(self):
        """Switch between TCP and Audio Device sources"""
        self.args.listen = "audio_device" if self.args.listen == "tcp" else "tcp"
        self.source_btn.config(text="Audio Device" if self.args.listen == "audio_device" else "TCP")
        self.update_controls_activity()
        self.send_command("update_params", vars(self.args))

    def update_controls_activity(self):
        """Update controls state based on audio source"""
        state = tk.NORMAL if self.args.listen == "audio_device" else tk.DISABLED
        self.record_btn.config(state=state)
        self.device_cb.config(state="readonly" if state == tk.NORMAL else tk.DISABLED)

    def on_device_change(self, event=None):
        """Handle audio device selection change"""
        selected = self.device_cb.get()
        if selected:
            device_id = int(selected.split("(ID: ")[1].rstrip(")"))
            self.args.audio_device = device_id
            self.send_command("update_params", vars(self.args))

    def create_audio_level_indicator(self):
        """Create audio level meter"""
        self.level_frame = ttk.Frame(self.main_frame)
        self.level_frame.pack(fill=tk.X, pady=5)

        self.level_label = ttk.Label(self.level_frame, text="Audio Level:")
        self.level_label.pack(side=tk.LEFT)

        self.level_bar = ttk.Progressbar(
            self.level_frame,
            orient=tk.HORIZONTAL,
            length=150,
            mode="determinate"
        )
        self.level_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def update_audio_level(self, level):
        """Update audio level meter"""
        self.level_bar["value"] = level

    def create_text_area(self):
        """Create text display area with controls"""
        self.text_frame = ttk.Frame(self.main_frame)
        self.text_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Toolbar buttons
        self.toolbar = ttk.Frame(self.text_frame)
        self.toolbar.pack(fill=tk.X, pady=5)

        self.clear_btn = tk.Button(
            self.toolbar,
            text="✗",
            font=("Monospace", 12, "bold"),
            command=self.clear_text
        )
        self.clear_btn.pack(side=tk.LEFT, fill=tk.X, padx=2)

        self.save_btn = tk.Button(
            self.toolbar,
            text="▽",
            font=("Monospace", 12, "bold"),
            command=self.save_text
        )
        self.save_btn.pack(side=tk.LEFT, fill=tk.X, padx=2)

        self.help_btn = tk.Button(
            self.toolbar,
            text="?",
            font=("Monospace", 12, "bold"),
            command=self.show_help
        )
        self.help_btn.pack(side=tk.RIGHT, fill=tk.X, padx=2)

        self.mute_btn = tk.Button(
            self.toolbar,
            text="▣",
            font=("Monospace", 12, "bold"),
            command=self.toggle_recording
        )
        self.mute_btn.pack(side=tk.RIGHT, fill=tk.X, padx=2)

        # Text display area
        self.text = tk.Text(self.text_frame, wrap=tk.WORD, height=10)
        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scroll = ttk.Scrollbar(
            self.text_frame,
            orient=tk.VERTICAL,
            command=self.text.yview
        )
        self.scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.text.config(yscrollcommand=self.scroll.set)

    def create_advanced_frame(self):
        """Create advanced options frame"""
        self.advanced_frame = ttk.Frame(self.main_frame)
        self.add_parameter_controls()

    def add_parameter_controls(self):
        """Add controls for all config parameters"""
        for action in self.parser._actions:
            if action.dest in ("help", "listen"):
                continue

            frame = ttk.Frame(self.advanced_frame)
            frame.pack(fill=tk.X, padx=5, pady=5)

            label = ttk.Label(frame, text=action.dest)
            label.pack(side=tk.LEFT)

            if action.help:
                self.create_tooltip(label, action.help)

            if action.choices:
                var = tk.StringVar(value=getattr(self.args, action.dest, action.default))
                cb = ttk.Combobox(frame, textvariable=var, values=action.choices)
                cb.pack(side=tk.RIGHT, fill=tk.X, expand=True)
                cb.bind("<<ComboboxSelected>>", self.on_parameter_change)
                self.widgets[action.dest] = (cb, var)
            elif action.type == bool or isinstance(action.default, bool):
                var = tk.BooleanVar(value=getattr(self.args, action.dest, action.default))
                cb = ttk.Checkbutton(frame, variable=var, command=self.on_parameter_change)
                cb.pack(side=tk.RIGHT)
                self.widgets[action.dest] = (cb, var)
            elif action.type in (float, int):
                var = tk.StringVar(value=str(getattr(self.args, action.dest, action.default)))
                entry = ttk.Entry(frame, textvariable=var)
                entry.pack(side=tk.RIGHT, fill=tk.X, expand=True)
                entry.bind("<KeyRelease>", self.on_parameter_change)
                self.widgets[action.dest] = (entry, var)
            else:
                var = tk.StringVar(value=getattr(self.args, action.dest, action.default))
                entry = ttk.Entry(frame, textvariable=var)
                entry.pack(side=tk.RIGHT, fill=tk.X, expand=True)
                entry.bind("<KeyRelease>", self.on_parameter_change)
                self.widgets[action.dest] = (entry, var)

    def create_tooltip(self, widget, text):
        """Create a tooltip for a widget"""
        tip = tk.Toplevel(widget)
        tip.withdraw()
        tip.overrideredirect(True)
        label = tk.Label(tip, text=text, background="yellow", relief="solid", borderwidth=1, padx=5, pady=3)
        label.pack()

        def enter(event):
            x, y, _, _ = widget.bbox("insert")
            x += widget.winfo_rootx() + 25
            y += widget.winfo_rooty() + 25
            tip.geometry(f"+{x}+{y}")
            tip.deiconify()

        def leave(event):
            tip.withdraw()

        widget.bind("<Enter>", enter)
        widget.bind("<Leave>", leave)

    def on_parameter_change(self, event=None):
        """Handle parameter changes in advanced options"""
        self.apply_btn.config(state=tk.NORMAL)

    def create_apply_button(self):
        """Create apply button for advanced options"""
        self.apply_btn = ttk.Button(
            self.advanced_frame,
            text="Apply",
            command=self.apply_changes,
            state=tk.DISABLED
        )
        self.apply_btn.pack(fill=tk.X, pady=5)

    def apply_changes(self):
        """Apply changes from advanced options"""
        if self.server_running:
            self.append_text("[WARNING] Cannot apply changes while server is running")
            return

        for param, (widget, var) in self.widgets.items():
            value = var.get()
            if isinstance(value, str) and value.strip() == "":
                value = None

            if isinstance(value, str) and value is not None:
                action = next((a for a in self.parser._actions if a.dest == param), None)
                if action and action.type:
                    try:
                        if action.type == bool:
                            value = bool(value)
                        else:
                            value = action.type(value)
                    except (ValueError, TypeError):
                        continue

            setattr(self.args, param, value)

        if not self.args.model:
            self.args.model = "large-v3-turbo"

        self.apply_btn.config(state=tk.DISABLED)
        self.append_text("[INFO] Configuration changes applied")
        self.send_command("update_params", vars(self.args))

    def send_command(self, command: str, params: dict = None):
        """Send command to server"""
        try:
            if self.gui_to_server is None:
                self.append_text("[ERROR] Server connection lost")
                return
                
            PipelineMessage.create_command(
                source='gui',
                command=command,
                **(params or {})
            ).send(self.gui_to_server)
        except EOFError:
            self.append_text("[ERROR] Server connection closed")
            self.gui_to_server = None
        except Exception as e:
            self.append_text(f"[ERROR] Failed to send command: {str(e)}")

    def toggle_recording(self):
        """Toggle audio recording state"""
        if self.recording:
            self.send_command("stop_recording")
            self.record_btn.config(text="Start Recording")
        else:
            self.send_command("start_recording")
            self.record_btn.config(text="Stop Recording")
        self.recording = not self.recording

    def toggle_advanced(self):
        """Toggle between basic and advanced views"""
        if self.advanced_options_visible:
            self.advanced_frame.pack_forget()
            self.text_frame.pack(fill=tk.BOTH, expand=True)
            self.advanced_btn.config(text="Advanced")
        else:
            self.text_frame.pack_forget()
            self.advanced_frame.pack(fill=tk.BOTH, expand=True)
            self.advanced_btn.config(text="Basic")
        self.advanced_options_visible = not self.advanced_options_visible

    def on_close(self):
        """Handle window close event"""
        if self.server_running:
            self.send_command("stop")
        self.root.destroy()

    def append_text(self, text: str):
        """Append text to the display area"""
        self.text.config(state=tk.NORMAL)
        self.text.insert(tk.END, text + "\n")
        self.text.see(tk.END)
        self.text.config(state=tk.DISABLED)

    def clear_text(self):
        """Clear the text display area"""
        self.text.config(state=tk.NORMAL)
        self.text.delete(1.0, tk.END)
        self.text.config(state=tk.DISABLED)

    def save_text(self):
        """Save text content to file"""
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.text.get(1.0, tk.END))
            self.append_text(f"[INFO] Text saved to {path}")

    def show_help(self):
        """Show help information"""
        self.append_text("[HELP] Foxy-Whisp GUI\n"
                       "Start/Stop Server - Control the processing pipeline\n"
                       "TCP/Audio Device - Switch audio source\n"
                       "Start/Stop Recording - Capture audio from device")

    def start_queue_listener(self):
        """Start thread to listen for server messages"""
        threading.Thread(target=self.listen_for_messages, daemon=True).start()

    def listen_for_messages(self):
        """Listen for messages from server"""
        while True:
            try:
                if self.server_to_gui is None:
                    print("[GUI] No server queue available")
                    break
                    
                msg = PipelineMessage.receive(self.server_to_gui, timeout=0.1)
                if msg:
                    print(f"[GUI] Raw message received: {msg.__dict__}")  # Отладочный вывод
                    # Добавляем прямой вывод для тестовых сообщений
                    if msg.is_log() and msg.source == 'test':
                        print(f"[GUI] Test message received: {msg.content['message']}")
                    self.handle_server_message(msg)
            except EOFError:
                print("[GUI] Queue EOF received")
                break
            except Exception as e:
                print(f"[GUI] Queue error: {str(e)}")
                self.append_text(f"[ERROR] Message receive error: {str(e)}")
                break

    def handle_server_message(self, msg: PipelineMessage):
        """Handle messages from server"""
        try:
            print(f"[GUI] Handling message type: {msg.type}")  # Отладочный вывод
            if msg.is_log():
                message = msg.content.get('message', '')
                self.append_text(f"[{msg.source.upper()}] {message}")
                print(f"[GUI] Log message handled: {message}")  # Отладочный вывод
            elif msg.is_status():
                self.handle_status_message(msg)
            elif msg.is_data():
                self.handle_data_message(msg)
        except Exception as e:
            print(f"[GUI] Error handling message: {str(e)}")  # Отладочный вывод
            self.append_text(f"[ERROR] Failed to handle message: {str(e)}")

    def handle_status_message(self, msg: PipelineMessage):
        """Handle status updates from server"""
        status = msg.content.get('status')
        if status == 'server_started' or status == 'pipeline_started':
            self.server_running = True
            self.server_btn.config(text="Stop Server")
            self.append_text("[STATUS] Server started")
        elif status == 'server_stopped' or status == 'server_initialized' or status == 'pipeline_stopped':
            self.server_running = False
            self.server_btn.config(text="Start Server")
            self.append_text("[STATUS] Server stopped")
        elif status == 'shutdown':
            self.server_running = False
            self.server_btn.config(text="Start Server", state=tk.DISABLED)
            self.append_text("[STATUS] Server shutdown")
        elif status == 'recording_started':
            self.recording = True
            self.record_btn.config(text="Stop Recording")
            self.append_text("[STATUS] Recording started")
        elif status == 'recording_stopped':
            self.recording = False
            self.record_btn.config(text="Start Recording")
            self.append_text("[STATUS] Recording stopped")

    def handle_data_message(self, msg: PipelineMessage):
        """Handle data messages from server"""
        data_type = msg.content.get('data_type')
        payload = msg.content.get('payload')
        
        if data_type == 'transcription':
            self.append_text(f"TRANSCRIPT: {payload}")
        elif data_type == 'audio_level':
            self.update_audio_level(payload)
        elif data_type == 'log':  # Добавляем обработку тестовых лог-сообщений
            self.append_text(f"TEST: {payload}")

    def run(self):
        """Run the GUI main loop"""
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser()
    add_shared_args(parser)
    args = parser.parse_args()

    # Создаем очереди
    gui_to_server = MPQueue()
    server_to_gui = MPQueue()

    # Создаем и запускаем сервер
    server = FoxyWhispServer(gui_to_server, server_to_gui, vars(args))
    server_proc = Process(target=server.run)
    server_proc.start()

    # Создаем GUI
    gui = FoxyWhispGUI(gui_to_server, server_to_gui, args, parser)
    
    try:
        gui.run()
    finally:
        try:
            # Отправляем команду shutdown если очередь еще существует
            if gui_to_server is not None and not gui_to_server._closed:
                gui_to_server.put({'type': 'command', 'command': 'shutdown'})
            
            # Ждем завершения сервера
            server_proc.join(timeout=5.0)
            
            # Принудительное завершение если не ответил
            if server_proc.is_alive():
                server_proc.terminate()
                server_proc.join(1.0)
        finally:
            # Безопасное закрытие очередей
            for queue in [gui_to_server, server_to_gui]:
                try:
                    if queue is not None and not queue._closed:
                        queue.close()
                        queue.join_thread()
                except:
                    pass

if __name__ == "__main__":
    main()