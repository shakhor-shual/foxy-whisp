# foxy_whisp_gui.py
from multiprocessing import Process
from multiprocessing import Queue as MPQueue
import tkinter as tk
from tkinter import ttk, filedialog
import threading
import argparse
from logic.foxy_utils import add_shared_args, logger
import os
import signal
from foxy_whisp_server import FoxyWhispServer
from logic.foxy_message import PipelineMessage
from logic.log_filter import LogFilter
from logic.message_handler import MessageHandler
import queue
import time

class FoxyWhispGUI:
    def __init__(self, gui_to_server: MPQueue, server_to_gui: MPQueue, args, parser):
        self.gui_to_server = gui_to_server
        self.server_to_gui = server_to_gui
        self.args = args
        # Устанавливаем микрофон как источник по умолчанию
        self.args.listen = "audio_device"
        self.parser = parser

        self.root = tk.Tk()
        self.root.geometry("300x900")
        self.root.title("Foxy-Whisp")

        self.server_running = False
        self.recording = False
        self.advanced_options_visible = False
        self.widgets = {}
        self.audio_input = None
        self.message_queue = queue.Queue()  # Queue for safe thread communication
        self.log_filter = LogFilter()
        self.source_initialized = False  # Добавляем флаг инициализации источника
        self.source_state = {
            'active': False,
            'recording_state': 'stopped',
            'is_configured': False
        }
        self._fade_update_after = None  # Add initialization
        self.message_handler = MessageHandler(gui_to_server, server_to_gui)
        self._setup_message_handlers()

        self.setup_gui()
        self.start_queue_listener()

    def _setup_message_handlers(self):
        """Setup message handler callbacks"""
        self.message_handler.register_callback('log', self.append_text_safe)
        self.message_handler.register_callback('status', self.append_text_safe)
        self.message_handler.register_callback('vad', self._update_vad_status)
        self.message_handler.register_callback('audio_level', self.update_audio_level)
        self.message_handler.register_callback('button', self._handle_button_update)
        self.message_handler.register_callback('device_info', self._update_device_list)

    def _handle_button_update(self, data):
        """Handle button updates from message handler"""
        try:
            if not isinstance(data, dict):
                return
                
            button_name = data.get('button')
            config = data.get('config')
            
            if button_name and config and hasattr(self, button_name):
                button = getattr(self, button_name)
                self.message_queue.put(('update_button', (button, config)))
        except Exception as e:
            print(f"Button update error: {e}")

    def _update_device_list(self, device_info):
        """Handle device info updates"""
        if isinstance(device_info, dict) and 'devices' in device_info:
            devices = [f"{dev['name']} (ID: {dev['index']})" for dev in device_info['devices']]
            self.device_cb['values'] = devices

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
            text="Старт",
            command=self.toggle_server
        )
        self.server_btn.pack(side=tk.LEFT, fill=tk.X, padx=5, pady=5)

        # Audio source toggle - изменен текст кнопки и начальное состояние
        self.source_btn = ttk.Button(
            self.control_frame,
            text="Mic" if self.args.listen == "audio_device" else "TCP",
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
        """Request audio devices list from running SRCstage"""
        self.send_command("get_audio_devices")

    def toggle_server(self):
        """Toggle server state and update UI"""
        if self.server_running:
            self.send_command("stop")
            self.server_btn.configure(state='disabled')  # Блокируем кнопку при остановке
        else:
            self.server_btn.configure(state='disabled')  # Блокируем при запуске
            self.send_command("start", vars(self.args))

    def toggle_audio_source(self):
        """Switch between TCP and Audio Device sources"""
        # Сохраняем текущий источник для возможного отката
        previous_source = self.args.listen
        
        try:
            # Переключаем источник
            self.args.listen = "audio_device" if self.args.listen == "tcp" else "tcp"
            self.source_btn.config(text="Mic" if self.args.listen == "audio_device" else "TCP")
            
            # Обновляем состояние элементов управления
            self.update_controls_activity()
            
            if self.server_running:
                # Останавливаем запись если активна
                if self.recording:
                    self.send_command("stop_recording")
                    time.sleep(0.2)  # Даем время на остановку записи
                
                # Останавливаем текущий источник
                self.send_command("stop_stage", {"stage": "src"})
                
                # Обновляем параметры
                self.send_command("update_params", vars(self.args))
                
                # Запускаем источник с новыми параметрами
                time.sleep(0.5)  # Даем время на применение параметров
                self.send_command("start_stage", {"stage": "src"})
                
                self.append_text(f"[GUI.INFO] Switched audio source to {self.args.listen}")
            else:
                # Просто обновляем параметры если сервер не запущен
                self.send_command("update_params", vars(self.args))
                
        except Exception as e:
            # В случае ошибки возвращаем предыдущий источник
            self.args.listen = previous_source
            self.source_btn.config(text="Mic" if self.args.listen == "audio_device" else "TCP")
            self.update_controls_activity()
            self.append_text(f"[GUI.ERROR] Failed to switch audio source: {str(e)}")

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
        """Create audio level meter and VAD indicator"""
        self.level_frame = ttk.Frame(self.main_frame)
        self.level_frame.pack(fill=tk.X, pady=5)

        # Audio Level indicator
        self.level_label = ttk.Label(self.level_frame, text="Audio Level:")
        self.level_label.pack(side=tk.LEFT)

        self.level_bar = ttk.Progressbar(
            self.level_frame,
            orient=tk.HORIZONTAL,
            length=150,
            mode="determinate"
        )
        self.level_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # VAD Status indicator
        self.vad_label = ttk.Label(
            self.level_frame, 
            text="VAD",
            background='gray',  # Default color
            width=6,
            anchor='center'
        )
        self.vad_label.pack(side=tk.RIGHT, padx=5)

        # Add VAD fade time slider
        self.fade_frame = ttk.Frame(self.main_frame)
        self.fade_frame.pack(fill=tk.X, pady=5)

        ttk.Label(self.fade_frame, text="VAD Fade:").pack(side=tk.LEFT)
        
        # Round initial value to nearest 100ms
        initial_fade_time = round(self.args.vad_fade_time / 100) * 100
        self.fade_value = tk.StringVar(value=str(initial_fade_time))
        ttk.Label(self.fade_frame, textvariable=self.fade_value, width=4).pack(side=tk.RIGHT)
        
        self.fade_slider = ttk.Scale(
            self.fade_frame,
            from_=0,
            to=1000,
            orient=tk.HORIZONTAL,
            command=self.on_fade_change
        )
        self.fade_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        # Set initial value
        self.fade_slider.set(initial_fade_time)
        
        # Add delay to prevent too frequent updates
        self._fade_update_after = None

    def on_fade_change(self, value):
        """Handle fade slider changes with debouncing"""
        try:
            # Cancel previous update if exists
            if self._fade_update_after:
                self.root.after_cancel(self._fade_update_after)
            
            # Round to nearest 100ms
            fade_time = round(float(value) / 100) * 100
            # Update display immediately
            self.fade_value.set(str(int(fade_time)))
            
            # Schedule actual update with delay
            self._fade_update_after = self.root.after(
                200,  # 200ms delay for debouncing
                lambda: self._send_fade_update(fade_time)
            )
        except ValueError:
            pass

    def _send_fade_update(self, fade_time: int):
        """Actually send the fade time update command"""
        self._fade_update_after = None
        self.send_command("update_vad_fade_time", {"fade_time_ms": fade_time})

    def update_audio_level(self, level_data):
        """Update audio level meter.
        Args:
            level_data: dict with keys 'level', 'timestamp', 'is_silence'
        """
        try:
            if isinstance(level_data, dict):
                level = level_data.get('level', 0)
                if isinstance(level, (int, float)):
                    self.level_bar["value"] = level
        except Exception as e:
            print(f"[GUI.ERROR] Failed to update audio level: {e}")

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

        # Text display area - Create this BEFORE filter frame
        self.text = tk.Text(self.text_frame, wrap=tk.WORD, height=10)
        self.scroll = ttk.Scrollbar(
            self.text_frame,
            orient=tk.VERTICAL,
            command=self.text.yview
        )

        # Add filter controls - NOW we can reference self.text
        self.filter_frame = ttk.Frame(self.text_frame)
        self.filter_frame.pack(fill=tk.X, pady=5)

        # Source filter with tooltip
        self.source_var = tk.StringVar(value="all")
        sources_frame = ttk.Frame(self.filter_frame)
        sources_frame.pack(side=tk.LEFT, padx=2)
        
        sources = ttk.Combobox(sources_frame, textvariable=self.source_var, 
                             values=["all", "server", "src", "asr", "test"])
        sources.pack(side=tk.LEFT)
        sources.bind("<<ComboboxSelected>>", self.update_filters)
        
        # Добавляем подсказку для источников
        self.create_tooltip(sources, "server: Server messages\n" +
                                   "src: Source stage messages (srcstage)\n" +
                                   "asr: ASR stage messages (asrstage)\n" +
                                   "test: Test messages")

        # Level filter
        self.level_var = tk.StringVar(value="all")
        levels = ttk.Combobox(self.filter_frame, textvariable=self.level_var,
                            values=["all", "debug", "info", "warning", "error"])
        levels.pack(side=tk.LEFT, padx=2)
        levels.bind("<<ComboboxSelected>>", self.update_filters)

        # Pattern filter
        self.pattern_var = tk.StringVar()
        pattern_entry = ttk.Entry(self.filter_frame, textvariable=self.pattern_var)
        pattern_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        pattern_entry.bind("<Return>", self.update_filters)

        # Clear filters button
        clear_btn = ttk.Button(self.filter_frame, text="Clear Filters", 
                             command=self.clear_filters)
        clear_btn.pack(side=tk.RIGHT, padx=2)

        # Now pack text and scrollbar AFTER filter frame
        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.text.config(yscrollcommand=self.scroll.set)

    def update_filters(self, event=None):
        """Update log filters and reapply to existing text"""
        self.log_filter.clear_all_filters()

        source = self.source_var.get()
        if source != "all":
            self.log_filter.add_source_filter(source)

        level = self.level_var.get()
        if level != "all":
            self.log_filter.add_level_filter(level)

        pattern = self.pattern_var.get().strip()
        if pattern:
            self.log_filter.add_pattern_filter(pattern)

        # Reapply filters to existing text
        self.reapply_filters()

    def clear_filters(self):
        """Clear all filters and restore all text"""
        self.source_var.set("all")
        self.level_var.set("all")
        self.pattern_var.set("")
        self.log_filter.clear_all_filters()
        self.reapply_filters()

    def reapply_filters(self):
        """Reapply filters to existing text"""
        # Save current text
        current_text = self.text.get(1.0, tk.END)
        self.text.config(state=tk.NORMAL)
        self.text.delete(1.0, tk.END)

        # Reapply each line with filters
        for line in current_text.split('\n'):
            if line.strip():
                try:
                    # Parse log line
                    parts = line.split(']', 1)
                    if len(parts) == 2:
                        source = parts[0].strip('[')
                        message = parts[1].strip()
                        level = "info"  # Default level
                        
                        # Нормализация source и level
                        if '.' in source:
                            source_parts = source.split('.')
                            source = source_parts[0]  # Берем основной источник
                            level = source_parts[-1].lower()  # Уровень всегда последний

                        # Apply filters
                        if self.log_filter.matches(source, level, message):
                            self.text.insert(tk.END, line + '\n')
                except Exception as e:
                    print(f"[GUI] Reapply filter error: {e}")  # Для отладки
                    # If parsing fails, just add the line
                    self.text.insert(tk.END, line + '\n')

        self.text.config(state=tk.DISABLED)
        self.text.see(tk.END)

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
        if not self.message_handler.send_command(command, params):
            self.append_text(f"[ERROR] Failed to send command: {command}")

    def toggle_recording(self):
        """Toggle audio recording state"""
        # Проверяем текущее состояние кнопки
        if self.record_btn['state'] == 'disabled':
            return
            
        if self.recording:
            self.record_btn.config(state='disabled', text="Stopping...")
            self.send_command("stop_recording")
        else:
            self.record_btn.config(state='disabled', text="Starting...")
            self.send_command("start_recording")

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
        """Enhanced window close handler with timeout"""
        try:
            if self.server_running:
                # Signal server to stop
                self.send_command("shutdown")
                
                # Wait briefly for server to acknowledge
                wait_start = time.time()
                while self.server_running and time.time() - wait_start < 2.0:
                    self.root.update()
                    time.sleep(0.1)
                
                # If still running, force close
                if self.server_running:
                    self.append_text("[WARNING] Server not responding, forcing shutdown")
            
            # Schedule destroy after pending events
            self.root.after(100, self._force_close)
            
        except Exception as e:
            print(f"Error during close: {e}")
            self.root.destroy()

    def _force_close(self):
        """Force close the application"""
        try:
            self.root.destroy()
        except:
            pass
        finally:
            # Force exit if needed
            os._exit(0)

    def append_text(self, text: str):
        """Append text with improved message parsing"""
        try:
            self.text.config(state=tk.NORMAL)
            self.text.insert(tk.END, text + "\n")
            self.text.see(tk.END)
            self.text.config(state=tk.DISABLED)
            
        except Exception as e:
            print(f"[GUI] Message parsing error: {e}, Text: {text}")

    def append_text_safe(self, text: str):
        """Thread-safe version of append_text"""
        self.message_queue.put(('append_text', text))

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
                    self.message_handler.handle_message(msg)
            except Exception as e:
                print(f"[GUI] Queue error: {e}")
                break

    def process_message_queue(self):
        """Process pending messages in the queue"""
        try:
            while True:
                try:
                    message = self.message_queue.get_nowait()
                    if isinstance(message, tuple):
                        action, args = message
                        if action == 'append_text':
                            self.append_text(args)
                        elif action == 'update_audio':
                            self.update_audio_level(args)
                        elif action == 'update_button':
                            button, config = args
                            if hasattr(button, 'configure'):
                                button.configure(**config)
                        elif action == 'update_vad':
                            current_color = self.vad_label.cget('background')
                            target_color = '#00ff00' if args else '#ff0000'
                            self._animate_color_change(current_color, target_color)
                        self.root.update()
                    self.message_queue.task_done()
                except queue.Empty:
                    break
        finally:
            if self.root:
                self.root.after(5, self.process_message_queue)

    def _animate_color_change(self, start_color, end_color, steps=10):
        """Плавное изменение цвета индикатора"""
        if start_color == end_color:
            return
            
        def hex_to_rgb(color):
            """Convert color to RGB values"""
            try:
                # Handle Tcl_Obj color names
                if not isinstance(color, str) or not color.startswith('#'):
                    # Get RGB from Tkinter
                    rgb = self.vad_label.winfo_rgb(color)
                    # Convert to hex format
                    return tuple(c//256 for c in rgb)
                # Handle hex colors
                color = color.lstrip('#')
                return tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
            except Exception as e:
                print(f"Color conversion error: {e}")
                return (0, 0, 0)

        def rgb_to_hex(rgb):
            """Convert RGB tuple to hex color"""
            return '#{:02x}{:02x}{:02x}'.format(*rgb)

        def interpolate_colors(start_rgb, end_rgb, step, total):
            """Интерполяция между RGB цветами"""
            return tuple(
                int(start + (float(step) / total) * (end - start))
                for start, end in zip(start_rgb, end_rgb)
            )

        # Convert colors to RGB
        start_rgb = hex_to_rgb(start_color)
        end_rgb = hex_to_rgb(end_color)

        def animate(step=0):
            if step <= steps:
                try:
                    # Interpolate and convert to hex
                    color = rgb_to_hex(interpolate_colors(start_rgb, end_rgb, step, steps))
                    self.vad_label.configure(background=color)
                    self.root.after(20, lambda: animate(step + 1))
                except Exception as e:
                    logger.error(f"Animation error: {str(e)}", exc_info=True)
            
        animate()

    def _update_vad_status(self, is_active: bool):
        """Direct VAD status update"""
        try:
            current = self.vad_label.cget('background')
            target = '#00ff00' if is_active else '#ff0000'
            if str(current) != str(target):  # Convert both to string for comparison
                self._animate_color_change(current, target)
        except Exception as e:
            logger.error(f"VAD update error: {str(e)}", exc_info=True)

    def run(self):
        """Run the GUI main loop"""
        # Start processing messages from queue
        self.process_message_queue()
        self.root.mainloop()


def main():
    """Enhanced main with proper cleanup"""
    parser = argparse.ArgumentParser()
    add_shared_args(parser)
    args = parser.parse_args()

    # Create queues
    gui_to_server = MPQueue()
    server_to_gui = MPQueue()
    server_proc = None

    def cleanup():
        """Cleanup function for safe shutdown"""
        nonlocal server_proc
        try:
            # Send shutdown command if queue is still open
            if gui_to_server and not gui_to_server._closed:
                gui_to_server.put({'type': 'command', 'command': 'shutdown'})
                
            if server_proc:
                # Wait briefly for graceful shutdown
                server_proc.join(timeout=2.0)
                
                # Force terminate if still alive
                if server_proc.is_alive():
                    server_proc.terminate()
                    server_proc.join(timeout=1.0)
                    
                    # Kill if still not responding
                    if server_proc.is_alive():
                        os.kill(server_proc.pid, signal.SIGKILL)
                
            # Close queues safely
            for q in [gui_to_server, server_to_gui]:
                if q and not q._closed:
                    q.close()
                    try:
                        q.join_thread()  # Add timeout if needed
                    except:
                        pass
                        
        except Exception as e:
            print(f"Error during cleanup: {e}")
            
        finally:
            # Force exit if needed
            os._exit(0)

    try:
        # Setup signal handlers
        signal.signal(signal.SIGINT, lambda sig, frame: cleanup())
        signal.signal(signal.SIGTERM, lambda sig, frame: cleanup())

        # Create and start server
        server = FoxyWhispServer(gui_to_server, server_to_gui, vars(args))
        server_proc = Process(target=server.run)
        server_proc.start()

        # Create and run GUI
        gui = FoxyWhispGUI(gui_to_server, server_to_gui, args, parser)
        gui.run()
        
    except Exception as e:
        print(f"Error in main: {e}")
        
    finally:
        cleanup()


if __name__ == "__main__":
    main()