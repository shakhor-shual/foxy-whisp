# foxy_whisp_gui.py
import tkinter as tk
from tkinter import ttk, filedialog
from multiprocessing import Process, Queue
import threading
import argparse
from logic.foxy_utils import add_shared_args, logger
from logic.local_audio_input import LocalAudioInput
from foxy_whisp_server import FoxyWhispServer
from logic.foxy_message import PipelineMessage


class FoxyWhispGUI:
    def __init__(self, gui_to_manager_queue, manager_to_gui_queue, args, parser):
        self.gui_to_manager_queue = gui_to_manager_queue
        self.manager_to_gui_queue = manager_to_gui_queue
        self.args = args
        self.parser = parser  # Сохраняем parser как атрибут    

        self.root = tk.Tk()
        self.root.geometry("300x900")
        self.root.title("Foxy-Whisp")

        self.server_running = False
        self.recording = False
        self.advanced_options_visible = False
        self.widgets = {}
        self.audio_input = None

        self.setup_gui()  # Исправлено на setup_gui (было setup_gui)
        self.start_queue_listener()

    def run(self):
        """Запуск основного цикла обработки событий tkinter."""
        self.root.mainloop()

    def setup_gui(self):
        """Настройка основного интерфейса."""
        self.create_main_frame()
        self.create_control_buttons()
        self.create_audio_level_indicator()
        self.create_text_area()
        self.create_advanced_frame()
        self.create_apply_button()
        self.root.protocol("WM_DELETE_WINDOW", self.gui_on_close)
        self.update_controls_activity()

    def create_main_frame(self):
        """Создание основного контейнера."""
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_control_buttons(self):
        """Создание кнопок управления и элементов выбора аудиоисточника."""
        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.pack(fill=tk.X, padx=5, pady=5)

        # Кнопка запуска/остановки сервера
        self.start_stop_button = ttk.Button(self.control_frame, text="Start Server", command=self.gui_toggle_server)
        self.start_stop_button.pack(side=tk.LEFT, fill=tk.X, padx=5, pady=5)

        # Кнопка переключения источника аудио (TCP/Audio Device)
        self.source_button = ttk.Button(self.control_frame, text="TCP", command=self.gui_toggle_audio_source)
        self.source_button.pack(side=tk.LEFT, fill=tk.X, padx=5, pady=5)

        # Кнопка включения/выключения записи аудио
        self.record_button = ttk.Button(self.control_frame, text="Start Recording", command=self.gui_toggle_recording)
        self.record_button.pack(side=tk.LEFT, fill=tk.X, padx=5, pady=5)

        # Комбобокс выбора аудиоустройства
        self.device_combobox = ttk.Combobox(self.control_frame, state="readonly")
        self.device_combobox.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        self.device_combobox.bind("<<ComboboxSelected>>", self.on_audio_device_change)

        # Обновляем список устройств
        self.update_audio_devices()

        # Кнопка перехода к расширенным настройкам
        self.advanced_button = ttk.Button(self.control_frame, text="To Advanced", command=self.gui_toggle_advanced)
        self.advanced_button.pack(side=tk.RIGHT, fill=tk.X, padx=5, pady=5)

    def update_audio_devices(self):
        """Обновляет список доступных аудиоустройств."""
        devices = LocalAudioInput.list_devices()
        input_devices = [f"{d['name']} (ID: {d['index']})" for d in devices if d["max_input_channels"] > 0]
        self.device_combobox["values"] = input_devices

        if input_devices:
            default_device = LocalAudioInput.get_default_input_device()
            self.device_combobox.set(f"{devices[default_device]['name']} (ID: {default_device})")
            self.args.audio_device = default_device

    def gui_toggle_audio_source(self):
        """Переключение между TCP и Audio Device."""
        if self.args.listen == "tcp":
            self.args.listen = "audio_device"
            self.source_button.config(text="Audio Device")
            self.update_audio_devices()
        else:
            self.args.listen = "tcp"
            self.source_button.config(text="TCP")

        self.update_controls_activity()
        self.gui_apply_changes()

    def update_controls_activity(self):
        """Обновление активности элементов в зависимости от выбранного источника аудио."""
        if self.args.listen == "tcp":
            self.record_button.config(state=tk.DISABLED)
            self.device_combobox.config(state=tk.DISABLED)
        else:
            self.record_button.config(state=tk.NORMAL)
            self.device_combobox.config(state="readonly")

    def on_audio_device_change(self, event=None):
        """Обработка изменения выбранного аудиоустройства."""
        selected_device = self.device_combobox.get()
        if selected_device:
            device_id = int(selected_device.split("(ID: ")[1].rstrip(")"))
            self.args.audio_device = device_id
            self.gui_apply_changes()

    def create_audio_level_indicator(self):
        """Создание индикатора уровня аудиосигнала."""
        self.audio_level_frame = ttk.Frame(self.main_frame)
        self.audio_level_frame.pack(fill=tk.X, pady=5)

        self.audio_level_label = ttk.Label(self.audio_level_frame, text="Audio Level:")
        self.audio_level_label.pack(side=tk.LEFT)

        self.audio_level_bar = ttk.Progressbar(self.audio_level_frame, orient=tk.HORIZONTAL, length=150, mode="determinate")
        self.audio_level_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def update_audio_level(self, level):
        """Обновление индикатора уровня аудиосигнала."""
        self.audio_level_bar["value"] = level
        self.audio_level_bar.update()

    def create_text_area(self):
        """Создание текстового поля и кнопок управления."""
        self.text_frame = ttk.Frame(self.main_frame)
        self.text_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.button_frame = ttk.Frame(self.text_frame)
        self.button_frame.pack(fill=tk.X, pady=5)

        self.clear_btn = tk.Button(self.button_frame, text="✗", font=("Monospace", 12, "bold"), command=self.gui_clear_text)
        self.clear_btn.pack(side=tk.LEFT, fill=tk.X, padx=2)

        self.save_btn = tk.Button(self.button_frame, text="▽", font=("Monospace", 12, "bold"), command=self.gui_save_text)
        self.save_btn.pack(side=tk.LEFT, fill=tk.X, padx=2)

        self.ask_btn = tk.Button(self.button_frame, text="?", font=("Monospace", 12, "bold"), command=self.gui_show_help)
        self.ask_btn.pack(side=tk.RIGHT, fill=tk.X, padx=2)

        self.mute_btn = tk.Button(self.button_frame, text="▣", font=("Monospace", 12, "bold"), command=self.gui_toggle_recording)
        self.mute_btn.pack(side=tk.RIGHT, fill=tk.X, padx=2)

        self.text_area = tk.Text(self.text_frame, wrap=tk.WORD, height=10)
        self.text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scrollbar = ttk.Scrollbar(self.text_frame, orient=tk.VERTICAL, command=self.text_area.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_area.config(yscrollcommand=self.scrollbar.set)

    def create_advanced_frame(self):
        """Создание фрейма для расширенных настроек."""
        self.advanced_frame = ttk.Frame(self.main_frame)
        self.add_parameter_controls()

    def add_parameter_controls(self):
        """Добавление элементов управления для параметров."""
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
                combobox = ttk.Combobox(frame, textvariable=var, values=action.choices)
                combobox.pack(side=tk.RIGHT, fill=tk.X, expand=True)
                combobox.bind("<<ComboboxSelected>>", self.on_parameter_change)
                self.widgets[action.dest] = (combobox, var)
            elif action.type == bool or isinstance(action.default, bool):
                var = tk.BooleanVar(value=getattr(self.args, action.dest, action.default))
                checkbutton = ttk.Checkbutton(frame, variable=var, command=self.on_parameter_change)
                checkbutton.pack(side=tk.RIGHT)
                self.widgets[action.dest] = (checkbutton, var)
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
        """Создание всплывающей подсказки."""
        tooltip = tk.Toplevel(widget)
        tooltip.withdraw()
        tooltip.overrideredirect(True)
        label = tk.Label(tooltip, text=text, background="yellow", relief="solid", borderwidth=1, padx=5, pady=3)
        label.pack()

        def on_enter(event):
            x, y, _, _ = widget.bbox("insert")
            x += widget.winfo_rootx() + 25
            y += widget.winfo_rooty() + 25
            tooltip.geometry(f"+{x}+{y}")
            tooltip.deiconify()

        def on_leave(event):
            tooltip.withdraw()

        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)

    def on_parameter_change(self, event=None):
        """Обработка изменений параметров."""
        self.apply_button.config(state=tk.NORMAL)

    def create_apply_button(self):
        """Создание кнопки 'Apply'."""
        self.apply_button = ttk.Button(self.advanced_frame, text="Apply", command=self.gui_apply_changes, state=tk.DISABLED)
        self.apply_button.pack(fill=tk.X, pady=5)

    def gui_apply_changes(self):
        """Применение изменений параметров."""
        if self.server_running:
            logger.warning("Cannot apply changes while server is running.")
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

        if hasattr(self, 'apply_button'):
            self.apply_button.config(state=tk.DISABLED)

        logger.info("Changes applied successfully.")
        self.send_command("update_params", vars(self.args))

    def send_command(self, action: str, params: dict = None):
        """Отправка команды серверу через PipelineMessage"""
        params = params or {}
        PipelineMessage.create_command(
            source='gui',
            command=action,
            **params
        ).send(self.gui_to_manager_queue)

    def gui_toggle_server(self):
        """Отправка команды на запуск/остановку сервера."""
        if self.server_running:
            self.send_command("stop_server")
        else:
            self.send_command("start_server", vars(self.args))

    def gui_toggle_recording(self):
        """Переключение состояния записи аудио."""
        if self.recording:
            self.send_command("stop_recording")
        else:
            self.send_command("start_recording")

    def gui_toggle_advanced(self):
        """Переключение между основным интерфейсом и расширенными настройками."""
        if self.advanced_options_visible:
            self.advanced_frame.pack_forget()
            self.advanced_options_visible = False
            self.text_frame.pack(fill=tk.BOTH, expand=True)
            self.advanced_button.config(text="To Advanced")
        else:
            self.text_frame.pack_forget()
            self.advanced_options_visible = True
            self.advanced_frame.pack(fill=tk.BOTH, expand=True)
            self.advanced_button.config(text="To Transcription")

    def gui_on_close(self):
        """Обработка закрытия окна."""
        if self.server_running:
            self.send_command("stop_server")
        self.root.destroy()

    def append_text(self, text):
        """Добавление текста в текстовое поле."""
        self.text_area.config(state=tk.NORMAL)
        self.text_area.insert(tk.END, text + "\n")
        self.text_area.see(tk.END)
        self.text_area.config(state=tk.DISABLED)

    def gui_clear_text(self):
        """Очистка текстового поля."""
        self.text_area.config(state=tk.NORMAL)
        self.text_area.delete(1.0, tk.END)
        self.text_area.config(state=tk.DISABLED)

    def gui_save_text(self):
        """Сохранение содержимого текстового поля в файл."""
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if file_path:
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(self.text_area.get(1.0, tk.END))

    def gui_show_help(self):
        """Заглушка для кнопки помощи."""
        print("Help button clicked")

    def start_queue_listener(self):
        """Запуск потока для получения данных от FoxyManager."""
        threading.Thread(target=self.listen_for_updates, daemon=True).start()

    def listen_for_updates(self):
        """Получение данных от FoxyManager и обновление GUI."""
        while True:
            message = PipelineMessage.receive(self.manager_to_gui_queue)
            if message:
                self.handle_server_message(message)

    def handle_server_message(self, message: PipelineMessage):
        """Обработка сообщений от сервера"""
        try:
            if message.is_log():
                log_level = message.content['level'].upper()
                log_msg = f"[{log_level}] {message.content['message']}"
                self.append_text(log_msg)
                
            elif message.is_status():
                status = message.get_status()
                if status == 'pipeline_started':
                    self.server_running = True
                    self.start_stop_button.config(text="Stop Server")
                    self.append_text("[STATUS] Server started")
                elif status == 'pipeline_stopped':
                    self.server_running = False
                    self.start_stop_button.config(text="Start Server")
                    self.append_text("[STATUS] Server stopped")
                    
            elif message.is_data():
                data_type = message.get_data_type()
                if data_type == 'transcription':
                    self.append_text(f"Transcript: {message.content['payload']}")
                elif data_type == 'audio_level':
                    self.update_audio_level(message.content['payload'])
                    
        except Exception as e:
            self.append_text(f"[ERROR] Failed to handle message: {str(e)}")
    # def handle_server_message(self, message: PipelineMessage):
    #     """Обработка сообщений от сервера."""
    #     try:
    #         if message.is_log():
    #             self.append_text(f"[{message.source.upper()}] {message.content['message']}")
    #         elif message.is_status():
    #             self.handle_status(message)
    #         elif message.is_data():
    #             self.handle_data(message)
    #     except Exception as e:
    #         logger.error(f"Error handling message: {e}")

    def handle_status(self, message: PipelineMessage):
        """Обработка статусных сообщений."""
        status = message.get_status()
        details = message.content.get('details', {})
        
        if status == 'pipeline_started':
            self.server_running = True
            self.start_stop_button.config(text="Stop Server")
        elif status == 'pipeline_stopped':
            self.server_running = False
            self.start_stop_button.config(text="Start Server")
        elif status == 'recording_started':
            self.recording = True
            self.record_button.config(text="Stop Recording")
        elif status == 'recording_stopped':
            self.recording = False
            self.record_button.config(text="Start Recording")

    def handle_data(self, message: PipelineMessage):
        """Обработка данных от сервера."""
        data_type = message.get_data_type()
        payload = message.content.get('payload')
        
        if data_type == 'transcription':
            self.append_text(f"[TRANSCRIPTION] {payload}")
        elif data_type == 'audio_level':
            self.update_audio_level(payload)


def main():
    parser = argparse.ArgumentParser()
    add_shared_args(parser)
    args = parser.parse_args()

    if args.gui:
        gui_to_manager = Queue()
        manager_to_gui = Queue()

        # Преобразуем args в словарь перед передачей
        args_dict = vars(args)
        manager_proc = Process(
            target=FoxyWhispServer,
            args=(gui_to_manager, manager_to_gui, args_dict)
        )
        manager_proc.start()

        gui = FoxyWhispGUI(gui_to_manager, manager_to_gui, args, parser)
        gui.run()

        manager_proc.join()
    else:
        print("GUI mode is disabled. Use FoxyManager directly.")

if __name__ == "__main__":
    main()