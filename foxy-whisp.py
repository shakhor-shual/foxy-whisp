#!/usr/bin/env python3
from logic.foxy_config import *
from logic.foxy_utils import load_audio_chunk, add_shared_args, set_logging, logger, get_port_status, create_tokenizer
from logic.local_audio_input import LocalAudioInput
from logic.foxy_manager import FoxyManager

import argparse
import tkinter as tk
from tkinter import ttk, filedialog
import threading



class FoxyServerGUI:
    def __init__(self, root, parser, args):
        self.root = root
        self.parser = parser
        self.args = args
        self.server_thread = None
        self.server_running = False
        self.stop_event = threading.Event()
        self.advanced_options_visible = False
        self.widgets = {}
        self.audio_input = None  # Объект LocalAudioInput

        self.setup_gui()

    def setup_gui(self):
        """Настройка основного интерфейса."""
        self.root.geometry("300x700")
        self.root.title("Foxy-Whisp")

        self.create_main_frame()
        self.create_buttons()
        self.create_text_area()
        self.create_advanced_frame()
        self.create_audio_source_controls()
        self.create_apply_button()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_main_frame(self):
        """Создание основного контейнера."""
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_buttons(self):
        """Создание кнопок управления."""
        self.start_stop_button = ttk.Button(self.main_frame, text="Start Server", command=self.toggle_server)
        self.start_stop_button.pack(fill=tk.X, pady=5)

        self.advanced_button = ttk.Button(self.main_frame, text="To Advanced", command=self.toggle_advanced)
        self.advanced_button.pack(fill=tk.X, pady=5)

    def create_text_area(self):
        """Создание текстового поля и кнопок управления."""
        self.text_frame = ttk.Frame(self.main_frame)
        self.text_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.button_frame = ttk.Frame(self.text_frame)
        self.button_frame.pack(fill=tk.X, pady=5)

        self.clear_btn = tk.Button(self.button_frame, text="✗", font=("Monospace", 12, "bold"), command=self.clear_text)
        self.clear_btn.pack(side=tk.LEFT, fill=tk.X, padx=2)

        self.save_btn = tk.Button(self.button_frame, text="▽", font=("Monospace", 12, "bold"), command=self.save_text)
        self.save_btn.pack(side=tk.LEFT, fill=tk.X, padx=2)

        self.ask_btn = tk.Button(self.button_frame, text="?", font=("Monospace", 12, "bold"), command=self.show_help)
        self.ask_btn.pack(side=tk.RIGHT, fill=tk.X, padx=2)

        self.mute_btn = tk.Button(self.button_frame, text="▣", font=("Monospace", 12, "bold"), command=self.toggle_mute)
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

    def create_audio_source_controls(self):
        """Создание элементов управления для выбора источника аудио."""
        audio_frame = ttk.Frame(self.advanced_frame)
        audio_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(audio_frame, text="Audio Source:").pack(side=tk.LEFT)
        self.source_var = tk.StringVar(value=self.args.listen)
        self.source_combobox = ttk.Combobox(audio_frame, textvariable=self.source_var, values=["tcp", "audio_device"])
        self.source_combobox.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.source_combobox.bind("<<ComboboxSelected>>", self.on_audio_source_change)

        self.device_frame = ttk.Frame(self.advanced_frame)
        ttk.Label(self.device_frame, text="Audio Device:").pack(side=tk.LEFT)
        self.device_var = tk.StringVar()
        self.device_combobox = ttk.Combobox(self.device_frame, textvariable=self.device_var)
        self.device_combobox.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.device_combobox.bind("<<ComboboxSelected>>", self.on_audio_device_change)

        refresh_button = ttk.Button(self.device_frame, text="Refresh", command=self.update_audio_devices)
        refresh_button.pack(side=tk.RIGHT, padx=5)

        self.update_audio_devices()
        self.update_audio_device_visibility()

    def on_audio_source_change(self, event=None):
        """Обработка изменения выбранного источника аудио."""
        self.args.listen = self.source_var.get()
        self.update_audio_device_visibility()
        self.apply_changes()

    def update_audio_device_visibility(self):
        """Обновляет видимость элементов управления аудиоустройством."""
        if self.args.listen == "audio_device":
            self.device_frame.pack(fill=tk.X, padx=5, pady=5)
        else:
            self.device_frame.pack_forget()

    def update_audio_devices(self):
        """Обновляет список доступных аудиоустройств."""
        devices = LocalAudioInput.list_devices()
        input_devices = [f"{d['name']} (ID: {d['index']})" for d in devices if d["max_input_channels"] > 0]
        self.device_combobox["values"] = input_devices

        if not self.device_var.get() and input_devices:
            default_device = LocalAudioInput.get_default_input_device()
            self.device_var.set(f"{devices[default_device]['name']} (ID: {default_device})")
            self.args.audio_device = default_device

    def on_audio_device_change(self, event=None):
        """Обработка изменения выбранного аудиоустройства."""
        selected_device = self.device_var.get()
        if selected_device:
            device_id = int(selected_device.split("(ID: ")[1].rstrip(")"))
            self.args.audio_device = device_id
            self.apply_changes()

    def create_apply_button(self):
        """Создание кнопки 'Apply'."""
        self.apply_button = ttk.Button(self.advanced_frame, text="Apply", command=self.apply_changes, state=tk.DISABLED)
        self.apply_button.pack(fill=tk.X, pady=5)

    def add_parameter_controls(self):
        """Добавление элементов управления для параметров."""
        for action in self.parser._actions:
            if action.dest == "help":
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

    def apply_changes(self):
        """Применение изменений параметров."""
        if self.server_running:
            logger.warning("Cannot apply changes while server is running.")
            return

        self.args.listen = self.source_var.get()
        if self.args.listen == "audio_device":
            selected_device = self.device_var.get()
            if selected_device:
                device_id = int(selected_device.split("(ID: ")[1].rstrip(")"))
                self.args.audio_device = device_id
            else:
                self.args.audio_device = LocalAudioInput.get_default_input_device()

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

    def toggle_server(self):
        """Переключение состояния сервера."""
        if self.server_running:
            self.stop_server()
            if not self.server_running:
                self.start_stop_button.config(text="Start Server")
            else:
                logger.error("Failed to stop the server.")
        else:
            self.start_server()
            if self.server_running:
                self.start_stop_button.config(text="Stop Server")
            else:
                logger.error("Failed to start the server.")

    def start_server(self):
        """Запуск сервера."""
        if self.server_running:
            logger.warning("Server is already running.")
            return

        self.apply_changes()
        self.stop_event.clear()

        try:
            self.server_thread = threading.Thread(target=self.run_server_wrapper, args=(self.args, self.stop_event, self.append_text))
            self.server_thread.start()
            self.server_running = True
            logger.info("Server started successfully.")
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            self.server_running = False

    def stop_server(self):
        """Остановка сервера."""
        if not self.server_running:
            logger.warning("Server is not running.")
            return

        self.stop_event.set()

        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(timeout=5)
            if not self.server_thread.is_alive():
                self.server_running = False
                logger.info("Server stopped successfully.")
            else:
                logger.error("Failed to stop server: thread is still alive.")
        else:
            self.server_running = False
            logger.info("Server stopped successfully.")

    def run_server_wrapper(self, args, stop_event, callback):
        """Обертка для запуска сервера."""
        try:
            server_manager = FoxyManager(args, stop_event, callback)
            server_manager.run()
        except Exception as e:
            logger.error(f"Error in server wrapper: {e}")
        finally:
            self.server_running = False
            self.start_stop_button.config(text="Start Server")

    def toggle_advanced(self):
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

    def on_close(self):
        """Обработка закрытия окна."""
        if self.server_running:
            self.stop_server()
        self.root.destroy()

    def append_text(self, text):
        """Добавление текста в текстовое поле."""
        self.text_area.config(state=tk.NORMAL)
        self.text_area.insert(tk.END, text)
        self.text_area.see(tk.END)
        self.text_area.config(state=tk.DISABLED)

    def clear_text(self):
        """Очистка текстового поля."""
        self.text_area.config(state=tk.NORMAL)
        self.text_area.delete(1.0, tk.END)
        self.text_area.config(state=tk.DISABLED)

    def save_text(self):
        """Сохранение содержимого текстового поля в файл."""
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if file_path:
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(self.text_area.get(1.0, tk.END))

    def show_help(self):
        """Заглушка для кнопки помощи."""
        print("Help button clicked")

    def toggle_mute(self):
        """Заглушка для кнопки отключения звука."""
        print("Mute button clicked")



def main():
    parser = argparse.ArgumentParser()
    add_shared_args(parser)  # Добавляем общие аргументы
    args = parser.parse_args()

    if args.gui:
        # Запуск GUI
        root = tk.Tk()
        app = FoxyServerGUI(root, parser, args)  # Передаем parser и args
        root.mainloop()
    else:
        # Запуск сервера в консольном режиме
        server_manager = FoxyManager(args, stop_event=None, callback=None)
        server_manager.run()

if __name__ == "__main__":
    main()

    