 #!/usr/bin/env python3
from  logic.foxy_config import *
from logic.foxy_utils import load_audio_chunk,  add_shared_args,  set_logging, logger, get_port_status, create_tokenizer

from logic.asr_backends import FasterWhisperASR, OpenaiApiASR, WhisperTimestampedASR
from logic.asr_processors import OnlineASRProcessor, VACOnlineASRProcessor
from logic.mqtt_handler import MQTTHandler
from logic.foxy_engine import FoxyCore, FoxySensory

import sys
import argparse
import os
import logging
import time
import socket

import tkinter as tk
from tkinter import ttk
import threading


import tkinter as tk
from tkinter import ttk
import threading

class FoxyServerGUI:
    def __init__(self, root, parser, args):
        self.root = root
        self.parser = parser  # Store parser object
        self.args = args  # Store args object
        self.server_thread = None
        self.server_running = False
        self.stop_event = threading.Event()  # Event to stop the server

        self.root.title("Whisper Server GUI")

        # Start/Stop button
        self.start_stop_button = ttk.Button(root, text="Start Server", command=self.toggle_server)
        self.start_stop_button.pack(pady=10)

        # "Advanced" button to expand additional parameters
        self.advanced_button = ttk.Button(root, text="Advanced", command=self.toggle_advanced)
        self.advanced_button.pack(pady=10)

        # Frame for additional parameters
        self.advanced_frame = ttk.Frame(root)
        self.advanced_options_visible = False

        # Apply button to apply changes
        self.apply_button = ttk.Button(self.advanced_frame, text="Apply", command=self.apply_changes, state=tk.DISABLED)
        self.apply_button.pack(pady=10)

        # Dictionary to store widgets and their variables
        self.widgets = {}

        # Add UI controls for all parameters
        self.add_parameter_controls()

        # Handle window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def add_parameter_controls(self):
        """Adds UI controls for all parameters from the parser."""
        for action in self.parser._actions:
            if action.dest == "help":  # Skip --help argument
                continue

            frame = ttk.Frame(self.advanced_frame)
            frame.pack(fill=tk.X, padx=5, pady=5)

            # Label now shows the parameter name (action.dest)
            label = ttk.Label(frame, text=action.dest)
            label.pack(side=tk.LEFT)

            # Add tooltip with parameter description (help)
            if action.help:
                self.create_tooltip(label, action.help)

            # Determine argument type and create the appropriate widget
            if action.choices:
                # Dropdown for parameters with choices
                var = tk.StringVar(value=getattr(self.args, action.dest, action.default))
                combobox = ttk.Combobox(frame, textvariable=var, values=action.choices)
                combobox.pack(side=tk.RIGHT, fill=tk.X, expand=True)
                combobox.bind("<<ComboboxSelected>>", self.on_parameter_change)
                self.widgets[action.dest] = (combobox, var)
            elif action.type == bool or isinstance(action.default, bool):
                # Checkbox for boolean parameters
                var = tk.BooleanVar(value=getattr(self.args, action.dest, action.default))
                checkbutton = ttk.Checkbutton(frame, variable=var, command=self.on_parameter_change)
                checkbutton.pack(side=tk.RIGHT)
                self.widgets[action.dest] = (checkbutton, var)
            elif action.type in (float, int):
                # Entry field for float/int parameters
                var = tk.StringVar(value=str(getattr(self.args, action.dest, action.default)))
                entry = ttk.Entry(frame, textvariable=var)
                entry.pack(side=tk.RIGHT, fill=tk.X, expand=True)
                entry.bind("<KeyRelease>", self.on_parameter_change)
                self.widgets[action.dest] = (entry, var)
            else:
                # Entry field for string parameters
                var = tk.StringVar(value=getattr(self.args, action.dest, action.default))
                entry = ttk.Entry(frame, textvariable=var)
                entry.pack(side=tk.RIGHT, fill=tk.X, expand=True)
                entry.bind("<KeyRelease>", self.on_parameter_change)
                self.widgets[action.dest] = (entry, var)

    def create_tooltip(self, widget, text):
        """Creates a tooltip that appears when hovering over a widget."""
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
        """Enables the Apply button when any parameter changes."""
        self.apply_button.config(state=tk.NORMAL)

    def apply_changes(self):
        """Applies parameter changes and restarts the server."""
        if self.server_running:
            self.stop_server()

        # Update args object with current values from GUI
        for param, (widget, var) in self.widgets.items():
            value = var.get()

            # Convert empty string to None
            if isinstance(value, str) and value.strip() == "":
                value = None

            if isinstance(value, str) and value is not None:
                # Convert string value to the correct type
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

        self.apply_button.config(state=tk.DISABLED)

        if self.server_running:
            self.start_server()

    def toggle_server(self):
        if self.server_running:
            self.stop_server()
        else:
            self.start_server()

    def start_server(self):
        """Starts the server in a separate thread."""
        if self.server_running:
            return

        self.apply_changes()

        self.stop_event.clear()
        self.server_running = True
        self.start_stop_button.config(text="Stop Server")

        self.server_thread = threading.Thread(target=self.run_server_wrapper, args=(self.args,))
        self.server_thread.start()

    def stop_server(self):
        """Stops the server."""
        if not self.server_running:
            return

        self.server_running = False
        self.start_stop_button.config(text="Start Server")
        self.stop_event.set()

        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(timeout=5)

    def run_server_wrapper(self, args):
        """Wrapper to start the server with stop support."""
        try:
            run_server(args, self.stop_event)
        except Exception:
            pass
        finally:
            self.server_running = False
            self.start_stop_button.config(text="Start Server")

    def toggle_advanced(self):
        """Expands/collapses additional parameters."""
        if self.advanced_options_visible:
            self.advanced_frame.pack_forget()
        else:
            self.advanced_frame.pack(pady=10)
        self.advanced_options_visible = not self.advanced_options_visible

    def on_close(self):
        """Handles window close event."""
        self.stop_server()
        self.root.destroy()



#####################################
def asr_factory(args, logfile=sys.stderr):
    """ Creates and configures an ASR and ASR Online instance based on the specified backend and arguments. """
    backend = args.backend
    if backend == "openai-api":
        logger.debug("Using OpenAI API.")
        asr = OpenaiApiASR(lan=args.lan)
    else:
        if backend == "faster-whisper":
            asr_cls = FasterWhisperASR
        else:
            asr_cls = WhisperTimestampedASR

        # Only for FasterWhisperASR and WhisperTimestampedASR
        size = args.model
        t = time.time()
        logger.info(f"Loading Whisper {size} model for {args.lan}...")
        asr = asr_cls(modelsize=size, lan=args.lan, cache_dir=args.model_cache_dir, model_dir=args.model_dir)
        e = time.time()
        logger.info(f"done. It took {round(e-t,2)} seconds.")

    # Apply common configurations
    if getattr(args, 'vad', False):  # Checks if VAD argument is present and True
        logger.info("Setting VAD filter")
        asr.use_vad()

    language = args.lan
    if args.task == "translate":
        asr.set_translate_task()
        tgt_language = "en"  # Whisper translates into English
    else:
        tgt_language = language  # Whisper transcribes in this language

    # Create the tokenizer
    if args.buffer_trimming == "sentence":
        tokenizer = create_tokenizer(tgt_language)
    else:
        tokenizer = None

    # Create the OnlineASRProcessor
    if args.vac:
        online = VACOnlineASRProcessor(args.min_chunk_size, asr,tokenizer,logfile=logfile,buffer_trimming=(args.buffer_trimming, args.buffer_trimming_sec))
    else:
        online = OnlineASRProcessor(asr,tokenizer,logfile=logfile,buffer_trimming=(args.buffer_trimming, args.buffer_trimming_sec))

    return asr, online

######################################################
def run_server(args, stop_event=None):
    set_logging(args, logger)

    # Проверяем, что модель не пустая
    if not args.model:
        logger.error("Модель не может быть пустой. Установлено значение по умолчанию: large-v3-turbo.")
        args.model = "large-v3-turbo"

    asr, online = asr_factory(args)
    if args.warmup_file and os.path.isfile(args.warmup_file):
        a = load_audio_chunk(args.warmup_file, 0, 1)
        asr.transcribe(a)
        logger.info("Whisper is warmed up.")
    else:
        logger.warning("Whisper is not warmed up. The first chunk processing may take longer.")

    mqtt_handler = MQTTHandler()
    mqtt_handler.connect_to_external_broker()

    if not mqtt_handler.connected:
        mqtt_handler.start_embedded_broker()

    if mqtt_handler.connected:
        mqtt_handler.publish_message(CONNECTION_TOPIC, "<foxy:started>")
    else:
        logging.error("MQTT client is not connected. Unable to publish message.")

    # Server loop
    if get_port_status(args.port) == 0:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((args.host, args.port))
            s.listen(1)
            s.settimeout(1)  # Устанавливаем таймаут для accept(), чтобы не блокировать навсегда
            logger.info('Listening on' + str((args.host, args.port)))
            
            while get_port_status(args.port) > 0:
                if stop_event and stop_event.is_set():  # Проверка на остановку
                    logger.info("Server stopping due to stop event.")
                    break

                try:
                    conn, addr = s.accept()  # Неблокирующий accept()
                    logger.info('Connected to client on {}'.format(addr))
                    snesory_object = FoxySensory(conn, mqtt_handler, tcp_echo=True)

                    while get_port_status(args.port) == 1:
                        if stop_event and stop_event.is_set():  # Проверка на остановку
                            logger.info("Server stopping due to stop event.")
                            break

                        proc = FoxyCore(snesory_object, online, args.min_chunk_size)
                        if not proc.process():
                            break
                    conn.close()
                    logger.info('Connection to client closed')
                except socket.timeout:
                    # Таймаут accept(), продолжаем цикл
                    continue
                except Exception as e:
                    logger.error(f"Error in server loop: {e}")
                    break

        logger.info('Connection closed, terminating.')
    else:
        logger.info(f'port {args.port} already IN USE, terminating.')


# ########################################################
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
        run_server(args)

if __name__ == "__main__":
    main()