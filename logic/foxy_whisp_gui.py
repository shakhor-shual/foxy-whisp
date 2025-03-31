class FoxyWhispGUI:
    # ...existing code...

    def handle_data_message(self, msg: PipelineMessage):
        """Handle data messages from server"""
        data_type = msg.content.get('data_type')
        payload = msg.content.get('payload')
        
        if data_type == 'audio_level':
            # Используем уровень напрямую для индикатора (0-100)
            level = payload.get('level', 0.0)
            self.update_audio_level_safe(level)
            
            # Обновляем текст метки
            self.level_label.config(text=f"Level: {level:.0f}%")
        # ...existing code...

    def update_audio_level(self, level_data):
        """Update audio level meter with widget existence check"""
        try:
            if hasattr(self, 'level_bar') and self.level_bar.winfo_exists():
                if isinstance(level_data, dict):
                    level = level_data.get('level', 0)
                    if isinstance(level, (int, float)):
                        self.level_bar["value"] = level
        except Exception as e:
            import traceback
            print(f"[GUI.ERROR] Failed to update audio level: {e}\n{traceback.format_exc()}")

    def _update_vad_status(self, is_active: bool):
        """Direct VAD status update with widget check"""
        try:
            if hasattr(self, 'vad_label') and self.vad_label.winfo_exists():
                current = self.vad_label.cget('background')
                target = '#00ff00' if is_active else '#ff0000'
                if str(current) != str(target):
                    self._animate_color_change(current, target)
        except Exception as e:
            import traceback
            logger.error(f"VAD update error: {str(e)}\n{traceback.format_exc()}")

    def _animate_color_change(self, start_color, end_color, steps=10):
        """Color animation with widget checks"""
        if not hasattr(self, 'vad_label') or not self.vad_label.winfo_exists():
            return
            
        if start_color == end_color:
            return
            
        try:
            # ...existing code...

            def animate(step=0):
                if step <= steps:
                    try:
                        if not self.vad_label.winfo_exists():
                            return
                        color = rgb_to_hex(interpolate_colors(start_rgb, end_rgb, step, steps))
                        self.vad_label.configure(background=color)
                        self.root.after(20, lambda: animate(step + 1))
                    except Exception as e:
                        import traceback
                        logger.error(f"Animation error: {str(e)}\n{traceback.format_exc()}")
            
            animate()
            
        except Exception as e:
            import traceback
            logger.error(f"Color animation error: {str(e)}\n{traceback.format_exc()}")