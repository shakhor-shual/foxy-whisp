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