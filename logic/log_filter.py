from typing import List, Set, Optional
import re

class LogFilter:
    def __init__(self):
        self.source_filters: Set[str] = set()  # фильтры по источнику
        self.level_filters: Set[str] = set()   # фильтры по уровню лога
        self.pattern_filters: List[re.Pattern] = []  # регулярные выражения для текста

    def add_source_filter(self, source: str):
        """Добавить фильтр по источнику сообщения"""
        self.source_filters.add(source.lower())

    def remove_source_filter(self, source: str):
        """Удалить фильтр по источнику"""
        self.source_filters.discard(source.lower())

    def add_level_filter(self, level: str):
        """Добавить фильтр по уровню лога"""
        self.level_filters.add(level.lower())

    def remove_level_filter(self, level: str):
        """Удалить фильтр по уровню"""
        self.level_filters.discard(level.lower())

    def add_pattern_filter(self, pattern: str):
        """Добавить фильтр по регулярному выражению"""
        try:
            self.pattern_filters.append(re.compile(pattern, re.IGNORECASE))
        except re.error:
            pass

    def clear_pattern_filters(self):
        """Очистить все фильтры по регулярным выражениям"""
        self.pattern_filters.clear()

    def clear_all_filters(self):
        """Очистить все фильтры"""
        self.source_filters.clear()
        self.level_filters.clear()
        self.pattern_filters.clear()

    def matches(self, source: str, level: str, message: str) -> bool:
        """Проверить, соответствует ли сообщение фильтрам"""
        # Если фильтры не установлены - пропускаем все
        if not (self.source_filters or self.level_filters or self.pattern_filters):
            return True

        # Нормализация входных данных
        source = source.lower().strip()
        level = level.lower().strip()

        # Отображение имен источников
        source_mappings = {
            'srcstage': 'src',
            'asrstage': 'asr',
            'system': 'server'
        }
        
        # Проверяем есть ли источник в маппинге
        if source in source_mappings:
            source = source_mappings[source]

        print(f"[FILTER] Normalized source: '{source}', Level: '{level}'")
        print(f"[FILTER] Active filters - Sources: {self.source_filters}, Levels: {self.level_filters}")

        # Проверяем фильтры источников
        if self.source_filters:
            if source not in self.source_filters:
                print(f"[FILTER] Source rejected: '{source}' not in {self.source_filters}")
                return False
            print(f"[FILTER] Source accepted: '{source}'")

        # Проверяем фильтры уровней
        if self.level_filters:
            if level not in self.level_filters:
                print(f"[FILTER] Level rejected: '{level}'")
                return False
            print(f"[FILTER] Level accepted: '{level}'")

        # Проверяем регулярные выражения
        if self.pattern_filters:
            if not any(pattern.search(message) for pattern in self.pattern_filters):
                print(f"[FILTER] Pattern rejected")
                return False
            print(f"[FILTER] Pattern accepted")

        return True
