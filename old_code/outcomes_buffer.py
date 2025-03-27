from logic.foxy_config import *
from logic.foxy_utils import logger
import sys

##################################################
class IdeasBuffer:
###############################
    def __init__(self, logfile=sys.stderr):
        """Инициализирует буфер гипотез."""
        self.ideas_buffer = []  # Неподтвержденные части текста
        self.commited_ideas = []  # Подтвержденные части текста
        self.new_ideas = []  # Новые части текста для обработки

        self.last_commited_time = 0  # Время последнего подтвержденного слова
        self.last_commited_word = None  # Последнее подтвержденное слово

        self.logfile = logfile

#################################
    def insert(self, new, timestamps_offset):
        """Вставляет новые данные в буфер."""

        # Применяем смещение к новым данным
        new = [(a + timestamps_offset, b + timestamps_offset, t) for a, b, t in new]

        # Фильтруем новые данные, оставляя только те, которые идут после последнего подтвержденного времени
        self.new_ideas = [(a, b, t) for a, b, t in new if a > self.last_commited_time - 0.1]

        if not self.new_ideas:
            return

        # Проверяем, есть ли пересечение с последним подтвержденным словом
        a, b, t = self.new_ideas[0]
        if abs(a - self.last_commited_time) < 1 and self.commited_ideas:
            self._remove_duplicate_ngrams()

#####################################
    def _remove_duplicate_ngrams(self):
        """Удаляет n-граммы, которые уже есть в подтвержденных данных."""
        cn = len(self.commited_ideas)
        nn = len(self.new_ideas)
        for i in range(1, min(min(cn, nn), 5) + 1):  # Проверяем n-граммы длиной от 1 до 5
            # Сравниваем n-граммы из подтвержденных и новых данных
            committed_ngram = " ".join([self.commited_ideas[-j][2] for j in range(1, i + 1)][::-1])
            new_ngram = " ".join(self.new_ideas[j - 1][2] for j in range(1, i + 1))

            if committed_ngram == new_ngram:
                # Удаляем дубликаты
                words = [repr(self.new_ideas.pop(0)) for _ in range(i)]
                logger.debug(f"Removing last {i} words: {' '.join(words)}")
                break

##############################
    def flush(self):
        """Извлекает подтвержденные части текста."""
        commit = []
        while self.new_ideas:
            na, nb, nt = self.new_ideas[0]

            if not self.ideas_buffer:
                break

            if nt == self.ideas_buffer[0][2]:
                commit.append((na, nb, nt))
                self.last_commited_word = nt
                self.last_commited_time = nb
                self.ideas_buffer.pop(0)
                self.new_ideas.pop(0)
            else:
                break

        self.ideas_buffer = self.new_ideas
        self.new_ideas = []
        self.commited_ideas.extend(commit)
        return commit

#########################
    def pop_commited(self, time_stamp):
        """Удаляет подтвержденные части текста, которые завершились до указанного времени. """
        while self.commited_ideas and self.commited_ideas[0][1] <= time_stamp:
            self.commited_ideas.pop(0)

##########################
    def complete(self):
        """Возвращает неподтвержденные части текста.

        Returns:
            Список кортежей (начало, конец, текст).
        """
        return self.ideas_buffer