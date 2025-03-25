import unittest
import sys
import os
from unittest.runner import TextTestRunner
from unittest.result import TestResult
import logging

class LoggingTestResult(unittest.TextTestResult):
    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        # Создаем директорию для логов
        os.makedirs('test_logs', exist_ok=True)
        
        # Настраиваем базовый логгер
        self.base_logger = logging.getLogger('TestLogger')
        self.base_logger.setLevel(logging.INFO)
        
    def startTest(self, test):
        super().startTest(test)
        test_name = test.id().split('.')[-1]
        log_path = os.path.join('test_logs', f"{test_name}.log")
        
        # Создаем логгер для текущего теста
        self.current_logger = logging.getLogger(test_name)
        self.current_logger.setLevel(logging.INFO)
        
        # Очищаем предыдущие обработчики
        for handler in self.current_logger.handlers[:]:
            self.current_logger.removeHandler(handler)
        
        # Добавляем файловый обработчик
        file_handler = logging.FileHandler(log_path)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.current_logger.addHandler(file_handler)
        
        # Добавляем обработчик для вывода в консоль
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.current_logger.addHandler(console_handler)
        
        self.current_logger.info(f"Starting test: {test_name}")

    def addSuccess(self, test):
        super().addSuccess(test)
        if hasattr(self, 'current_logger'):
            self.current_logger.info("Test passed successfully")

    def addError(self, test, err):
        super().addError(test, err)
        if hasattr(self, 'current_logger'):
            self.current_logger.error(f"Test error: {err[1]}")

    def addFailure(self, test, err):
        super().addFailure(test, err)
        if hasattr(self, 'current_logger'):
            self.current_logger.error(f"Test failed: {err[1]}")

def run_tests():
    # Загружаем все тесты из директории tests
    loader = unittest.TestLoader()
    tests = loader.discover('tests', pattern='test_*.py')
    
    # Создаем и настраиваем runner
    runner = TextTestRunner(
        verbosity=2,
        resultclass=LoggingTestResult
    )
    
    # Запускаем тесты
    result = runner.run(tests)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
