#!/bin/bash

# Находим все процессы, запущенные с foxy_whisp_gui.py
PIDS=$(ps -ef | grep 'python3 ./foxy_whisp_gui.py' | grep -v grep | awk '{print $2}')

if [ -z "$PIDS" ]; then
    echo "Нет зависших процессов foxy_whisp_gui.py"
    exit 0
fi

# Сначала отправляем SIGTERM (мягкое завершение)
echo "Попытка корректного завершения процессов: $PIDS"
kill $PIDS

# Ждём немного, чтобы процессы завершились
sleep 3

# Проверяем, какие процессы ещё живы
PIDS=$(ps -ef | grep 'python3 ./foxy_whisp_gui.py' | grep -v grep | awk '{print $2}')

if [ -n "$PIDS" ]; then
    echo "Процессы не завершились, принудительное завершение: $PIDS"
    kill -9 $PIDS
else
    echo "Все процессы завершены."
fi
