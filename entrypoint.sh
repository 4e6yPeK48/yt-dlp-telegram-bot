#!/bin/sh
set -e

SESSION_FILE="/app/telethon.session"

echo "[+] Контейнер запущен"

if [ ! -f "$SESSION_FILE" ]; then
  echo "[+] Сессия telethon не найдена, выполняется вход"
  python -m app.scripts.telethon_login
else
  echo "[+] telethon.session найдена, пропуск входа"
fi

echo "[+] Запуск main.py"
exec python -m app.main
