# yt\-dlp Telegram Bot (Windows/Linux)

Кратко: асинхронный бот на `aiogram 3` и `yt-dlp` ищет и скачивает аудио, отправляет пользователю в Telegram. Нужен установленный `ffmpeg`.

## Требования

1. Python `3.10+`.
2. `ffmpeg` в `PATH`.
3. Telegram Bot Token в переменной окружения `BOT_TOKEN`.
4. Доступ в интернет к Telegram API и источникам медиа.

## Установка на Windows

1. Установить Python и `ffmpeg`:
   - Python: скачать с `python.org` и добавить в `PATH`.
   - ffmpeg: установить через Chocolatey `choco install ffmpeg` или вручную и добавить `bin` в `PATH`.
2. Клонировать проект:
   - В папку, например `C:\Users\...\PycharmProjects\yt-dlp-telegram-bot`.
3. Создать виртуальное окружение и установить зависимости:

```powershell
# PowerShell
git clone https://github.com/4e6yPeK48/yt-dlp-telegram-bot yt-dlp-telegram-bot
cd .\yt-dlp-telegram-bot\
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

4. Установить `BOT_TOKEN` и запустить:

```powershell
# PowerShell
$env:BOT_TOKEN="ваш_токен_бота"
python .\main.py
```

5. Постоянная установка токена (по желанию):

```powershell
# PowerShell (сделает переменную постоянной)
setx BOT_TOKEN "ваш_токен_бота"
```

## Установка на Ubuntu

1. Установить пакеты:

```bash
# bash
sudo apt update
sudo apt install -y python3-venv python3-pip ffmpeg git
```

2. Развернуть проект:

```bash
# bash
sudo mkdir -p /opt
cd /opt
sudo git clone https://github.com/4e6yPeK48/yt-dlp-telegram-bot
sudo chown -R $USER:$USER /opt/yt-dlp-telegram-bot
cd /opt/yt-dlp-telegram-bot
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

3. Запуск вручную:

```bash
# bash
export BOT_TOKEN="ваш_токен_бота"
python main.py
```

## Автозапуск через `systemd` на Ubuntu

1. Создать сервис:

```ini
# ini
# файл: /etc/systemd/system/yt-dlp-bot.service
[Unit]
Description=yt-dlp Telegram Bot
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/yt-dlp-telegram-bot
Environment=PYTHONUNBUFFERED=1
Environment=BOT_TOKEN=ваш_токен_бота
ExecStart=/opt/yt-dlp-telegram-bot/.venv/bin/python /opt/yt-dlp-telegram-bot/main.py
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

2. Применить и запустить:

```bash
# bash
sudo systemctl daemon-reload
sudo systemctl enable --now yt-dlp-bot
systemctl status yt-dlp-bot
```

## Логи

1. Файлы логов: в `logs/` рядом с `main.py`:
   - `logs/app.debug.log`, `logs/app.info.log`, `logs/app.warn.log`, `logs/app.error.log`.
   - Ротация по полуночи, хранится 7 копий.
2. Просмотр:

```bash
# bash
tail -f logs/app.info.log
journalctl -u yt-dlp-bot -f
```

## Cookies и временные файлы

1. Cookies сохраняются в `cookies/{user_id}_cookies.txt`.
2. Медиа скачиваются во временные каталоги ОС, после отправки файлы удаляются. Папки вида `out_*` в системном `Temp` могут оставаться до очистки.

## Переменные окружения через `.env` \(опционально\)

Если нужно читать `.env`, добавьте загрузку переменных в код и установите пакет `python-dotenv`:

```python
# python
from dotenv import load_dotenv
load_dotenv()  # разместить до чтения os.getenv(...)
```

Пример файла `'.env'`:

```ini
# ini
BOT_TOKEN=ваш_токен_бота
```

## Обновление бота на сервере

```bash
# bash
cd /opt/yt-dlp-telegram-bot
git pull
source .venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart yt-dlp-bot
```

## Частые проблемы

1. Нет `ffmpeg`: конвертация в `mp3` не выполняется, отправка не состоится.
2. `aiogram 3.7+`: инициализируйте `Bot` с `default=DefaultBotProperties(parse_mode="HTML")`.
3. Windows MAX\_PATH: при очень длинных именах включите длинные пути в реестре или упростите шаблон имени в коде.