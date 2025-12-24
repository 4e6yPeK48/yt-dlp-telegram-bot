# yt-dlp Telegram Bot

Лёгкий асинхронный Telegram‑бот на `aiogram 3`, который ищет и скачивает аудио/видео с помощью `yt-dlp` и доставляет их
пользователям. Поддерживает опциональный Telethon‑fallback для доставки больших файлов и централизованные серверные
cookies.

## Ключевые возможности

- Принятие ссылки или поискового запроса. Для прямой ссылки показывается карточка с метаданными; для запроса — список
  результатов поиска на Youtube (с пагинацией).
- Режимы скачивания: `auto` / `audio` / `video` / `video_nosound`.
- Генерация миниатюр (320×320 JPEG) и встраивание метаданных для аудио.
- Доставка через Telegram Bot API; при превышении лимита Bot API — попытка альтернативной доставки через авторизованный
  Telethon‑аккаунт.
- Поддержка пользовательских `cookies.txt` (Netscape) и серверных cookies с периодическим обновлением.
- In‑memory per‑user state: локи для предотвращения параллельных загрузок, отложенные задачи, история и токены для
  безопасного выбора скачивания.
- Надёжное логирование с ротацией логов.

## Быстрый старт (Ubuntu)

1. Клонировать репозиторий и создать виртуальное окружение:
   ```bash
   git clone <repo>
   cd <repo>
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -U pip
   pip install -r requirements.txt
   ```
2. Установить переменные окружения (или поместить в `.env`):
    - `BOT_TOKEN` — токен Telegram‑бота (обязательно)
    - опционально: `TELETHON_API_ID`, `TELETHON_API_HASH`, `TELETHON_SESSION`
    - при использовании серверных cookies: `SERVER_COOKIES_SOURCES_JSON`
3. Запуск:
   ```bash
   python main.py
   ```

## Конфигурация

Основные опции находятся в `config.py`:

- `BOT_TOKEN`, `TELETHON_API_ID`, `TELETHON_API_HASH`, `TELETHON_SESSION`
- `CONCURRENT_DOWNLOADS`, `MAX_PLAYLIST_ITEMS`, `DURATION_LIMIT_SEC`
- `MAX_FILE_MB` (жёсткий лимит на скачиваемый файл)
- `TG_MAX_UPLOAD_MB` (лимит бота для переключения на Telethon)
- `SERVER_COOKIES_SOURCES_JSON` и `SERVER_COOKIES_MAP` — для серверных cookies

Каталоги cookies и серверных cookies: `cookies/` и `server_cookies/` (создаются автоматически).

## Как пользоваться

- Отправьте ссылку — бот покажет карточку с названием, длительностью и превью; выберите, что скачать.
- Отправьте текстовый запрос — бот выполнит поиск (YouTube) и покажет до `MAX_RESULTS` результатов с пагинацией.
- Если источник требует авторизации/защиту, отправьте `cookies.txt` как документ (формат Netscape, макс. 5 МБ); бот
  повторит операцию автоматически.
- Для больших файлов бот предложит альтернативную доставку через Telethon: надо отправить любое сообщение
  авторизованному аккаунту (рукопожатие), затем файл будет доставлен MTProto.

## Telethon‑fallback

- Опционален. Если включён и настроены переменные (`TELETHON_*`), Telethon используется для доставки файлов, превышающих
  лимит Bot API.
- Процесс: бот уведомляет пользователя, ожидает рукопожатие, затем Telethon загружает файл от имени авторизованного
  аккаунта.
- Скрипт для локальной авторизации: `scripts/telethon_login.py`.

## Серверные cookies

- Источники и их URL задаются в `SERVER_COOKIES_SOURCES_JSON`.
- Скрипт для одноразовой загрузки server cookies: `scripts/fetch_server_cookies.py`.
- Сервис загрузит файлы в `server_cookies/` и будет использовать их для соответствующих доменов.

## Логи и мониторинг

- Логирование настроено в `utils/logging.py` с ротацией в папке `logs/`.
- В консоли отображаются записи с префиксом `bot` на уровне INFO; подробности сохраняются в файлы `app.debug.log`,
  `app.info.log`, `app.warn.log`, `app.error.log`.

## Тесты

- Юнит‑тесты покрывают core‑функциональность: состояние, клавиатуры, утилиты.
- Запуск тестов:
  ```bash
  pip install -r requirements.txt
  pytest -q
  ```

## Структура проекта (кратко)

- `main.py` — входная точка, настройка логирования и polling.
- `bot/` — хендлеры, клавиатуры, routing (`commands.py`, `messages.py`, `callbacks.py`, `keyboards.py`).
- `services/` — интеграции: `ytdlp.py`, `telegram.py`, `telethon_client.py`, `media.py`, `server_cookies.py`.
- `storage/state.py` — in‑memory state (локи, pending токены, история, задачи).
- `utils/` — хелперы: `text.py`, `validators.py`, логирование.
- `scripts/` — утилиты для администрирования.
- `tests/` — pytest тесты.

## Безопасность и ограничения

- Файлы cookies хранятся локально в `cookies/{user_id}_cookies.txt` и ограничены настройкой `COOKIES_MAX_BYTES`.
- Ограничения по длительности (`DURATION_LIMIT_SEC`), числу элементов плейлиста (`MAX_PLAYLIST_ITEMS`) и размеру файла (
  `MAX_FILE_MB`) предотвращают перегрузку сервера.
- История — in‑memory; при необходимости можно заменить на постоянное хранилище.

## Примечания для эксплуатации

- Для production‑запуска рекомендуется systemd‑unit с указанием рабочего каталога и venv.
- Для автоматического обновления серверных cookies используйте `scripts/fetch_server_cookies.py` в cron или systemd
  timer.
