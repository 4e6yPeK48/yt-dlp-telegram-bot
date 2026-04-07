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
    - для `yt-dlp` желательно иметь JS runtime в `PATH` (`node`, `deno`, `bun` или `qjs`); при необходимости можно задать `YTDLP_JS_RUNTIME=node`
3. Запуск:
   ```bash
   python main.py
   ```

---

## Запуск через Docker

Этот проект можно запускать в Docker. Ниже описан минимальный, безопасный и воспроизводимый способ запуска бота в Docker.

### Требования (пример для Ubuntu / Debian)
- Установите Docker Engine и, при необходимости, Docker Compose:
  - Следуйте официальному руководству по установке Docker: https://docs.docker.com/engine/install/
  - При желании установите Docker Compose: https://docs.docker.com/compose/install/
- Убедитесь, что ваш пользователь может выполнять команды Docker (добавьте в группу `docker` или используйте `sudo`).

### Подготовка окружения
1. Создайте рабочую директорию и файл окружения:
   ```bash
   mkdir -p ~/ytbot && cd ~/ytbot
   cat > .env <<'EOF'
   BOT_TOKEN=your_bot_token_here
   TELETHON_API_ID=your_api_id_here
   TELETHON_API_HASH=your_api_hash_here
   TELETHON_FALLBACK_ENABLED=true_or_false
   PHONE=your_telethon_phone_number_here
   EOF
   ```
   - Поместите необходимые секреты в `~/ytbot/.env`. Храните этот файл в приватном доступе.

2. (Опционально) При необходимости отредактируйте другие переменные окружения, используемые проектом (см. `config.py` на GitHub).

> В контейнере `nodejs` уже устанавливается через `Dockerfile`, поэтому предупреждение про отсутствие JS runtime обычно исчезает.

### Запуск контейнера
- Скачайте образ и запустите (в фоне, с политикой перезапуска, используя `--env-file`):
  ```bash
  docker pull m4estro777/ytbot:latest
  docker run -d \
    --name ytbot \
    --restart unless-stopped \
    --env-file .env \
    m4estro777/ytbot:latest
  ```
  - `--env-file .env` загружает переменные из `~/ytbot/.env`.
  - при первом запуске бот потребует авторизацию Telethon‑аккаунта (если включён Telethon‑fallback).

### Обновление / повторный деплой
```bash
docker pull m4estro777/ytbot:latest
docker stop ytbot && docker rm ytbot
# повторно выполните ту же команду docker run, что и выше
```

### Логи и управление
- Просмотр логов:
  ```bash
  docker logs -f ytbot
  # или просмотреть файлы логов внутри запущенного контейнера
  docker exec -it ytbot ls -la /app/logs
  ```
- Остановка / запуск:
  ```bash
  docker stop ytbot
  docker start ytbot
  ```
- Удаление контейнера:
  ```bash
  docker stop ytbot && docker rm ytbot
  ```

### Опционально: пример `docker-compose`
Создайте `docker-compose.yml` в `~/ytbot`:
```yaml
version: "3.8"
services:
  ytbot:
    image: m4estro777/ytbot:latest
    restart: unless-stopped
    env_file:
      - .env
    volumes:
      - ./cookies:/app/cookies
      - ./server_cookies:/app/server_cookies
```
Запуск:
```bash
docker compose up -d
```

---

## Настройка с Poetry

Этот проект использует Poetry для управления зависимостями Python и виртуальными окружениями. Инструкции ниже предполагают использование Linux.

### Требования

* Python 3.11 или новее
* Poetry (установите, следуя официальному руководству: https://python-poetry
* Системные пакеты, необходимые для инструментов и обработки медиа:

  * ffmpeg
  * Библиотеки для сборки Pillow (пример для Debian/Ubuntu): `sudo apt install -y libjpeg-dev zlib1g-dev`
  * (Опционально) libcurl / другие системные библиотеки для некоторых дополнительных функций

### Быстрый старт (рекомендуется)

1. Укажите Poetry использовать правильную версию Python:

   * `poetry env use python3.11`

2. Установите зависимости (создаст изолированное виртуальное окружение):

   * `poetry install`

3. Используйте виртуальное окружение через shell или `poetry run` для команд:

   * Войти в shell: `poetry shell`
   * Или выполнить отдельную команду: `poetry run <command>`

### Переменные окружения

Создайте файл `.env` (или установите переменные окружения) с необходимыми ключами. Пример `.env`:

```env
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELETHON_API_ID=123456
TELETHON_API_HASH=your_api_hash
PHONE=+1234567890   # требуется только для скрипта входа через Telethon
```

Проект автоматически подгружает `.env` при необходимости.

### Telethon (альтернативная доставка) — вход в аккаунт

Если включён fallback через Telethon, нужно один раз авторизовать сессию пользователя:

* Запустите скрипт входа:

  * `poetry run python app/scripts/telethon_login.py`

Скрипт запустит Telethon и запросит телефон/код. После успешного входа будет создан файл сессии (например, `telethon.session`), который будет использовать бот.

### Тесты и линтеры

* Запуск unit-тестов:

  * `poetry run pytest`
* Проверка типов и линтеры:

  * `poetry run mypy`
  * `poetry run black --check .`
  * `poetry run flake8`

### Экспорт `requirements.txt`

В этом репозитории настроен `poetry-plugin-export`. Чтобы экспортировать фиксированные зависимости в `requirements.txt` (для деплоя или контейнеров):

* `poetry export -f requirements.txt -o requirements.txt --without-hashes --dev`
  (уберите `--dev`, чтобы исключить dev-зависимости)

### Запуск бота

Запускайте бота через окружение Poetry. Пример (замените на ваш реальный entrypoint, если отличается):

* `poetry run python -m app`
  или
* `poetry run python path/to/entrypoint.py`

Используйте `poetry run`, чтобы команда выполнялась внутри виртуального окружения Poetry.

### Примечания и устранение проблем

* Если возникают ошибки сборки Pillow, установите системные библиотеки для работы с изображениями (см. Требования).
* Если нужно доставить большой файл и Bot API не справляется, бот попытается использовать fallback через Telethon — убедитесь, что сессия Telethon авторизована и пользователь следует инструкциям бота.
* Чтобы пересоздать `requirements.txt` для Docker, выполните шаг экспорта выше и скопируйте файл в ваш образ.

### Полезные команды (резюме)

```bash
poetry env use python3.11
poetry install
poetry shell                # или используйте `poetry run ...`
poetry run pytest
poetry run python app/scripts/telethon_login.py
poetry export -f requirements.txt -o requirements.txt --without-hashes --dev
```


---


## Конфигурация

Основные опции находятся в `config.py`:

- `BOT_TOKEN`, `TELETHON_API_ID`, `TELETHON_API_HASH`, `TELETHON_SESSION`
- `CONCURRENT_DOWNLOADS`, `MAX_PLAYLIST_ITEMS`, `DURATION_LIMIT_SEC`
- `MAX_FILE_MB` (жёсткий лимит на скачиваемый файл)
- `TG_MAX_UPLOAD_MB` (лимит бота для переключения на Telethon)
- `SERVER_COOKIES_SOURCES_JSON` и `SERVER_COOKIES_MAP` — для серверных cookies
- `YTDLP_JS_RUNTIME` — выбор JS runtime для `yt-dlp` (`auto`, `node`, `deno`, `bun`, `quickjs`)

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
