import asyncio
import os
import re
from dotenv import load_dotenv
import math
import tempfile
import shutil
from contextlib import suppress
from urllib.parse import urlparse
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Pattern
import logging
from logging.handlers import TimedRotatingFileHandler
import io
from PIL import Image, ImageOps
from PIL.Image import Resampling
import secrets

from aiogram import Bot, Dispatcher, Router, F
from aiogram.filters import CommandStart, Command
from aiogram.types import (
    Message,
    CallbackQuery,
    InlineKeyboardButton,
    FSInputFile,
    ReplyKeyboardMarkup,
    KeyboardButton,
)
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.client.default import DefaultBotProperties

from yt_dlp import YoutubeDL  # type: ignore[import-untyped]
from yt_dlp.utils import DownloadError  # type: ignore[import-untyped]

# ========= Настройки =========
load_dotenv()

BOT_TOKEN: Optional[str] = os.getenv("BOT_TOKEN")
MAX_RESULTS: int = 25
PAGE_SIZE: int = 5
CONCURRENT_DOWNLOADS: int = 2
AUDIO_EXTS: Set[str] = {".mp3", ".m4a", ".opus", ".webm", ".ogg", ".flac", ".wav"}
IMAGE_EXTS: Set[str] = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTS: Set[str] = {".mp4", ".mkv", ".webm", ".mov", ".m4v"}
MAX_PLAYLIST_ITEMS: int = 10
DURATION_LIMIT_SEC: int = 30 * 60
MAX_QUERY_LEN: int = 120

THUMB_SIZE: Tuple[int, int] = (320, 320)
THUMB_MAX_BYTES: int = 200 * 1024

CAPTION_MAX_LEN: int = 1000
TG_MAX_UPLOAD_BYTES: int = int(os.getenv("TG_MAX_UPLOAD_MB", "50")) * 1024 * 1024
COOKIES_MAX_BYTES: int = int(os.getenv("COOKIES_MAX_MB", "5")) * 1024 * 1024
ALLOWED_COOKIES_EXTS: Set[str] = {".txt"}

BTN_MENU: str = "🏠 Меню (/start, /menu)"
BTN_HELP: str = "❓ Помощь (/help)"
BTN_SETTINGS: str = "⚙️ Настройки (/settings)"

# ========= Глобальные объекты =========
router: Router = Router()
dp: Dispatcher = Dispatcher()
dp.include_router(router)
download_sem: asyncio.Semaphore = asyncio.Semaphore(CONCURRENT_DOWNLOADS)

USER_SEARCHES: Dict[int, Dict[str, Any]] = {}
AWAITING_COOKIES: Dict[int, Dict[str, Any]] = {}
COOKIES_DIR: str = os.path.join(os.getcwd(), "cookies")
os.makedirs(COOKIES_DIR, exist_ok=True)
USER_SETTINGS: Dict[int, Dict[str, str]] = {}
USER_LOCKS: Dict[int, asyncio.Lock] = {}
PENDING_DOWNLOADS: Dict[str, Dict[str, Any]] = {}


# ========= Логирование =========
def setup_logging(log_dir: str = "logs") -> None:
    """Настраивает логирование: консольный вывод и ротацию по уровням.

    Args:
        log_dir (str): Директория для файлов логов.
    """
    os.makedirs(log_dir, exist_ok=True)
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    class OnlyLoggerFilter(logging.Filter):
        """Пропускает только записи выбранного логгера (по префиксу имени)."""
        def __init__(self, prefix: str) -> None:
            super().__init__()
            self.prefix = prefix

        def filter(self, record: logging.LogRecord) -> bool:
            return record.name.startswith(self.prefix)

    def make_rotating(path: str, level: int) -> TimedRotatingFileHandler:
        handler = TimedRotatingFileHandler(
            filename=os.path.join(log_dir, path),
            when="midnight",
            backupCount=7,
            encoding="utf-8",
        )
        handler.setLevel(level)
        handler.setFormatter(fmt)
        return handler

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    console.addFilter(OnlyLoggerFilter("bot"))
    root.addHandler(console)

    root.addHandler(make_rotating("app.debug.log", logging.DEBUG))

    info_h = make_rotating("app.info.log", logging.INFO)
    info_h.addFilter(OnlyLoggerFilter("bot"))
    root.addHandler(info_h)

    root.addHandler(make_rotating("app.warn.log", logging.WARNING))
    root.addHandler(make_rotating("app.error.log", logging.ERROR))

    third_party = [
        ("aiogram", "aiogram.error.log"),
        ("aiohttp", "aiohttp.error.log"),
        ("yt_dlp", "yt-dlp.error.log"),
    ]
    for name, fname in third_party:
        lg = logging.getLogger(name)
        lg.setLevel(logging.DEBUG)
        errh = TimedRotatingFileHandler(
            filename=os.path.join(log_dir, fname),
            when="midnight",
            backupCount=7,
            encoding="utf-8",
        )
        errh.setLevel(logging.ERROR)
        errh.setFormatter(fmt)
        lg.addHandler(errh)

    logging.getLogger("aiogram").setLevel(logging.INFO)
    logging.getLogger("aiohttp").setLevel(logging.INFO)
    logging.getLogger("yt_dlp").setLevel(logging.INFO)


logger: logging.Logger = logging.getLogger("bot")


# ========= Основная логика =========
def is_url(text: str) -> bool:
    """Проверяет, является ли строка URL со схемой http/https.

    Args:
        text (str): Исходная строка.

    Returns:
        bool: True, если строка похожа на URL, иначе False.
    """
    with suppress(Exception):
        u = urlparse(text.strip())
        return u.scheme in {"http", "https"} and bool(u.netloc)
    return False


def slice_page(items: List[Any], page: int, page_size: int) -> Tuple[List[Any], int]:
    """Возвращает элементы указанной страницы и общее число страниц.

    Args:
        items (List[Any]): Полный список элементов.
        page (int): Номер страницы (0-индексация).
        page_size (int): Размер страницы.

    Returns:
        Tuple[List[Any], int]: Элементы текущей страницы и всего страниц.
    """
    pages = max(1, math.ceil(len(items) / page_size))
    page = max(0, min(page, pages - 1))
    start = page * page_size
    end = start + page_size
    return items[start:end], pages


def get_user_mode(user_id: int) -> str:
    """Возвращает текущий режим пользователя.

    Args:
        user_id (int): Идентификатор пользователя Telegram.

    Returns:
        str: Один из: 'auto', 'audio', 'video', 'video_nosound'.
    """
    st = USER_SETTINGS.get(user_id)
    return (st or {}).get("mode", "auto")


def set_user_mode(user_id: int, mode: str) -> None:
    """Сохраняет выбранный режим пользователя.

    Args:
        user_id (int): Идентификатор пользователя.
        mode (str): Режим ('auto'|'audio'|'video'|'video_nosound').
    """
    USER_SETTINGS[user_id] = {"mode": mode}


def is_audio_platform(url: str) -> bool:
    """Эвристически определяет, что ресурс аудио-ориентирован.

    Args:
        url (str): URL ресурса.

    Returns:
        bool: True если сайт похоже аудио-площадка.
    """
    try:
        u = urlparse(url)
        host = (u.netloc or "").lower()
        path = (u.path or "").lower()
    except Exception:
        return False
    audio_hosts = [
        "music.youtube.",
        "soundcloud.com",
        "bandcamp.com",
        "mixcloud.com",
        "audius.co",
        "hearthis.at",
        "promodj.com",
        "music.yandex.",
        "yandex.ru/music",
        "deezer.com",
        "napster.com",
    ]
    return any(h in host for h in audio_hosts) or "/music" in path


def decide_effective_mode(user_mode: str, url: str) -> str:
    """Определяет итоговый режим скачивания.

    Args:
        user_mode (str): Режим выбранный пользователем ('auto', 'audio', 'video', 'video_nosound').
        url (str): URL источника.

    Returns:
        str: Итоговый режим ('audio'|'video'|'video_nosound').
    """
    if user_mode == "auto":
        return "audio" if is_audio_platform(url) else "video"
    return user_mode


def is_youtube_url(url: str) -> bool:
    """Определяет, относится ли URL к YouTube/YouTube Music.

    Args:
        url (str): Проверяемый URL.

    Returns:
        bool: True если URL относится к YouTube.
    """
    try:
        host = (urlparse(url).netloc or "").lower()
    except Exception:
        return False
    return any(
        h in host
        for h in ("youtube.", "youtu.be", "music.youtube.")
    )


def build_results_kb(user_id: int) -> InlineKeyboardBuilder:
    """Строит инлайн-клавиатуру результатов поиска с пагинацией.

    Args:
        user_id (int): Идентификатор пользователя.

    Returns:
        InlineKeyboardBuilder: Сконструированный билдер.
    """
    state = USER_SEARCHES.get(user_id) or {}
    results: List[Dict[str, Any]] = state.get("results", [])
    page: int = state.get("page", 0)

    current, pages = slice_page(results, page, PAGE_SIZE)
    kb = InlineKeyboardBuilder()

    for idx, entry in enumerate(current):
        global_index = page * PAGE_SIZE + idx
        title = entry.get("title") or "Без названия"
        if len(title) > 64:
            title = title[:61] + "..."
        kb.button(text=title, callback_data=f"pick:{global_index}")

    if not results:
        kb.button(text="Нет результатов", callback_data="noop")
    kb.adjust(1)

    if results:
        kb.row(
            InlineKeyboardButton(text="« Назад", callback_data="page:prev"),
            InlineKeyboardButton(text=f"{page + 1}/{pages}", callback_data="noop"),
            InlineKeyboardButton(text="Вперёд »", callback_data="page:next"),
        )
    kb.row(InlineKeyboardButton(text="❌ Отмена", callback_data="cancel"))
    return kb


def build_settings_kb(user_id: int) -> InlineKeyboardBuilder:
    """Строит инлайн-меню выбора режима скачивания.

    Args:
        user_id (int): Идентификатор пользователя.

    Returns:
        InlineKeyboardBuilder: Клавиатура настроек.
    """
    mode = get_user_mode(user_id)
    text: Dict[str, str] = {
        "auto": "Автоопределение 🤖",
        "audio": "Только аудио 🎵",
        "video": "Только видео (со звуком) 🎬🔊",
        "video_nosound": "Только видео (без звука) 🎬🔇",
    }
    kb = InlineKeyboardBuilder()
    for m in ["auto", "audio", "video", "video_nosound"]:
        pref = "✅ " if mode == m else "• "
        kb.button(text=pref + text[m], callback_data=f"setmode:{m}")
    kb.adjust(1)
    kb.row(InlineKeyboardButton(text="Закрыть", callback_data="settings:close"))
    return kb


def make_dl_token() -> str:
    """Генерирует уникальный токен для отложенного скачивания.

    Returns:
        str: Токен (10 символов [A-Za-z0-9]).
    """
    t = ""
    for _ in range(5):
        t = secrets.token_urlsafe(6).replace("-", "").replace("_", "")[:10]
        if t not in PENDING_DOWNLOADS:
            break
    return t


def build_download_choice_kb(user_id: int, token: str) -> InlineKeyboardBuilder:
    """Строит клавиатуру выбора типа скачивания для конкретного URL.

    Args:
        user_id (int): Идентификатор пользователя.
        token (str): Токен сохранённого URL.

    Returns:
        InlineKeyboardBuilder: Клавиатура выбора.
    """
    kb = InlineKeyboardBuilder()
    kb.row(InlineKeyboardButton(text="🎵 Скачать аудио", callback_data=f"dl:audio:{token}"))
    kb.row(InlineKeyboardButton(text="🎬 Скачать видео", callback_data=f"dl:video:{token}"))
    kb.row(InlineKeyboardButton(text="📥 Лучшее качество (авто)", callback_data=f"dl:auto:{token}"))
    kb.row(InlineKeyboardButton(text="⚙️ Изменить тип скачивания", callback_data="settings:open"))
    return kb


def save_pending_url(user_id: int, url: str) -> str:
    """Сохраняет URL для последующего выбора режима отправки.

    Args:
        user_id (int): Идентификатор пользователя.
        url (str): Сохранённый URL.

    Returns:
        str: Токен сохранения.
    """
    token = make_dl_token()
    PENDING_DOWNLOADS[token] = {"user_id": user_id, "url": url}
    return token


def build_main_reply_kb() -> ReplyKeyboardMarkup:
    """Строит основную reply-клавиатуру.

    Returns:
        ReplyKeyboardMarkup: Клавиатура с основными командами.
    """
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text=BTN_MENU), KeyboardButton(text=BTN_HELP), KeyboardButton(text=BTN_SETTINGS)],
        ],
        resize_keyboard=True,
        is_persistent=True,
    )


def parse_main_button_intent(text: str) -> Optional[str]:
    t = (text or "").strip()
    if not t:
        return None
    low = t.lower()

    if re.search(r"/start\b", low) or re.search(r"/menu\b", low):
        return "menu"
    if re.search(r"/help\b", low):
        return "help"
    if re.search(r"/settings\b", low):
        return "settings"

    cleaned = re.sub(r"[^\w\sА-Яа-яёЁ-]", " ", low)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if re.search(r"\bменю\b", cleaned):
        return "menu"
    if re.search(r"\bпомощ", cleaned):
        return "help"
    if re.search(r"\bнастрой", cleaned):
        return "settings"

    return None


async def try_cb_answer(cb: CallbackQuery, text: Optional[str] = None) -> None:
    """Безопасно отправляет ответ на callback.

    Args:
        cb (CallbackQuery): Callback-запрос.
        text (Optional[str]): Текст уведомления.
    """
    with suppress(Exception):
        await cb.answer(text)


def get_cb_chat_id(cb: CallbackQuery) -> Optional[int]:
    """Получает chat_id из CallbackQuery.

    Args:
        cb (CallbackQuery): Объект запроса.

    Returns:
        Optional[int]: Идентификатор чата или None.
    """
    msg_obj = cb.message
    if msg_obj is not None and isinstance(msg_obj, Message):
        return msg_obj.chat.id
    if cb.from_user is not None:
        return cb.from_user.id
    return None


def sanitize_query(text: str) -> str:
    """Очищает поисковый запрос (управляющие символы, пробелы, длину).

    Args:
        text (str): Исходный текст.

    Returns:
        str: Санитизированный запрос.
    """
    t = re.sub(r"[\x00-\x1f\x7f]", "", text)
    t = re.sub(r"[\u200B-\u200F\u202A-\u202E\u2060-\u206F]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > MAX_QUERY_LEN:
        t = t[:MAX_QUERY_LEN]
    return t


def make_caption(text: str, limit: int = CAPTION_MAX_LEN) -> str:
    """Очищает текст и обрезает его для подписи (однострочно).

    Args:
        text (str): Исходный текст.
        limit (int): Максимальная длина.

    Returns:
        str: Подготовленная подпись.
    """
    t = re.sub(r"[\x00-\x1f\x7f]", "", text or "")
    t = re.sub(r"[\u200B-\u200F\u202A-\u202E\u2060-\u206F]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > limit:
        t = t[: limit - 1] + "…"
    return t


def make_multiline_caption(text: str, limit: int = CAPTION_MAX_LEN) -> str:
    """Очищает текст (с сохранением перевода строк) и обрезает до лимита.

    Args:
        text (str): Исходный текст.
        limit (int): Максимальная длина.

    Returns:
        str: Подготовленный многострочный текст.
    """
    t = text or ""
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[\x00-\x09\x0B-\x0C\x0E-\x1F\x7F]", "", t)
    t = re.sub(r"[\u200B-\u200F\u202A-\u202E\u2060-\u206F]", "", t)
    lines = [line.rstrip() for line in t.split("\n")]
    t = "\n".join(lines)
    if len(t) > limit:
        t = t[: limit - 1] + "…"
    return t


def get_user_lock(user_id: int) -> asyncio.Lock:
    """Получает или создаёт Lock для пользователя.

    Args:
        user_id (int): Идентификатор пользователя.

    Returns:
        asyncio.Lock: Lock пользователя.
    """
    lock = USER_LOCKS.get(user_id)
    if lock is None:
        lock = asyncio.Lock()
        USER_LOCKS[user_id] = lock
    return lock


async def begin_user_download(user_id: int) -> Optional[asyncio.Lock]:
    """Пытается захватить пользовательский Lock перед загрузкой.

    Args:
        user_id (int): Идентификатор пользователя.

    Returns:
        Optional[asyncio.Lock]: Захваченный Lock или None если занят.
    """
    lock = get_user_lock(user_id)
    if lock.locked():
        return None
    await lock.acquire()
    return lock


def end_user_download(lock: Optional[asyncio.Lock]) -> None:
    """Освобождает захваченный Lock.

    Args:
        lock (Optional[asyncio.Lock]): Объект блокировки.
    """
    if lock and lock.locked():
        lock.release()


async def ytdlp_extract(
        url_or_query: str, ydl_opts: Dict[str, Any], download: bool
) -> Dict[str, Any]:
    """Вызывает yt-dlp (извлечение или скачивание) в отдельном потоке.

    Args:
        url_or_query (str): URL или поисковый запрос.
        ydl_opts (Dict[str, Any]): Опции yt-dlp.
        download (bool): True для скачивания, False для извлечения.

    Returns:
        Dict[str, Any]: Результат extract_info.
    """

    def _run() -> Dict[str, Any]:
        with YoutubeDL(ydl_opts) as ydl:
            return ydl.extract_info(url_or_query, download=download)

    return await asyncio.to_thread(_run)


def format_duration_hms(dur_any: Optional[Any]) -> str:
    """Форматирует длительность в мм:сс или чч:мм:сс.

    Args:
        dur_any (Optional[Any]): Длительность в секундах.

    Returns:
        str: Форматированная строка или '—'.
    """
    if isinstance(dur_any, (int, float)) and dur_any >= 0:
        sec = int(dur_any)
        h, rem = divmod(sec, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
    return "—"


async def extract_basic_info(url: str, cookies_path: Optional[str] = None) -> Dict[str, Any]:
    """Извлекает базовую информацию без скачивания.

    Args:
        url (str): URL ресурса.
        cookies_path (Optional[str]): Путь к cookies.txt.

    Returns:
        Dict[str, Any]: title, duration, channel, thumbnail.
    """
    ydl_opts: Dict[str, Any] = {
        "quiet": True,
        "skip_download": True,
        "noplaylist": False,
        "playlist_items": "1",
        "logger": logging.getLogger("yt_dlp"),
    }
    if cookies_path and os.path.exists(cookies_path):
        ydl_opts["cookiefile"] = cookies_path
    info = await ytdlp_extract(url, ydl_opts, download=False)
    item = info
    try:
        entries = info.get("entries") if isinstance(info, dict) else None
        if isinstance(entries, list) and entries:
            item = entries[0]
    except Exception:
        pass

    def _pick_thumb(it: Dict[str, Any]) -> Optional[str]:
        t = it.get("thumbnail")
        if t:
            return t
        ts = it.get("thumbnails")
        if isinstance(ts, list) and ts:
            # пробуем выбрать самый приоритетный/большой
            def key_fn(x: Dict[str, Any]) -> Tuple[int, int, int]:
                pref = int(x.get("preference") or 0)
                w = int(x.get("width") or 0)
                h = int(x.get("height") or 0)
                return (pref, w * h, w + h)

            try:
                ts_sorted = sorted(ts, key=key_fn, reverse=True)
                return ts_sorted[0].get("url")
            except Exception:
                with suppress(Exception):
                    return ts[-1].get("url")
        return None

    title = (
            (item.get("title") if isinstance(item, dict) else None)
            or (item.get("fulltitle") if isinstance(item, dict) else None)
            or (item.get("id") if isinstance(item, dict) else None)
            or "Без названия"
    )
    duration = (item.get("duration") if isinstance(item, dict) else None)
    channel = ""
    if isinstance(item, dict):
        channel = item.get("uploader") or item.get("channel") or ""
    thumbnail = _pick_thumb(item if isinstance(item, dict) else {})

    return {"title": title, "duration": duration, "channel": channel, "thumbnail": thumbnail}


async def search_tracks(query: str, cookies_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Ищет треки на YouTube и фильтрует по длительности.

    Args:
        query (str): Поисковая строка.
        cookies_path (Optional[str]): Путь к cookies.txt.

    Returns:
        List[Dict[str, Any]]: Список словарей (title, url, duration, channel).
    """
    ydl_opts: Dict[str, Any] = {
        "quiet": True,
        "skip_download": True,
        "noplaylist": True,
        "default_search": "ytsearch",
    }
    if cookies_path and os.path.exists(cookies_path):
        ydl_opts["cookiefile"] = cookies_path

    info = await ytdlp_extract(f"ytsearch{MAX_RESULTS}:{query}", ydl_opts, download=False)
    entries = info.get("entries") or []
    results: List[Dict[str, Any]] = []
    for e in entries:
        duration = e.get("duration")
        if isinstance(duration, (int, float)) and duration > DURATION_LIMIT_SEC:
            continue
        url = e.get("webpage_url") or e.get("url")
        if not url and e.get("id"):
            url = f"https://www.youtube.com/watch?v={e['id']}"
        title = e.get("title") or "Без названия"
        channel = e.get("uploader") or e.get("channel") or ""
        results.append(
            {"title": title, "url": url, "duration": duration, "channel": channel}
        )
    return results


def find_files_by_exts(root: str, exts: Set[str]) -> List[str]:
    """Находит файлы с указанными расширениями.

    Args:
        root (str): Корневая директория.
        exts (Set[str]): Набор расширений (с точкой).

    Returns:
        List[str]: Пути найденных файлов.
    """
    out: List[str] = []
    for base, _, files in os.walk(root):
        for name in files:
            if os.path.splitext(name)[1].lower() in exts:
                out.append(os.path.join(base, name))
    return out


def find_audio_files(root: str) -> List[str]:
    """Находит аудиофайлы.

    Args:
        root (str): Корень поиска.

    Returns:
        List[str]: Пути аудио.
    """
    return find_files_by_exts(root, AUDIO_EXTS)


def find_video_files(root: str) -> List[str]:
    """Находит видеофайлы.

    Args:
        root (str): Корень поиска.

    Returns:
        List[str]: Пути видео.
    """
    return find_files_by_exts(root, VIDEO_EXTS)


def find_image_files(root: str) -> List[str]:
    """Находит изображения.

    Args:
        root (str): Корень поиска.

    Returns:
        List[str]: Пути изображений.
    """
    return find_files_by_exts(root, IMAGE_EXTS)


def process_thumbnail(src_path: str, out_dir: str) -> Optional[str]:
    """Готовит миниатюру: 320x320 JPEG ≤ заданного лимита.

    Args:
        src_path (str): Исходный файл.
        out_dir (str): Директория назначения.

    Returns:
        Optional[str]: Путь к миниатюре или None.
    """
    try:
        with Image.open(src_path) as im:
            im = im.convert("RGB")
            im = ImageOps.fit(
                im, THUMB_SIZE, method=Resampling.LANCZOS
            )
            quality = 90
            min_q = 40
            step = 5
            out_path = os.path.join(
                out_dir,
                f"{os.path.splitext(os.path.basename(src_path))[0]}_320.jpg",
            )
            last_size: Optional[int] = None
            while quality >= min_q:
                buf = io.BytesIO()
                im.save(
                    buf,
                    format="JPEG",
                    quality=quality,
                    optimize=True,
                    progressive=True,
                    subsampling="4:2:0",
                )
                size = buf.tell()
                if size <= THUMB_MAX_BYTES:
                    with open(out_path, "wb") as f:
                        f.write(buf.getvalue())
                    logging.getLogger("bot").info(
                        "Подготовлена обложка %s (%dx%d, %d байт, quality=%d)",
                        out_path,
                        THUMB_SIZE[0],
                        THUMB_SIZE[1],
                        size,
                        quality,
                    )
                    return out_path
                last_size = size
                quality -= step
            logging.getLogger("bot").warning(
                "Не удалось сжать обложку до %d байт, пропускаю (минимальное качество %d, размер %d байт)",
                THUMB_MAX_BYTES,
                min_q,
                last_size or -1,
            )
            return None
    except Exception as e:
        logging.getLogger("bot").warning(
            "Не удалось обработать обложку %s: %s", src_path, e
        )
        return None


def norm_base(path: str) -> str:
    """Возвращает имя файла без расширения и хвоста после '#'.

    Args:
        path (str): Путь к файлу.

    Returns:
        str: Базовое имя.
    """
    name = os.path.basename(path)
    name = name.split("#", 1)[0]
    base, _ = os.path.splitext(name)
    return base


def extract_id_from_base(base: str) -> Optional[str]:
    """Извлекает ID из квадратных скобок.

    Args:
        base (str): Базовое имя.

    Returns:
        Optional[str]: Извлечённый ID или None.
    """
    m = re.search(r"\[([0-9A-Za-z_-]{6,})\]", base)
    return m.group(1) if m else None


def make_duration_match_filter(max_seconds: int) -> Callable[[Dict[str, Any]], Optional[str]]:
    """Создаёт фильтр yt-dlp, отвергающий слишком длинные записи.

    Args:
        max_seconds (int): Максимальная длительность.

    Returns:
        Callable[[Dict[str, Any]], Optional[str]]: Фильтр (строка-причина или None).
    """

    def _mf(info: Dict[str, Any]) -> Optional[str]:
        dur = info.get("duration")
        if isinstance(dur, (int, float)) and dur > max_seconds:
            return f"duration>{max_seconds}"
        return None

    return _mf


async def download_media_to_temp(url: str, mode: str, cookies_path: Optional[str] = None) -> List[
    Tuple[str, Optional[str]]]:
    """Скачивает медиа и подготавливает миниатюры во временные директории.

    Args:
        url (str): Ссылка.
        mode (str): Режим ('audio'|'video'|'video_nosound').
        cookies_path (Optional[str]): Путь к cookies.txt.

    Returns:
        List[Tuple[str, Optional[str]]]: Пары (путь к медиа, путь к миниатюре или None).
    """
    tmpdir = tempfile.mkdtemp(prefix="dl_")
    if mode == "audio":
        postprocessors = [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            },
            {"key": "FFmpegThumbnailsConvertor", "format": "jpg"},
            {"key": "EmbedThumbnail"},
            {"key": "FFmpegMetadata"},
        ]
        ydl_format = "bestaudio/best"
        extra: Dict[str, Any] = {}
    elif mode == "video":
        postprocessors = [
            {"key": "FFmpegThumbnailsConvertor", "format": "jpg"},
            {"key": "FFmpegMetadata"},
        ]
        ydl_format = "bv*+ba/b"
        extra = {"merge_output_format": "mp4", "recode_video": "mp4"}
    else:
        postprocessors = [
            {"key": "FFmpegThumbnailsConvertor", "format": "jpg"},
            {"key": "FFmpegMetadata"},
        ]
        ydl_format = "bestvideo/best"
        extra = {"recode_video": "mp4"}

    ydl_opts: Dict[str, Any] = {
        "quiet": True,
        "format": ydl_format,
        "outtmpl": os.path.join(tmpdir, "%(title)s [%(id)s].%(ext)s"),
        "noplaylist": False,
        "postprocessors": postprocessors,
        "writethumbnail": True,
        "write_all_thumbnails": True,
        "convert_thumbnails": "jpg",
        "prefer_ffmpeg": True,
        "nocheckcertificate": True,
        "logger": logging.getLogger("yt_dlp"),
        "playlist_items": f"1-{MAX_PLAYLIST_ITEMS}",
        "max_downloads": MAX_PLAYLIST_ITEMS,
        "match_filter": make_duration_match_filter(DURATION_LIMIT_SEC),
        **extra,
    }
    if cookies_path and os.path.exists(cookies_path):
        ydl_opts["cookiefile"] = cookies_path

    async with download_sem:
        try:
            logger.info("Начало загрузки (%s): %s", mode, url)
            await ytdlp_extract(url, ydl_opts, download=True)
        except DownloadError as e:
            raise e
        except Exception as e:
            raise e

    if mode == "audio":
        media_files = find_audio_files(tmpdir)
    else:
        media_files = find_video_files(tmpdir)
    image_files = find_image_files(tmpdir)
    logger.info(
        "Файлов найдено (media=%d, images=%d)", len(media_files), len(image_files)
    )
    if not media_files:
        shutil.rmtree(tmpdir, ignore_errors=True)
        return []

    stable_dir = tempfile.mkdtemp(prefix="out_")

    images_by_base: Dict[str, List[str]] = {}
    for img in image_files:
        clean_base = norm_base(img)
        images_by_base.setdefault(clean_base, []).append(img)

    items: List[Tuple[str, Optional[str]]] = []
    for m in media_files:
        m_base = norm_base(m)
        m_dst = os.path.join(stable_dir, os.path.basename(m))
        with suppress(Exception):
            shutil.move(m, m_dst)

        possible_imgs = list(images_by_base.get(m_base, []))
        if not possible_imgs:
            vid = extract_id_from_base(m_base)
            if vid:
                needle = f"[{vid}]"
                for img in image_files:
                    name_wo_hash = os.path.basename(img).split("#", 1)[0]
                    if needle in name_wo_hash:
                        possible_imgs.append(img)

        t_src: Optional[str] = None
        if possible_imgs:
            with suppress(Exception):
                possible_imgs.sort(key=lambda p: os.path.getsize(p), reverse=True)
            t_src = possible_imgs[0]

        t_dst: Optional[str] = None
        if t_src and os.path.exists(t_src):
            moved = os.path.join(stable_dir, os.path.basename(t_src))
            with suppress(Exception):
                shutil.move(t_src, moved)
            logger.info("Обрабатываю обложку: %s", moved)
            processed = process_thumbnail(moved, stable_dir)
            if os.path.exists(moved) and (not processed or processed != moved):
                with suppress(Exception):
                    os.remove(moved)
            if processed and os.path.exists(processed):
                t_dst = processed

        items.append((m_dst, t_dst))

    shutil.rmtree(tmpdir, ignore_errors=True)
    return items


async def send_media_files(
        bot: Bot,
        chat_id: int,
        items: List[Tuple[str, Optional[str]]],
        method: str,
        media_arg: str,
        extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Отправляет файлы по одному.

    Args:
        bot (Bot): Экземпляр бота.
        chat_id (int): ID чата.
        items (List[Tuple[str, Optional[str]]]): Список медиа.
        method (str): Метод Telegram API.
        media_arg (str): Аргумент ('audio'|'video').
        extra (Optional[Dict[str, Any]]): Дополнительные параметры.
    """
    for media_path, thumb_path in items:
        try:
            title = os.path.splitext(os.path.basename(media_path))[0]
            caption = make_caption(title)

            with suppress(Exception):
                size = os.path.getsize(media_path)
                if size and size > TG_MAX_UPLOAD_BYTES:
                    size_mb = size / (1024 * 1024)
                    lim_mb = TG_MAX_UPLOAD_BYTES / (1024 * 1024)
                    logger.info(
                        "Готовлюсь отправлять файл: %s (%.2f МБ, лимит %.0f МБ)",
                        title,
                        size_mb,
                        lim_mb,
                    )
                    if size > TG_MAX_UPLOAD_BYTES:
                        logger.info(
                            "Пропускаю файл: %s (%.2f МБ) — превышает лимит Telegram (%.0f МБ)",
                            title,
                            size_mb,
                            lim_mb,
                        )
                        await bot.send_message(
                            chat_id,
                            f"⚠️ Файл «{caption}» ({size_mb:.1f} МБ) превышает лимит Telegram ({lim_mb:.0f} МБ). Пропускаю.",
                        )
                        continue

            kwargs: Dict[str, Any] = {
                "chat_id": chat_id,
                "caption": caption,
                "parse_mode": None,
                media_arg: FSInputFile(media_path),
            }
            if thumb_path and os.path.exists(thumb_path):
                kwargs["thumbnail"] = FSInputFile(thumb_path)
            if extra:
                kwargs.update(extra)
            await getattr(bot, method)(**kwargs)
        finally:
            with suppress(Exception):
                os.remove(media_path)
            if thumb_path:
                with suppress(Exception):
                    os.remove(thumb_path)
            await asyncio.sleep(0.3)

    parents = {os.path.dirname(p) for p, _ in items}
    for d in parents:
        base = os.path.basename(d)
        if base.startswith("out_"):
            with suppress(Exception):
                shutil.rmtree(d, ignore_errors=True)


async def send_audio_files(
        bot: Bot, chat_id: int, items: List[Tuple[str, Optional[str]]]
) -> None:
    """Отправляет аудиофайлы.

    Args:
        bot (Bot): Экземпляр бота.
        chat_id (int): ID чата.
        items (List[Tuple[str, Optional[str]]]): Список медиа.
    """
    await send_media_files(bot, chat_id, items, method="send_audio", media_arg="audio")


async def send_video_files(
        bot: Bot, chat_id: int, items: List[Tuple[str, Optional[str]]]
) -> None:
    """Отправляет видеофайлы.

    Args:
        bot (Bot): Экземпляр бота.
        chat_id (int): ID чата.
        items (List[Tuple[str, Optional[str]]]): Список медиа.
    """
    await send_media_files(
        bot,
        chat_id,
        items,
        method="send_video",
        media_arg="video",
        extra={"supports_streaming": True},
    )


async def send_by_mode(
        bot: Bot, chat_id: int, mode: str, items: List[Tuple[str, Optional[str]]]
) -> None:
    """Выбирает способ отправки по режиму.

    Args:
        bot (Bot): Экземпляр бота.
        chat_id (int): ID чата.
        mode (str): Режим.
        items (List[Tuple[str, Optional[str]]]): Медиа.
    """
    if mode == "audio":
        await send_audio_files(bot, chat_id, items)
    else:
        await send_video_files(bot, chat_id, items)


def remember_cookie_request(user_id: int, kind: str, url: Optional[str] = None, mode: Optional[str] = None) -> None:
    """Сохраняет ожидание cookies.

    Args:
        user_id (int): Пользователь.
        kind (str): Тип ('download'|'search').
        url (Optional[str]): URL для повтора.
        mode (Optional[str]): Режим ('audio'|'video'|'video_nosound'|'auto').
    """
    payload: Dict[str, Any] = {"kind": kind, "asked": True}
    if url:
        payload["url"] = url
    if mode:
        payload["mode"] = mode
    AWAITING_COOKIES[user_id] = payload


def remember_search_cookie_request(user_id: int, query: str) -> None:
    """Сохраняет ожидание cookies для поиска.

    Args:
        user_id (int): Пользователь.
        query (str): Поисковый запрос.
    """
    AWAITING_COOKIES[user_id] = {"kind": "search", "query": query, "asked": True}


def get_user_cookies_path(user_id: int) -> str:
    """Возвращает путь к cookies.txt пользователя.

    Args:
        user_id (int): Идентификатор пользователя.

    Returns:
        str: Путь к cookies.txt.
    """
    return os.path.join(COOKIES_DIR, f"{user_id}_cookies.txt")


@router.message(CommandStart())
async def cmd_start(msg: Message) -> None:
    """Команда /start — сбрасывает состояние и показывает инструкцию.

    Args:
        msg (Message): Входящее сообщение.
    """
    uid = msg.from_user.id if msg.from_user is not None else None
    if uid is not None:
        USER_SEARCHES.pop(uid, None)
        AWAITING_COOKIES.pop(uid, None)
    logger.info("Команда /start от пользователя %s", str(uid))
    await msg.answer(
        "✨ Отправьте ссылку — скачаю по вашим настройкам.\n"
        "📝 Или отправьте название — покажу список из 25 результатов.\n"
        "🍪 Если нужен доступ — пришлите файл cookies.txt.",
        reply_markup=build_main_reply_kb(),
    )


@router.message(Command("menu"))
async def cmd_menu(msg: Message) -> None:
    await cmd_start(msg)


@router.message(Command("help"))
async def cmd_help(msg: Message) -> None:
    """Команда /help — краткая справка.

    Args:
        msg (Message): Сообщение команды.
    """
    logger.info("Команда /help от пользователя %s", str(msg.from_user.id if msg.from_user else None))
    await msg.answer(
        "ℹ️ Как пользоваться:\n"
        "• 🔗 Ссылка → скачивание по выбранному режиму.\n"
        "• 🔎 Текст запроса → 25 результатов, 5 страниц по 5 кнопок.\n"
        "• ⚙️ /settings — сменить дефолтный тип скачивания.\n"
        "• 🍪 Если просит cookies — отправьте cookies.txt.",
        reply_markup=build_main_reply_kb(),
    )


@router.message(Command("settings"))
async def cmd_settings(msg: Message) -> None:
    """Открывает меню настроек.

    Args:
        msg (Message): Сообщение команды.
    """
    if msg.from_user is None:
        await msg.answer(
            "⚙️ Настройки недоступны для этого типа сообщения.",
            reply_markup=build_main_reply_kb(),
        )
        return
    logger.info("Открытие настроек пользователем %s", str(msg.from_user.id))
    await msg.answer(
        "⚙️ Настройки типа скачивания:",
        reply_markup=build_settings_kb(msg.from_user.id).as_markup(),
    )


@router.callback_query(F.data == "settings:open")
async def cb_settings_open(cb: CallbackQuery) -> None:
    """Callback открытия настроек.

    Args:
        cb (CallbackQuery): Запрос.
    """
    await try_cb_answer(cb)
    if cb.from_user is None:
        return
    if cb.message is not None and isinstance(cb.message, Message):
        await cb.message.answer(
            "⚙️ Настройки типа скачивания:",
            reply_markup=build_settings_kb(cb.from_user.id).as_markup(),
        )


@router.callback_query(F.data == "settings:close")
async def cb_settings_close(cb: CallbackQuery) -> None:
    """Callback закрытия настроек.

    Args:
        cb (CallbackQuery): Запрос.
    """
    await try_cb_answer(cb)
    if cb.message is not None and isinstance(cb.message, Message):
        with suppress(Exception):
            await cb.message.delete()
        with suppress(Exception):
            await cb.message.edit_reply_markup(reply_markup=None)


@router.callback_query(F.data.startswith("setmode:"))
async def cb_set_mode(cb: CallbackQuery) -> None:
    """Выбор режима скачивания.

    Args:
        cb (CallbackQuery): Запрос с режимом.
    """
    data = cb.data or ""
    if not data.startswith("setmode:"):
        await try_cb_answer(cb, "⚠️ Некорректные данные.")
        return
    mode = data.split(":", 1)[1]
    if mode not in {"auto", "audio", "video", "video_nosound"}:
        await cb.answer("⚠️ Неизвестный режим.")
        return
    if cb.from_user is None:
        await cb.answer("⚠️ Не удалось определить пользователя.")
        return
    set_user_mode(cb.from_user.id, mode)
    logger.info("Режим пользователя %s изменён на %s", cb.from_user.id, mode)
    kb = build_settings_kb(cb.from_user.id)
    if cb.message is not None and isinstance(cb.message, Message):
        with suppress(Exception):
            await cb.message.edit_reply_markup(reply_markup=kb.as_markup())
    await cb.answer("✅ Режим обновлён.")


@router.callback_query(F.data.startswith("dl:"))
async def cb_download_choice(cb: CallbackQuery, bot: Bot) -> None:
    """Обрабатывает выбор режима скачивания для сохранённого URL.

    Args:
        cb (CallbackQuery): Callback с данными вида dl:<mode>:<token>.
        bot (Bot): Экземпляр бота.
    """
    data = cb.data or ""
    parts = data.split(":")
    if len(parts) != 3:
        await try_cb_answer(cb, "⚠️ Некорректные данные.")
        return
    _, mode_sel, token = parts
    if mode_sel not in {"audio", "video", "auto"}:
        await try_cb_answer(cb, "⚠️ Неизвестный режим.")
        return
    pend = PENDING_DOWNLOADS.get(token)
    if not pend:
        await try_cb_answer(cb, "ℹ️ Ссылка устарела. Отправьте её снова.")
        return
    user_id = pend.get("user_id")
    url = pend.get("url")
    if not isinstance(user_id, int) or not isinstance(url, str):
        await try_cb_answer(cb, "⚠️ Ошибка данных.")
        return

    with suppress(Exception):
        PENDING_DOWNLOADS.pop(token, None)

    if mode_sel == "auto":
        mode = decide_effective_mode(get_user_mode(user_id), url)
    else:
        mode = mode_sel

    logger.info("Выбор скачивания: user=%s, mode=%s, url=%s", str(user_id), mode, url[:200])

    if cb.message is not None and isinstance(cb.message, Message):
        with suppress(Exception):
            await cb.message.edit_reply_markup(reply_markup=None)

    lock = await begin_user_download(user_id)
    if not lock:
        await try_cb_answer(cb, "⏳ Идёт другая загрузка.")
        return

    chat_id = get_cb_chat_id(cb)
    if chat_id is None:
        end_user_download(lock)
        await try_cb_answer(cb)
        return

    await try_cb_answer(cb)
    await bot.send_message(chat_id, "⏳ Скачиваю, подождите...")
    try:
        cookies_path = get_user_cookies_path(user_id)
        files = await download_media_to_temp(url, mode=mode, cookies_path=cookies_path)
        if not files:
            logger.info("Загрузка завершена: нечего отправлять (user=%s, mode=%s)", str(user_id), mode)
            await bot.send_message(
                chat_id,
                "😕 Нечего отправлять. Возможно, превышен лимит длительности (30 минут).",
            )
            return
        logger.info("Загрузка завершена: файлов к отправке %d (user=%s, mode=%s)", len(files), str(user_id), mode)
        await send_by_mode(bot, chat_id, mode, files)
        logger.info("Отправка завершена: отправлено %d файлов (user=%s, mode=%s)", len(files), str(user_id), mode)
    except DownloadError:
        logger.info("Загрузка требует cookies (user=%s, mode=%s)", str(user_id), mode)
        remember_cookie_request(user_id, kind="download", url=url, mode=mode)
        await bot.send_message(
            chat_id,
            "🍪 Источник требует cookies или произошла ошибка.\nПришлите файл cookies.txt для повтора попытки.",
        )
    except Exception:
        logger.info("Ошибка при загрузке (user=%s, mode=%s)", str(user_id), mode)
        await bot.send_message(chat_id, "❌ Произошла ошибка при загрузке. Попробуйте позже.")
    finally:
        end_user_download(lock)


async def send_info_card(
        bot: Bot,
        chat_id: int,
        url: str,
        user_id: int,
        reply_markup: Optional[Any] = None,
) -> None:
    """Отправляет карточку найденного файла.
    """
    caption_fallback = "🎧 Файл найден:\n\nВыберите, что скачать для этой ссылки:"
    try:
        logger.info("Показываю карточку информации (user=%s, url=%s)", str(user_id), url[:200])
        info = await extract_basic_info(url, cookies_path=get_user_cookies_path(user_id))
        title = str(info.get("title") or "Без названия")
        dur_s = info.get("duration")
        dur_str = format_duration_hms(dur_s)
        channel = str(info.get("channel") or "")
        show_channel = is_youtube_url(url) and bool(channel)
        parts = [
            "🎧 Файл найден:",
            "",
            f"Название: {title}",
        ]
        if show_channel:
            parts.append(f"Канал: {channel}")
        parts.append(f"Длительность: {dur_str}")
        parts.append("")
        parts.append("Выберите, что скачать для этой ссылки:")
        caption = make_multiline_caption("\n".join(parts))
        thumb_url = info.get("thumbnail")
        if isinstance(thumb_url, str) and thumb_url.strip():
            with suppress(Exception):
                await bot.send_photo(
                    chat_id=chat_id,
                    photo=thumb_url.strip(),
                    caption=caption,
                    parse_mode=None,
                    reply_markup=reply_markup,
                )
                return
        await bot.send_message(
            chat_id,
            caption,
            parse_mode=None,
            reply_markup=reply_markup,
        )
    except Exception:
        await bot.send_message(
            chat_id,
            caption_fallback,
            parse_mode=None,
            reply_markup=reply_markup,
        )


@router.message(F.text)
async def handle_text(msg: Message, bot: Bot) -> None:
    """Обрабатывает текст: команды/кнопки, URL (меню скачивания) или поиск.

    Args:
        msg (Message): Входящее сообщение.
        bot (Bot): Экземпляр бота.
    """
    raw = (msg.text or "").strip()
    intent = parse_main_button_intent(raw)
    if intent == "menu":
        await cmd_start(msg)
        return
    if intent == "help":
        await cmd_help(msg)
        return
    if intent == "settings":
        await cmd_settings(msg)
        return

    url = raw
    uid = msg.from_user.id if msg.from_user is not None else None
    logger.info("Запрос от %s: %s", str(uid), url[:200] if url else "")
    if not url:
        await msg.answer("⚠️ Пустой запрос.")
        return
    if is_url(url):
        logger.info("Обнаружена ссылка. Показываю карточку выбора (user=%s)", str(uid))
        if uid is None:
            logger.info("Не удалось определить пользователя для ссылки.")
            await msg.answer("⚠️ Не удалось определить пользователя.")
            return
        token = save_pending_url(uid, url)
        kb = build_download_choice_kb(uid, token)
        await send_info_card(
            bot,
            msg.chat.id,
            url,
            uid,
            reply_markup=kb.as_markup()
        )
        return
    query = sanitize_query(url)
    if not query:
        logger.info("Пустой или некорректный поисковый запрос (user=%s)", str(uid))
        await msg.answer("⚠️ Некорректный запрос.")
        return
    logger.info("Начинаю поиск (user=%s, query=%s)", str(uid), query[:120])
    await msg.answer("🔎 Ищу...")
    try:
        cookies_path = get_user_cookies_path(uid) if uid is not None else None
        results = await search_tracks(query, cookies_path=cookies_path)
        logger.info("Поиск завершён: найдено %d (user=%s)", len(results), str(uid))
        if uid is not None:
            USER_SEARCHES[uid] = {"results": results, "page": 0}
        if not results:
            logger.info("Ничего не найдено (user=%s)", str(uid))
            await msg.answer("🙁 Ничего не найдено (или превышен лимит длительности).")
            return
        kb = build_results_kb(uid if uid is not None else 0)
        logger.info("Показываю результаты поиска (user=%s)", str(uid))
        await msg.answer("📋 Результаты поиска:", reply_markup=kb.as_markup())
    except DownloadError as e:
        if uid is not None:
            remember_search_cookie_request(uid, query)
        logger.info('Поиск требует cookies (user=%s): %s', str(uid), str(e))
        await msg.answer(
            "🍪 Источник требует cookies или защиту (YouTube может просить вход).\n"
            "Пришлите файл cookies.txt — повторю поиск с cookies."
        )
    except Exception as e:
        logger.info('Ошибка поиска для "%s": %s', query, str(e))
        await msg.answer("❌ Ошибка поиска. Попробуйте позже.")


@router.callback_query(F.data == "noop")
async def handle_noop(cb: CallbackQuery) -> None:
    """Пустой callback.

    Args:
        cb (CallbackQuery): Запрос.
    """
    await try_cb_answer(cb)


@router.callback_query(F.data == "cancel")
async def handle_cancel(cb: CallbackQuery) -> None:
    """Отмена списка результатов и ожидания cookies.

    Args:
        cb (CallbackQuery): Запрос.
    """
    if cb.from_user is not None:
        USER_SEARCHES.pop(cb.from_user.id, None)
        AWAITING_COOKIES.pop(cb.from_user.id, None)
    if cb.message is not None and isinstance(cb.message, Message):
        with suppress(Exception):
            await cb.message.edit_reply_markup(reply_markup=None)
    await try_cb_answer(cb, "❌ Отменено.")


@router.callback_query(F.data == "page:next")
async def handle_next_page(cb: CallbackQuery) -> None:
    """Переход к следующей странице результатов.

    Args:
        cb (CallbackQuery): Запрос.
    """
    if cb.from_user is None:
        await try_cb_answer(cb, "ℹ️ Нет пользователя.")
        return
    state = USER_SEARCHES.get(cb.from_user.id)
    if not state:
        await try_cb_answer(cb, "ℹ️ Нет активного списка.")
        return
    results = state["results"]
    page = state.get("page", 0)
    _, pages = slice_page(results, page, PAGE_SIZE)
    state["page"] = (page + 1) % pages
    kb = build_results_kb(cb.from_user.id)
    if cb.message is not None and isinstance(cb.message, Message):
        with suppress(Exception):
            await cb.message.edit_reply_markup(reply_markup=kb.as_markup())
    await try_cb_answer(cb)


@router.callback_query(F.data == "page:prev")
async def handle_prev_page(cb: CallbackQuery) -> None:
    """Переход к предыдущей странице результатов.

    Args:
        cb (CallbackQuery): Запрос.
    """
    if cb.from_user is None:
        await try_cb_answer(cb, "ℹ️ Нет пользователя.")
        return
    state = USER_SEARCHES.get(cb.from_user.id)
    if not state:
        await try_cb_answer(cb, "ℹ️ Нет активного списка.")
        return
    results = state["results"]
    page = state.get("page", 0)
    _, pages = slice_page(results, page, PAGE_SIZE)
    state["page"] = (page - 1 + pages) % pages
    kb = build_results_kb(cb.from_user.id)
    if cb.message is not None and isinstance(cb.message, Message):
        with suppress(Exception):
            await cb.message.edit_reply_markup(reply_markup=kb.as_markup())
    await try_cb_answer(cb)


@router.callback_query(F.data.startswith("pick:"))
async def handle_pick(cb: CallbackQuery, bot: Bot) -> None:
    """Начинает загрузку выбранного результата.
    """
    data = cb.data or ""
    if ":" not in data:
        await try_cb_answer(cb, "⚠️ Некорректные данные.")
        return
    idx_str = data.split(":", 1)[1]
    with suppress(ValueError):
        idx = int(idx_str)
        if cb.from_user is None:
            await try_cb_answer(cb, "ℹ️ Не удалось определить пользователя.")
            return
        state = USER_SEARCHES.get(cb.from_user.id)
        if not state:
            await try_cb_answer(cb, "ℹ️ Список результатов устарел.")
            return
        results: List[Dict[str, Any]] = state["results"]
        if idx < 0 or idx >= len(results):
            await try_cb_answer(cb, "⚠️ Некорректный выбор.")
            return
        url = results[idx].get("url")
        if not url:
            await try_cb_answer(cb, "⚠️ Нет URL для выбранного трека.")
            return

        logger.info("Выбор результата #%d пользователем %s: %s", idx, cb.from_user.id, (url or "")[:200])

        token = save_pending_url(cb.from_user.id, url)
        kb = build_download_choice_kb(cb.from_user.id, token)

        await try_cb_answer(cb)

        with suppress(Exception):
            USER_SEARCHES.pop(cb.from_user.id, None)
        if cb.message is not None and isinstance(cb.message, Message):
            with suppress(Exception):
                await cb.message.delete()
            with suppress(Exception):
                await cb.message.edit_reply_markup(reply_markup=None)

        chat_id = get_cb_chat_id(cb)
        if chat_id is not None:
            await send_info_card(
                bot,
                chat_id,
                url,
                cb.from_user.id,
                reply_markup=kb.as_markup(),
            )
        return


@router.message(F.document)
async def handle_document(msg: Message, bot: Bot) -> None:
    """Обрабатывает загрузку cookies.txt и повторяет операцию.
    """
    if msg.from_user is None:
        logger.info("Получен файл, но не удалось определить пользователя.")
        await msg.answer("📄 Файл получен, но не удалось определить пользователя.")
        return
    pending = AWAITING_COOKIES.get(msg.from_user.id)
    if not pending:
        logger.info("Получен файл от %s, но cookies не требуются.", msg.from_user.id)
        await msg.answer("📄 Файл получен, но сейчас cookies не требуются.")
        return

    cookies_path = get_user_cookies_path(msg.from_user.id)
    doc = msg.document
    if doc is None:
        logger.info("Не удалось прочитать файл cookies от %s.", msg.from_user.id)
        await msg.answer("❌ Не удалось прочитать файл.")
        return

    name_l = (doc.file_name or "").lower()
    ext = os.path.splitext(name_l)[1]
    size = doc.file_size or 0
    logger.info("Получен файл cookies от %s: %s (%d байт)", msg.from_user.id, doc.file_name, size)
    if ext not in ALLOWED_COOKIES_EXTS:
        logger.info("Некорректный формат файла cookies от %s: %s", msg.from_user.id, ext)
        await msg.answer("⚠️ Нужен файл cookies в формате Netscape: cookies.txt.")
        return
    if size and size > COOKIES_MAX_BYTES:
        lim_mb = COOKIES_MAX_BYTES / (1024 * 1024)
        cur_mb = size / (1024 * 1024)
        logger.info("Слишком большой файл cookies от %s: %.2f МБ (лимит %.0f МБ)", msg.from_user.id, cur_mb, lim_mb)
        await msg.answer(
            f"⚠️ Слишком большой cookies.txt ({cur_mb:.1f} МБ). Максимум {lim_mb:.0f} МБ."
        )
        return

    try:
        await bot.download(doc, destination=cookies_path)
        with suppress(Exception):
            real_size = os.path.getsize(cookies_path)
            logger.info("Cookies сохранены для %s: %s (%d байт)", msg.from_user.id, cookies_path, real_size)
    except Exception:
        logger.info("Не удалось сохранить файл cookies от %s.", msg.from_user.id)
        await msg.answer("❌ Не удалось сохранить cookies.txt.")
        return

    with suppress(Exception):
        real_size = os.path.getsize(cookies_path)
        if real_size > COOKIES_MAX_BYTES:
            lim_mb = COOKIES_MAX_BYTES / (1024 * 1024)
            cur_mb = real_size / (1024 * 1024)
            with suppress(Exception):
                os.remove(cookies_path)
            logger.info("Слишком большой сохранённый файл cookies от %s: %.2f МБ (лимит %.0f МБ)", msg.from_user.id, cur_mb, lim_mb)
            await msg.answer(
                f"⚠️ Слишком большой cookies.txt ({cur_mb:.1f} МБ). Максимум {lim_mb:.0f} МБ."
            )
            return

    logger.info("Повтор операции с cookies для %s.", msg.from_user.id)
    await msg.answer("🍪 Cookies получены. Пробую снова...")

    pending_kind = (pending.get("kind") or "").lower()
    if pending_kind == "search":
        query_any = pending.get("query")
        if not isinstance(query_any, str) or not query_any.strip():
            logger.info("Нет запроса для повтора поиска с cookies от %s.", msg.from_user.id)
            await msg.answer("❌ Нет запроса для повтора поиска.")
            return
        query = query_any.strip()
        logger.info("Повтор поиска с cookies (user=%s, query=%s)", msg.from_user.id, query[:120])
        AWAITING_COOKIES.pop(msg.from_user.id, None)
        try:
            results = await search_tracks(query, cookies_path=cookies_path)
            logger.info("Поиск с cookies: найдено %d (user=%s)", len(results), msg.from_user.id)
            USER_SEARCHES[msg.from_user.id] = {"results": results, "page": 0}
            if not results:
                logger.info("Ничего не найдено с cookies от %s.", msg.from_user.id)
                await msg.answer("🙁 Ничего не найдено даже с cookies.")
                return
            kb = build_results_kb(msg.from_user.id)
            logger.info("Показываю результаты поиска с cookies (user=%s)", msg.from_user.id)
            await msg.answer("📋 Результаты поиска:", reply_markup=kb.as_markup())
        except Exception:
            logger.info("Ошибка поиска с cookies от %s.", msg.from_user.id)
            await msg.answer("❌ Не удалось выполнить поиск даже с cookies.")
        return

    url_any = pending.get("url")
    if not isinstance(url_any, str) or not url_any:
        logger.info("Нет URL для повтора загрузки с cookies от %s.", msg.from_user.id)
        await msg.answer("❌ Нет URL для повтора.")
        return
    url = url_any

    pending_mode = pending.get("mode")
    if isinstance(pending_mode, str) and pending_mode in {"audio", "video", "video_nosound"}:
        mode = pending_mode
    elif pending_mode == "auto":
        mode = decide_effective_mode(get_user_mode(msg.from_user.id), url)
    else:
        mode = decide_effective_mode(get_user_mode(msg.from_user.id), url)

    logger.info("Повтор загрузки с cookies (user=%s, mode=%s, url=%s)", msg.from_user.id, mode, url[:200])

    AWAITING_COOKIES.pop(msg.from_user.id, None)
    lock = await begin_user_download(msg.from_user.id)
    if not lock:
        logger.info("Не удалось начать загрузку с cookies: другая загрузка идёт (user=%s)", msg.from_user.id)
        await msg.answer("⏳ Идёт другая загрузка. Дождитесь завершения.")
        return
    try:
        files = await download_media_to_temp(url, mode=mode, cookies_path=cookies_path)
        if not files:
            logger.info("Загрузка с cookies завершена: нечего отправлять (user=%s, mode=%s)", msg.from_user.id, mode)
            await msg.answer(
                "😕 Не удалось скачать даже с cookies (возможно, превышен лимит длительности)."
            )
            return
        logger.info("Загрузка с cookies завершена: файлов к отправке %d (user=%s, mode=%s)", len(files), msg.from_user.id, mode)
        await send_by_mode(bot, msg.chat.id, mode, files)
        logger.info("Отправка (cookies) завершена: отправлено %d файлов (user=%s, mode=%s)", len(files), msg.from_user.id, mode)
    except Exception:
        logger.info("Ошибка при загрузке с cookies (user=%s, mode=%s)", msg.from_user.id, mode)
        await msg.answer("❌ Не удалось скачать даже с cookies. Скипаю.")
    finally:
        end_user_download(lock)


async def main() -> None:
    """Точка входа приложения: настройка логирования и старт поллинга.

    Raises:
        RuntimeError: Если отсутствует BOT_TOKEN.
    """
    setup_logging()
    if not BOT_TOKEN:
        raise RuntimeError("Не задана переменная окружения BOT_TOKEN")
    bot = Bot(
        BOT_TOKEN,
        default=DefaultBotProperties(parse_mode="HTML"),
    )
    logger.info("Старт поллинга")
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
