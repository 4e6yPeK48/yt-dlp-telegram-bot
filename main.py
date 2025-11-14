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


# ========= Логирование =========
def setup_logging(log_dir: str = "logs") -> None:
    """Настраивает логирование: консоль и ротация по уровням.

    Args:
        log_dir: Директория для файлов логов.
    """
    os.makedirs(log_dir, exist_ok=True)
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

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
    root.addHandler(console)

    root.addHandler(make_rotating("app.debug.log", logging.DEBUG))
    root.addHandler(make_rotating("app.info.log", logging.INFO))
    root.addHandler(make_rotating("app.warn.log", logging.WARNING))
    root.addHandler(make_rotating("app.error.log", logging.ERROR))

    logging.getLogger("aiogram").setLevel(logging.INFO)
    logging.getLogger("aiohttp").setLevel(logging.INFO)
    logging.getLogger("yt_dlp").setLevel(logging.INFO)


logger: logging.Logger = logging.getLogger("bot")


# ========= Основная логика =========
def is_url(text: str) -> bool:
    """Проверяет, является ли строка URL со схемой http/https.

    Args:
        text: Исходная строка.

    Returns:
        True, если строка похожа на URL, иначе False.
    """
    with suppress(Exception):
        u = urlparse(text.strip())
        return u.scheme in {"http", "https"} and bool(u.netloc)
    return False


def slice_page(items: List[Any], page: int, page_size: int) -> Tuple[List[Any], int]:
    """Возвращает элементы выбранной страницы и общее число страниц.

    Args:
        items: Полный список элементов.
        page: Номер страницы (0-индексация).
        page_size: Размер страницы.

    Returns:
        Кортеж (элементы текущей страницы, всего страниц).
    """
    pages = max(1, math.ceil(len(items) / page_size))
    page = max(0, min(page, pages - 1))
    start = page * page_size
    end = start + page_size
    return items[start:end], pages


def get_user_mode(user_id: int) -> str:
    """Возвращает режим пользователя.

    Args:
        user_id: Идентификатор пользователя Telegram.

    Returns:
        Один из: 'auto', 'audio', 'video', 'video_nosound'.
    """
    st = USER_SETTINGS.get(user_id)
    return (st or {}).get("mode", "auto")


def set_user_mode(user_id: int, mode: str) -> None:
    """Сохраняет режим пользователя.

    Args:
        user_id: Идентификатор пользователя.
        mode: Режим ('auto'|'audio'|'video'|'video_nosound').
    """
    USER_SETTINGS[user_id] = {"mode": mode}


def is_audio_platform(url: str) -> bool:
    """Эвристика определения аудио-площадки.

    Args:
        url: URL ресурса.

    Returns:
        True, если сайт предположительно аудио-ориентированный.
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
        user_mode: Выбранный пользователем режим.
        url: URL источника.

    Returns:
        Режим ('audio'|'video'|'video_nosound').
    """
    if user_mode == "auto":
        return "audio" if is_audio_platform(url) else "video"
    return user_mode


def build_results_kb(user_id: int) -> InlineKeyboardBuilder:
    """Строит инлайн-клавиатуру с результатами поиска и пагинацией.

    Args:
        user_id: Идентификатор пользователя.

    Returns:
        Экземпляр InlineKeyboardBuilder.
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
    kb.row(InlineKeyboardButton(text="Настройки ⚙️", callback_data="settings:open"))
    kb.row(InlineKeyboardButton(text="❌ Отмена", callback_data="cancel"))
    return kb


def build_settings_kb(user_id: int) -> InlineKeyboardBuilder:
    """Строит инлайн-меню выбора режима скачивания.

    Args:
        user_id: Идентификатор пользователя.

    Returns:
        Экземпляр InlineKeyboardBuilder.
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


# ==== Новая постоянная стартовая клавиатура и меню настроек (ReplyKeyboard) ====
MAIN_BUTTONS: List[str] = ["/start", "/help", "/settings"]


def build_main_reply_kb() -> ReplyKeyboardMarkup:
    """Строит основную reply-клавиатуру.

    Returns:
        Разметка клавиатуры.
    """
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="/start"), KeyboardButton(text="/help")],
            [KeyboardButton(text="/settings")],
        ],
        resize_keyboard=True,
        is_persistent=True,
    )


async def try_cb_answer(cb: CallbackQuery, text: Optional[str] = None) -> None:
    """Безопасно отвечает на callback-запрос.

    Args:
        cb: Объект callback.
        text: Текст всплывающего уведомления.
    """
    with suppress(Exception):
        await cb.answer(text)


def get_cb_chat_id(cb: CallbackQuery) -> Optional[int]:
    """Безопасно получить chat_id из CallbackQuery.

    Args:
        cb: Объект callback-запроса.

    Returns:
        Идентификатор чата или None, если определить не удалось.

    Примечание:
        Если сообщение недоступно (InaccessibleMessage/None), используется from_user.id (личный чат).
    """
    msg_obj = cb.message
    if msg_obj is not None and isinstance(msg_obj, Message):
        return msg_obj.chat.id
    if cb.from_user is not None:
        return cb.from_user.id
    return None


def sanitize_query(text: str) -> str:
    """Санитизирует поисковый запрос: удаляет служебные символы и нормализует пробелы.

    Args:
        text: Исходная строка.

    Returns:
        Очищенный и усечённый запрос.
    """
    t = re.sub(r"[\x00-\x1f\x7f]", "", text)
    t = re.sub(r"[\u200B-\u200F\u202A-\u202E\u2060-\u206F]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > MAX_QUERY_LEN:
        t = t[:MAX_QUERY_LEN]
    return t


def make_caption(text: str, limit: int = CAPTION_MAX_LEN) -> str:
    t = re.sub(r"[\x00-\x1f\x7f]", "", text or "")
    t = re.sub(r"[\u200B-\u200F\u202A-\u202E\u2060-\u206F]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > limit:
        t = t[: limit - 1] + "…"
    return t


def get_user_lock(user_id: int) -> asyncio.Lock:
    """Возвращает или создаёт Lock для пользователя.

    Args:
        user_id: Идентификатор пользователя.

    Returns:
        Экземпляр asyncio.Lock.
    """
    lock = USER_LOCKS.get(user_id)
    if lock is None:
        lock = asyncio.Lock()
        USER_LOCKS[user_id] = lock
    return lock


async def begin_user_download(user_id: int) -> Optional[asyncio.Lock]:
    """Пытается захватить пользовательский Lock перед началом загрузки.

    Args:
        user_id: Идентификатор пользователя.

    Returns:
        Захваченный Lock или None, если уже занято.
    """
    lock = get_user_lock(user_id)
    if lock.locked():
        return None
    await lock.acquire()
    return lock


def end_user_download(lock: Optional[asyncio.Lock]) -> None:
    """Освобождает ранее захваченный Lock.

    Args:
        lock: Объект блокировки.
    """
    if lock and lock.locked():
        lock.release()


async def ytdlp_extract(
    url_or_query: str, ydl_opts: Dict[str, Any], download: bool
) -> Dict[str, Any]:
    """Выполняет извлечение/скачивание через yt-dlp в отдельном потоке.

    Args:
        url_or_query: URL или поисковый запрос.
        ydl_opts: Параметры yt-dlp.
        download: True для скачивания, False для извлечения метаданных.

    Returns:
        Словарь с информацией от yt-dlp.
    """

    def _run() -> Dict[str, Any]:
        with YoutubeDL(ydl_opts) as ydl:
            return ydl.extract_info(url_or_query, download=download)

    return await asyncio.to_thread(_run)


async def search_tracks(query: str) -> List[Dict[str, Any]]:
    """Ищет треки на YouTube и применяет ограничение по длительности.

    Args:
        query: Поисковая строка.

    Returns:
        Список словарей результатов: title, url, duration, channel.
    """
    ydl_opts: Dict[str, Any] = {
        "quiet": True,
        "skip_download": True,
        "noplaylist": True,
        "default_search": "ytsearch",
    }
    info = await ytdlp_extract(
        f"ytsearch{MAX_RESULTS}:{query}", ydl_opts, download=False
    )
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
    """Находит файлы по множеству расширений.

    Args:
        root: Корневая директория.
        exts: Набор расширений (в нижнем регистре, с точкой).

    Returns:
        Список путей к файлам.
    """
    out: List[str] = []
    for base, _, files in os.walk(root):
        for name in files:
            if os.path.splitext(name)[1].lower() in exts:
                out.append(os.path.join(base, name))
    return out


def find_audio_files(root: str) -> List[str]:
    """Находит аудиофайлы в директории.

    Args:
        root: Корень поиска.

    Returns:
        Список аудиофайлов.
    """
    return find_files_by_exts(root, AUDIO_EXTS)


def find_video_files(root: str) -> List[str]:
    """Находит видеофайлы в директории.

    Args:
        root: Корень поиска.

    Returns:
        Список видеофайлов.
    """
    return find_files_by_exts(root, VIDEO_EXTS)


def find_image_files(root: str) -> List[str]:
    """Находит изображения в директории.

    Args:
        root: Корень поиска.

    Returns:
        Список изображений.
    """
    return find_files_by_exts(root, IMAGE_EXTS)


def process_thumbnail(src_path: str, out_dir: str) -> Optional[str]:
    """Преобразует изображение для Telegram: 320x320 JPEG, ≤200KB.

    Args:
        src_path: Путь к исходной картинке.
        out_dir: Директория для вывода.

    Returns:
        Путь к подготовленному файлу или None при неудаче.
    """
    try:
        with Image.open(src_path) as im:
            im = im.convert("RGB")
            im = ImageOps.fit(
                im, THUMB_SIZE, method=Resampling.LANCZOS
            )  # заменено: Image.LANCZOS -> Resampling.LANCZOS
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
        path: Путь к файлу.

    Returns:
        Базовое имя без расширения и хвоста.
    """
    name = os.path.basename(path)
    name = name.split("#", 1)[0]
    base, _ = os.path.splitext(name)
    return base


def extract_id_from_base(base: str) -> Optional[str]:
    """Извлекает ID в квадратных скобках из базового имени.

    Args:
        base: Базовое имя файла.

    Returns:
        ID или None.
    """
    m = re.search(r"\[([0-9A-Za-z_-]{6,})\]", base)
    return m.group(1) if m else None


def make_duration_match_filter(
    max_seconds: int,
) -> Callable[[Dict[str, Any]], Optional[str]]:
    """Создаёт фильтр yt-dlp, отвергающий записи длиннее max_seconds.

    Args:
        max_seconds: Максимально допустимая длительность в секундах.

    Returns:
        Функция-фильтр, возвращающая строку-причину или None.
    """

    def _mf(info: Dict[str, Any]) -> Optional[str]:
        dur = info.get("duration")
        if isinstance(dur, (int, float)) and dur > max_seconds:
            return f"duration>{max_seconds}"
        return None

    return _mf


async def download_media_to_temp(
    url: str,
    mode: str,
    cookies_path: Optional[str] = None,
) -> List[Tuple[str, Optional[str]]]:
    """Скачивает медиа во временную директорию и подготавливает обложки.

    Args:
        url: Ссылка на ресурс.
        mode: Режим ('audio'|'video'|'video_nosound').
        cookies_path: Путь к cookies.txt, если требуется.

    Returns:
        Список кортежей (media_path, optional_thumbnail_path).
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
    """Отправляет медиафайлы по одному с опциональными обложками.

    Args:
        bot: Экземпляр бота.
        chat_id: Идентификатор чата.
        items: Список пар (путь к файлу, путь к обложке или None).
        method: Имя метода Telegram API ('send_audio'|'send_video').
        media_arg: Имя аргумента медиа ('audio'|'video').
        extra: Дополнительные аргументы к вызову отправки.
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
    """Отправляет список аудиофайлов.

    Args:
        bot: Экземпляр бота.
        chat_id: Идентификатор чата.
        items: Список медиа для отправки.
    """
    await send_media_files(bot, chat_id, items, method="send_audio", media_arg="audio")


async def send_video_files(
    bot: Bot, chat_id: int, items: List[Tuple[str, Optional[str]]]
) -> None:
    """Отправляет список видеофайлов.

    Args:
        bot: Экземпляр бота.
        chat_id: Идентификатор чата.
        items: Список медиа для отправки.
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
    """Выбирает способ отправки в зависимости от режима.

    Args:
        bot: Экземпляр бота.
        chat_id: Идентификатор чата.
        mode: Режим ('audio'|'video'|'video_nosound').
        items: Список медиа.
    """
    if mode == "audio":
        await send_audio_files(bot, chat_id, items)
    else:
        await send_video_files(bot, chat_id, items)


def remember_cookie_request(user_id: int, kind: str, url: str) -> None:
    """Сохраняет состояние ожидания cookies для пользователя.

    Args:
        user_id: Идентификатор пользователя.
        kind: Тип запроса ('download'|'pick').
        url: URL, который нужно повторить.
    """
    AWAITING_COOKIES[user_id] = {"kind": kind, "url": url, "asked": True}


def get_user_cookies_path(user_id: int) -> str:
    """Возвращает путь к файлу cookies пользователя.

    Args:
        user_id: Идентификатор пользователя.

    Returns:
        Путь к файлу cookies.txt.
    """
    return os.path.join(COOKIES_DIR, f"{user_id}_cookies.txt")


@router.message(CommandStart())
async def cmd_start(msg: Message) -> None:
    """Стартовая команда: сбрасывает состояние и показывает инструкцию.

    Args:
        msg: Входящее сообщение команды /start.
    """
    uid = msg.from_user.id if msg.from_user is not None else None
    if uid is not None:
        USER_SEARCHES.pop(uid, None)
        AWAITING_COOKIES.pop(uid, None)
    await msg.answer(
        "✨ Отправьте ссылку — скачаю по вашим настройкам (лучшее качество). Плейлисты до 10.\n"
        "📝 Или отправьте название — покажу список из 25 результатов.\n"
        "⚙️ Команда: /settings — выбрать тип скачивания.\n"
        "🍪 Если нужен доступ — пришлите файл cookies.txt.",
        reply_markup=build_main_reply_kb(),
    )


@router.message(Command("help"))
async def cmd_help(msg: Message) -> None:
    """Показывает краткую справку по использованию.

    Args:
        msg: Входящее сообщение команды /help.
    """
    await msg.answer(
        "ℹ️ Как пользоваться:\n"
        "• 🔗 Ссылка → скачивание по выбранному режиму (авто/аудио/видео/без звука).\n"
        "• 🔎 Текст запроса → 25 результатов, 5 страниц по 5 кнопок.\n"
        "• ⚙️ /settings — сменить тип скачивания.\n"
        "• 🍪 Если просит cookies — отправьте cookies.txt.",
        reply_markup=build_main_reply_kb(),
    )


@router.message(Command("settings"))
async def cmd_settings(msg: Message) -> None:
    """Открывает меню настроек (inline)."""
    if msg.from_user is None:
        await msg.answer(
            "⚙️ Настройки недоступны для этого типа сообщения.",
            reply_markup=build_main_reply_kb(),
        )
        return
    await msg.answer(
        "⚙️ Настройки типа скачивания:",
        reply_markup=build_settings_kb(msg.from_user.id).as_markup(),
    )


@router.callback_query(F.data == "settings:open")
async def cb_settings_open(cb: CallbackQuery) -> None:
    """Открывает настройки из инлайн-кнопки."""
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
    """Закрывает сообщение с меню настроек."""
    await try_cb_answer(cb)
    if cb.message is not None and isinstance(cb.message, Message):
        with suppress(Exception):
            await cb.message.delete()
        with suppress(Exception):
            await cb.message.edit_reply_markup(reply_markup=None)


@router.callback_query(F.data.startswith("setmode:"))
async def cb_set_mode(cb: CallbackQuery) -> None:
    """Устанавливает режим скачивания из инлайн-меню.

    Args:
        cb: CallbackQuery с выбранным режимом.
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
    kb = build_settings_kb(cb.from_user.id)
    if cb.message is not None and isinstance(cb.message, Message):
        with suppress(Exception):
            await cb.message.edit_reply_markup(reply_markup=kb.as_markup())
    await cb.answer("✅ Режим обновлён.")


@router.message(F.text)
async def handle_text(msg: Message, bot: Bot) -> None:
    """Обрабатывает текстовые сообщения: URL или поисковый запрос.

    При URL — сразу скачивает; при тексте — выполняет поиск.

    Args:
        msg: Входящее текстовое сообщение.
        bot: Экземпляр бота для отправки ответов.
    """
    raw = (msg.text or "").strip()
    text = raw
    uid = msg.from_user.id if msg.from_user is not None else None
    logger.info("Запрос от %s: %s", str(uid), text[:200] if text else "")
    if not text:
        await msg.answer("⚠️ Пустой запрос.")
        return
    if is_url(text):
        if uid is None:
            await msg.answer("⚠️ Не удалось определить пользователя.")
            return
        mode = decide_effective_mode(get_user_mode(uid), text)
        lock = await begin_user_download(uid)
        if not lock:
            await msg.answer("⏳ Идёт другая загрузка. Дождитесь завершения.")
            return
        await msg.answer("⏳ Скачиваю, подождите...")
        try:
            files = await download_media_to_temp(text, mode=mode)
            if not files:
                await msg.answer(
                    "😕 Нечего отправлять. Возможно, превышен лимит длительности (30 минут)."
                )
                return
            await send_by_mode(bot, msg.chat.id, mode, files)
        except DownloadError as e:
            logger.warning("Требуются cookies или ошибка загрузки: %s", e)
            remember_cookie_request(uid, kind="download", url=text)
            await msg.answer(
                "🍪 Источник требует cookies или произошла ошибка.\nПришлите файл cookies.txt для повтора попытки."
            )
        except Exception:
            logger.exception("Ошибка при загрузке по URL")
            await msg.answer("❌ Произошла ошибка при загрузке. Попробуйте позже.")
        finally:
            end_user_download(lock)
        return
    query = sanitize_query(text)
    if not query:
        await msg.answer("⚠️ Некорректный запрос.")
        return
    await msg.answer("🔎 Ищу треки...")
    try:
        results = await search_tracks(query)
        if uid is not None:
            USER_SEARCHES[uid] = {"results": results, "page": 0}
        if not results:
            await msg.answer("🙁 Ничего не найдено (или превышен лимит длительности).")
            return
        kb = build_results_kb(uid if uid is not None else 0)
        await msg.answer("📋 Результаты поиска:", reply_markup=kb.as_markup())
    except Exception:
        await msg.answer("❌ Ошибка поиска. Попробуйте позже.")


@router.callback_query(F.data == "noop")
async def handle_noop(cb: CallbackQuery) -> None:
    """Обрабатывает пустой callback.

    Args:
        cb: CallbackQuery без действия.
    """
    await try_cb_answer(cb)


@router.callback_query(F.data == "cancel")
async def handle_cancel(cb: CallbackQuery) -> None:
    """Отменяет текущий список результатов и ожидание cookies.

    Args:
        cb: CallbackQuery с действием отмены.
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
    """Листает список результатов вперёд.

    Args:
        cb: CallbackQuery листания вперёд.
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
    """Листает список результатов назад.

    Args:
        cb: CallbackQuery листания назад.
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
    """Начинает загрузку выбранного результата из списка поиска.

    Args:
        cb: CallbackQuery с выбранным индексом результата.
        bot: Экземпляр бота для отправки сообщений и медиа.
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

        mode = decide_effective_mode(get_user_mode(cb.from_user.id), url)
        lock = await begin_user_download(cb.from_user.id)
        if not lock:
            await try_cb_answer(cb, "⏳ Загрузка уже выполняется.")
            return
        await try_cb_answer(cb)
        chat_id = get_cb_chat_id(cb)
        if chat_id is None:
            end_user_download(lock)
            return
        await bot.send_message(chat_id, "⏳ Скачиваю выбранный элемент...")
        try:
            files = await download_media_to_temp(url, mode=mode)
            if not files:
                await bot.send_message(
                    chat_id,
                    "😕 Нечего отправлять. Возможно, превышен лимит длительности (30 минут).",
                )
                return
            await send_by_mode(bot, chat_id, mode, files)
        except DownloadError:
            remember_cookie_request(cb.from_user.id, kind="pick", url=url)
            await bot.send_message(
                chat_id,
                "🍪 Источник требует cookies или произошла ошибка.\nПришлите файл cookies.txt для повтора попытки.",
            )
        except Exception:
            await bot.send_message(
                chat_id, "❌ Ошибка при загрузке выбранного элемента."
            )
        finally:
            end_user_download(lock)


@router.message(F.document)
async def handle_document(msg: Message, bot: Bot) -> None:
    """Принимает файл cookies.txt и повторяет прошлую попытку загрузки.

    Args:
        msg: Сообщение с документом cookies.txt.
        bot: Экземпляр бота, используемый для скачивания и ответов.
    """
    if msg.from_user is None:
        await msg.answer("📄 Файл получен, но не удалось определить пользователя.")
        return
    pending = AWAITING_COOKIES.get(msg.from_user.id)
    if not pending:
        await msg.answer("📄 Файл получен, но сейчас cookies не требуются.")
        return

    cookies_path = get_user_cookies_path(msg.from_user.id)
    doc = msg.document
    if doc is None:
        await msg.answer("❌ Не удалось прочитать файл.")
        return

    name_l = (doc.file_name or "").lower()
    ext = os.path.splitext(name_l)[1]
    size = doc.file_size or 0
    if ext not in ALLOWED_COOKIES_EXTS:
        await msg.answer("⚠️ Нужен файл cookies в формате Netscape: cookies.txt.")
        return
    if size and size > COOKIES_MAX_BYTES:
        lim_mb = COOKIES_MAX_BYTES / (1024 * 1024)
        cur_mb = size / (1024 * 1024)
        await msg.answer(
            f"⚠️ Слишком большой cookies.txt ({cur_mb:.1f} МБ). Максимум {lim_mb:.0f} МБ."
        )
        return

    try:
        await bot.download(doc, destination=cookies_path)
    except Exception:
        await msg.answer("❌ Не удалось сохранить cookies.txt.")
        return

    with suppress(Exception):
        real_size = os.path.getsize(cookies_path)
        if real_size > COOKIES_MAX_BYTES:
            lim_mb = COOKIES_MAX_BYTES / (1024 * 1024)
            cur_mb = real_size / (1024 * 1024)
            with suppress(Exception):
                os.remove(cookies_path)
            await msg.answer(
                f"⚠️ Слишком большой cookies.txt ({cur_mb:.1f} МБ). Максимум {lim_mb:.0f} МБ."
            )
            return

    await msg.answer("🍪 Cookies получены. Пробую снова...")

    url_any = pending.get("url")
    if not isinstance(url_any, str) or not url_any:
        await msg.answer("❌ Нет URL для повтора.")
        return
    url = url_any
    AWAITING_COOKIES.pop(msg.from_user.id, None)
    lock = await begin_user_download(msg.from_user.id)
    if not lock:
        await msg.answer("⏳ Идёт другая загрузка. Дождитесь завершения.")
        return
    try:
        mode = decide_effective_mode(get_user_mode(msg.from_user.id), url)
        files = await download_media_to_temp(url, mode=mode, cookies_path=cookies_path)
        if not files:
            await msg.answer(
                "😕 Не удалось скачать даже с cookies (возможно, превышен лимит длительности)."
            )
            return
        await send_by_mode(bot, msg.chat.id, mode, files)
    except Exception:
        await msg.answer("❌ Не удалось скачать даже с cookies. Скипаю.")
    finally:
        end_user_download(lock)


async def main() -> None:
    """Точка входа: настройка логирования и старт long-polling.

    Returns:
        None.
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
