import asyncio
import os
import re
from dotenv import load_dotenv
import math
import tempfile
import shutil
from contextlib import suppress
from urllib.parse import urlparse
from typing import Any, Dict, List, Optional, Tuple
import logging
from logging.handlers import TimedRotatingFileHandler
import io
from PIL import Image, ImageOps

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

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

# ========= Настройки =========
load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
MAX_RESULTS = 25
PAGE_SIZE = 5
CONCURRENT_DOWNLOADS = 2
AUDIO_EXTS = {".mp3", ".m4a", ".opus", ".webm", ".ogg", ".flac", ".wav"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_PLAYLIST_ITEMS = 10  # ограничение длины плейлиста
# Добавлены видеорасширения
VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".m4v"}

# Требования к обложке для Telegram
THUMB_SIZE = (320, 320)
THUMB_MAX_BYTES = 200 * 1024

# ========= Глобальные объекты =========
router = Router()
dp = Dispatcher()
dp.include_router(router)
download_sem = asyncio.Semaphore(CONCURRENT_DOWNLOADS)

# user_id -> {"results": List[Dict], "page": int}
USER_SEARCHES: Dict[int, Dict[str, Any]] = {}

# user_id -> pending action payload:
#   {"kind": "download", "url": str, "asked": bool}
#   {"kind": "pick", "url": str, "asked": bool}
AWAITING_COOKIES: Dict[int, Dict[str, Any]] = {}

# user_id -> saved cookies file path
COOKIES_DIR = os.path.join(os.getcwd(), "cookies")
os.makedirs(COOKIES_DIR, exist_ok=True)

# user_id -> {"mode": "auto"|"audio"|"video"|"video_nosound"}
USER_SETTINGS: Dict[int, Dict[str, str]] = {}

# ========= Логирование =========
def setup_logging(log_dir: str = "logs") -> None:
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

    # Корневой логгер
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    # Консоль INFO+
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    root.addHandler(console)

    # Отдельные файлы по уровням
    root.addHandler(make_rotating("app.debug.log", logging.DEBUG))
    root.addHandler(make_rotating("app.info.log", logging.INFO))
    root.addHandler(make_rotating("app.warn.log", logging.WARNING))
    root.addHandler(make_rotating("app.error.log", logging.ERROR))

    # Сторонние логгеры
    logging.getLogger("aiogram").setLevel(logging.INFO)
    logging.getLogger("aiohttp").setLevel(logging.INFO)
    logging.getLogger("yt_dlp").setLevel(logging.INFO)


# Инициализация логгера модуля
logger = logging.getLogger("bot")


# ========= Основная логика =========
def is_url(text: str) -> bool:
    with suppress(Exception):
        u = urlparse(text.strip())
        return u.scheme in {"http", "https"} and bool(u.netloc)
    return False


def slice_page(items: List[Any], page: int, page_size: int) -> Tuple[List[Any], int]:
    pages = max(1, math.ceil(len(items) / page_size))
    page = max(0, min(page, pages - 1))
    start = page * page_size
    end = start + page_size
    return items[start:end], pages


def get_user_mode(user_id: int) -> str:
    st = USER_SETTINGS.get(user_id)
    return (st or {}).get("mode", "auto")


def set_user_mode(user_id: int, mode: str) -> None:
    USER_SETTINGS[user_id] = {"mode": mode}


def is_audio_platform(url: str) -> bool:
    """
    Эвристика: music.youtube и популярные аудиоплощадки — аудио; иначе — видео.
    """
    try:
        u = urlparse(url)
        host = (u.netloc or "").lower()
        path = (u.path or "").lower()
    except Exception:
        return False
    audio_hosts = [
        "music.youtube.", "soundcloud.com", "bandcamp.com", "mixcloud.com",
        "audius.co", "hearthis.at", "promodj.com", "music.yandex.", "yandex.ru/music",
        "deezer.com", "napster.com"
    ]
    return any(h in host for h in audio_hosts) or "/music" in path


def decide_effective_mode(user_mode: str, url: str) -> str:
    if user_mode == "auto":
        return "audio" if is_audio_platform(url) else "video"
    return user_mode


def build_results_kb(user_id: int) -> InlineKeyboardBuilder:
    state = USER_SEARCHES.get(user_id) or {}
    results: List[Dict[str, Any]] = state.get("results", [])
    page: int = state.get("page", 0)

    current, pages = slice_page(results, page, PAGE_SIZE)
    kb = InlineKeyboardBuilder()

    for idx, entry in enumerate(current):
        global_index = page * PAGE_SIZE + idx
        title = entry.get("title") or "Без названия"
        # Укорачиваем подпись кнопки
        if len(title) > 64:
            title = title[:61] + "..."
        kb.button(text=title, callback_data=f"pick:{global_index}")

    if not results:
        kb.button(text="Нет результатов", callback_data="noop")
    kb.adjust(1)

    # Навигация
    if results:
        kb.row(
            InlineKeyboardButton(text="« Назад", callback_data="page:prev"),
            InlineKeyboardButton(text=f"{page + 1}/{pages}", callback_data="noop"),
            InlineKeyboardButton(text="Вперёд »", callback_data="page:next"),
        )
    # Добавляем кнопку настроек (откроет меню настроек reply-клавиатуры)
    kb.row(InlineKeyboardButton(text="Настройки ⚙️", callback_data="settings:open"))
    kb.row(InlineKeyboardButton(text="Отмена", callback_data="cancel"))
    return kb


def build_settings_kb(user_id: int) -> InlineKeyboardBuilder:
    mode = get_user_mode(user_id)
    text = {
        "auto": "Автоопределение",
        "audio": "Только аудио",
        "video": "Только видео (со звуком)",
        "video_nosound": "Только видео (без звука)",
    }
    kb = InlineKeyboardBuilder()
    for m in ["auto", "audio", "video", "video_nosound"]:
        pref = "✅ " if mode == m else "• "
        kb.button(text=pref + text[m], callback_data=f"setmode:{m}")
    kb.adjust(1)
    kb.row(InlineKeyboardButton(text="Закрыть", callback_data="settings:close"))
    return kb

# ==== Новая постоянная стартовая клавиатура и меню настроек (ReplyKeyboard) ====
MAIN_BUTTONS = ["/start", "/help", "/settings"]
SETTINGS_TEXT_TO_MODE: Dict[str, str] = {
    "Автоопределение": "auto",
    "Только аудио": "audio",
    "Только видео (со звуком)": "video",
    "Только видео (без звука)": "video_nosound",
}
BACK_BUTTON_TEXT = "⬅ Назад"
# Добавлено: поддержка распознавания выбора с/без префикса "✅ "
SETTINGS_TITLES = list(SETTINGS_TEXT_TO_MODE.keys())
SETTINGS_REPLY_RE = re.compile(r'^(?:✅\s*)?(%s)$' % '|'.join(map(re.escape, SETTINGS_TITLES)))

def build_main_reply_kb() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="/start"), KeyboardButton(text="/help")],
            [KeyboardButton(text="/settings")],
        ],
        resize_keyboard=True,
        is_persistent=True,
    )

# Изменено: делаем клавиатуру настроек динамической и помечаем текущий режим "✅"
def build_settings_reply_kb(user_id: int) -> ReplyKeyboardMarkup:
    mode = get_user_mode(user_id)
    rows = []
    for title, m in SETTINGS_TEXT_TO_MODE.items():
        prefix = "✅ " if m == mode else ""
        rows.append([KeyboardButton(text=f"{prefix}{title}")])
    rows.append([KeyboardButton(text=BACK_BUTTON_TEXT)])
    return ReplyKeyboardMarkup(
        keyboard=rows,
        resize_keyboard=True,
        is_persistent=True,
    )

async def ytdlp_extract(url_or_query: str, ydl_opts: Dict[str, Any], download: bool) -> Dict[str, Any]:
    def _run() -> Dict[str, Any]:
        with YoutubeDL(ydl_opts) as ydl:
            return ydl.extract_info(url_or_query, download=download)

    return await asyncio.to_thread(_run)


async def search_tracks(query: str) -> List[Dict[str, Any]]:
    # Ищем первые 25 результатов
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "noplaylist": True,
        "default_search": "ytsearch",
    }
    info = await ytdlp_extract(f"ytsearch{MAX_RESULTS}:{query}", ydl_opts, download=False)
    entries = info.get("entries") or []
    results: List[Dict[str, Any]] = []
    for e in entries:
        # Нормализуем URL
        url = e.get("webpage_url") or e.get("url")
        if not url and e.get("id"):
            url = f"https://www.youtube.com/watch?v={e['id']}"
        title = e.get("title") or "Без названия"
        duration = e.get("duration")  # в секундах, может быть None
        channel = e.get("uploader") or e.get("channel") or ""
        results.append(
            {
                "title": title,
                "url": url,
                "duration": duration,
                "channel": channel,
            }
        )
    return results


def find_audio_files(root: str) -> List[str]:
    out: List[str] = []
    for base, _, files in os.walk(root):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext in AUDIO_EXTS:
                out.append(os.path.join(base, name))
    return out


def find_video_files(root: str) -> List[str]:
    out: List[str] = []
    for base, _, files in os.walk(root):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext in VIDEO_EXTS:
                out.append(os.path.join(base, name))
    return out


def find_image_files(root: str) -> List[str]:
    out: List[str] = []
    for base, _, files in os.walk(root):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext in IMAGE_EXTS:
                out.append(os.path.join(base, name))
    return out


def process_thumbnail(src_path: str, out_dir: str) -> Optional[str]:
    """
    Приводит картинку к требованиям Telegram: 320x320, JPEG, <= 200KB.
    Возвращает путь к сжатому файлу или None при неудаче.
    """
    try:
        with Image.open(src_path) as im:
            im = im.convert("RGB")
            im = ImageOps.fit(im, THUMB_SIZE, method=Image.LANCZOS)
            quality = 90
            min_q = 40
            step = 5
            out_path = os.path.join(
                out_dir,
                f"{os.path.splitext(os.path.basename(src_path))[0]}_320.jpg",
            )
            last_size = None
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
        logging.getLogger("bot").warning("Не удалось обработать обложку %s: %s", src_path, e)
        return None


def norm_base(path: str) -> str:
    # Сначала убираем хвост после '#', затем отрезаем расширение
    name = os.path.basename(path)
    name = name.split('#', 1)[0]
    base, _ = os.path.splitext(name)
    return base


def extract_id_from_base(base: str) -> Optional[str]:
    m = re.search(r'\[([0-9A-Za-z_-]{6,})\]', base)
    return m.group(1) if m else None


async def download_media_to_temp(
        url: str,
        mode: str,
        cookies_path: Optional[str] = None,
) -> List[Tuple[str, Optional[str]]]:
    """
    Возвращает список (path, thumb_path). Для mode:
      - "audio": отдаёт аудиофайлы
      - "video"/"video_nosound": отдаёт видеофайлы
    """
    tmpdir = tempfile.mkdtemp(prefix="dl_")
    # Постпроцессоры под режим
    if mode == "audio":
        postprocessors = [
            {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"},
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
        # Лучшее качество видео+аудио, при возможности объединяем в mp4
        ydl_format = "bv*+ba/b"
        extra = {
            "merge_output_format": "mp4",
            "recode_video": "mp4",
        }
    else:  # "video_nosound"
        postprocessors = [
            {"key": "FFmpegThumbnailsConvertor", "format": "jpg"},
            {"key": "FFmpegMetadata"},
        ]
        ydl_format = "bestvideo/best"
        extra = {
            "recode_video": "mp4",
        }

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
    logger.info("Файлов найдено (media=%d, images=%d)", len(media_files), len(image_files))
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
                    name_wo_hash = os.path.basename(img).split('#', 1)[0]
                    if needle in name_wo_hash:
                        possible_imgs.append(img)

        t_src = None
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


async def send_audio_files(bot: Bot, chat_id: int, items: List[Tuple[str, Optional[str]]]) -> None:
    for audio_path, thumb_path in items:
        try:
            logger.info("Отправка файла: %s", audio_path)
            title = os.path.splitext(os.path.basename(audio_path))[0]
            if thumb_path and os.path.exists(thumb_path):
                try:
                    size = os.path.getsize(thumb_path)
                    logger.info("Обложка для трека отправляется: %s (%d байт)", thumb_path, size)
                except Exception:
                    logger.info("Обложка для трека отправляется: %s", thumb_path)
            else:
                logger.info("Отправка без обложки для: %s", audio_path)

            thumb_input = FSInputFile(thumb_path) if thumb_path and os.path.exists(thumb_path) else None
            await bot.send_audio(
                chat_id=chat_id,
                audio=FSInputFile(audio_path),
                caption=title,
                thumbnail=thumb_input,
            )
        finally:
            # Удаляем файл после отправки
            with suppress(Exception):
                os.remove(audio_path)
            if thumb_path:
                with suppress(Exception):
                    os.remove(thumb_path)
            await asyncio.sleep(0.3)


async def send_video_files(bot: Bot, chat_id: int, items: List[Tuple[str, Optional[str]]]) -> None:
    for video_path, thumb_path in items:
        try:
            logger.info("Отправка видео: %s", video_path)
            title = os.path.splitext(os.path.basename(video_path))[0]
            thumb_input = FSInputFile(thumb_path) if thumb_path and os.path.exists(thumb_path) else None
            await bot.send_video(
                chat_id=chat_id,
                video=FSInputFile(video_path),
                caption=title,
                thumbnail=thumb_input,
                supports_streaming=True,
            )
        finally:
            with suppress(Exception):
                os.remove(video_path)
            if thumb_path:
                with suppress(Exception):
                    os.remove(thumb_path)
            await asyncio.sleep(0.3)


def remember_cookie_request(user_id: int, kind: str, url: str) -> None:
    AWAITING_COOKIES[user_id] = {"kind": kind, "url": url, "asked": True}


def get_user_cookies_path(user_id: int) -> str:
    return os.path.join(COOKIES_DIR, f"{user_id}_cookies.txt")


@router.message(CommandStart())
async def cmd_start(msg: Message) -> None:
    # Сброс локального состояния пользователя
    USER_SEARCHES.pop(msg.from_user.id, None)
    AWAITING_COOKIES.pop(msg.from_user.id, None)
    await msg.answer(
        "Отправьте ссылку — скачаю по вашим настройкам (лучшее качество). Плейлисты до 10.\n"
        "Или отправьте название — покажу список из 25 результатов.\n"
        "Команда: /settings — выбрать тип скачивания.\n"
        "Если нужен доступ — пришлите файл cookies.txt.",
        reply_markup=build_main_reply_kb(),
    )


@router.message(Command("help"))
async def cmd_help(msg: Message) -> None:
    await msg.answer(
        "Как пользоваться:\n"
        "• Ссылка → скачивание по выбранному режиму (авто/аудио/видео/видео без звука).\n"
        "• Текст запроса → 25 результатов, 5 страниц по 5 кнопок.\n"
        "• /settings — сменить тип скачивания.\n"
        "• Если просит cookies — отправьте cookies.txt.",
        reply_markup=build_main_reply_kb(),
    )


@router.message(Command("settings"))
async def cmd_settings(msg: Message) -> None:
    await msg.answer("Настройки типа скачивания:", reply_markup=build_settings_reply_kb(msg.from_user.id))


@router.callback_query(F.data == "settings:open")
async def cb_settings_open(cb: CallbackQuery) -> None:
    # Открываем меню настроек (reply-клавиатура)
    with suppress(Exception):
        await cb.answer()
    await cb.message.answer("Настройки типа скачивания:", reply_markup=build_settings_reply_kb(cb.from_user.id))


# Изменено: обработчик выбора режима через regex (принимает текст с/без "✅")
@router.message(F.text.regexp(SETTINGS_REPLY_RE))
async def handle_settings_choice(msg: Message) -> None:
    raw = (msg.text or "").strip()
    if raw.startswith("✅"):
        raw = raw[1:].strip()
    mode = SETTINGS_TEXT_TO_MODE.get(raw)
    if not mode:
        return
    set_user_mode(msg.from_user.id, mode)
    await msg.answer(f"Режим обновлён: {raw}", reply_markup=build_settings_reply_kb(msg.from_user.id))


# Кнопка "Назад" — вернуться к стартовой клавиатуре
@router.message(F.text == BACK_BUTTON_TEXT)
async def handle_settings_back(msg: Message) -> None:
    await msg.answer("Возврат к начальной клавиатуре.", reply_markup=build_main_reply_kb())


@router.callback_query(F.data.startswith("setmode:"))
async def cb_set_mode(cb: CallbackQuery) -> None:
    mode = cb.data.split(":", 1)[1]
    if mode not in {"auto", "audio", "video", "video_nosound"}:
        await cb.answer("Неизвестный режим.")
        return
    set_user_mode(cb.from_user.id, mode)
    kb = build_settings_kb(cb.from_user.id)
    with suppress(Exception):
        await cb.message.edit_reply_markup(reply_markup=kb.as_markup())
    await cb.answer("Режим обновлён.")


@router.message(F.text)
async def handle_text(msg: Message, bot: Bot) -> None:
    text = (msg.text or "").strip()
    logger.info("Запрос от %s: %s", msg.from_user.id, text[:200])

    if not text:
        await msg.answer("Пустой запрос.")
        return

    if is_url(text):
        mode = decide_effective_mode(get_user_mode(msg.from_user.id), text)
        await msg.answer("Скачиваю, подождите...")
        try:
            files = await download_media_to_temp(text, mode=mode)
            if not files:
                await msg.answer("Нечего отправлять.")
                return
            if mode == "audio":
                await send_audio_files(bot, msg.chat.id, files)
            else:
                await send_video_files(bot, msg.chat.id, files)
        except DownloadError as e:
            logger.warning("Требуются cookies или ошибка загрузки: %s", e)
            remember_cookie_request(msg.from_user.id, kind="download", url=text)
            await msg.answer(
                "Источник требует cookies или произошла ошибка.\n"
                "Пришлите файл cookies.txt для повтора попытки."
            )
        except Exception:
            logger.exception("Ошибка при загрузке по URL")
            await msg.answer("Произошла ошибка при загрузке. Попробуйте позже.")
        return

    # Иначе — поиск
    await msg.answer("Ищу треки...")
    try:
        results = await search_tracks(text)
        USER_SEARCHES[msg.from_user.id] = {"results": results, "page": 0}
        if not results:
            await msg.answer("Ничего не найдено.")
            return
        kb = build_results_kb(msg.from_user.id)
        await msg.answer("Результаты поиска:", reply_markup=kb.as_markup())
    except Exception:
        await msg.answer("Ошибка поиска. Попробуйте позже.")


@router.callback_query(F.data == "noop")
async def handle_noop(cb: CallbackQuery) -> None:
    with suppress(Exception):
        await cb.answer()


@router.callback_query(F.data == "cancel")
async def handle_cancel(cb: CallbackQuery) -> None:
    USER_SEARCHES.pop(cb.from_user.id, None)
    AWAITING_COOKIES.pop(cb.from_user.id, None)
    with suppress(Exception):
        await cb.message.edit_reply_markup(reply_markup=None)
    await cb.answer("Отменено.")


@router.callback_query(F.data == "page:next")
async def handle_next_page(cb: CallbackQuery) -> None:
    state = USER_SEARCHES.get(cb.from_user.id)
    if not state:
        await cb.answer("Нет активного списка.")
        return
    results = state["results"]
    page = state.get("page", 0)
    _, pages = slice_page(results, page, PAGE_SIZE)
    page = (page + 1) % pages
    state["page"] = page
    kb = build_results_kb(cb.from_user.id)
    with suppress(Exception):
        await cb.message.edit_reply_markup(reply_markup=kb.as_markup())
    await cb.answer()


@router.callback_query(F.data == "page:prev")
async def handle_prev_page(cb: CallbackQuery) -> None:
    state = USER_SEARCHES.get(cb.from_user.id)
    if not state:
        await cb.answer("Нет активного списка.")
        return
    results = state["results"]
    page = state.get("page", 0)
    _, pages = slice_page(results, page, PAGE_SIZE)
    page = (page - 1 + pages) % pages
    state["page"] = page
    kb = build_results_kb(cb.from_user.id)
    with suppress(Exception):
        await cb.message.edit_reply_markup(reply_markup=kb.as_markup())
    await cb.answer()


@router.callback_query(F.data.startswith("pick:"))
async def handle_pick(cb: CallbackQuery, bot: Bot) -> None:
    idx_str = cb.data.split(":", 1)[1]
    with suppress(ValueError):
        idx = int(idx_str)
        state = USER_SEARCHES.get(cb.from_user.id)
        if not state:
            await cb.answer("Список результатов устарел.")
            return
        results: List[Dict[str, Any]] = state["results"]
        if idx < 0 or idx >= len(results):
            await cb.answer("Некорректный выбор.")
            return
        url = results[idx].get("url")
        if not url:
            await cb.answer("Нет URL для выбранного трека.")
            return

        mode = decide_effective_mode(get_user_mode(cb.from_user.id), url)
        # Вместо всплывающего уведомления — сообщение в чат
        with suppress(Exception):
            await cb.answer()
        await bot.send_message(cb.message.chat.id, "Скачиваю выбранный элемент...")
        try:
            files = await download_media_to_temp(url, mode=mode)
            if not files:
                await bot.send_message(cb.message.chat.id, "Нечего отправлять.")
                return
            if mode == "audio":
                await send_audio_files(bot, cb.message.chat.id, files)
            else:
                await send_video_files(bot, cb.message.chat.id, files)
        except DownloadError:
            remember_cookie_request(cb.from_user.id, kind="pick", url=url)
            await bot.send_message(
                cb.message.chat.id,
                "Источник требует cookies или произошла ошибка.\n"
                "Пришлите файл cookies.txt для повтора попытки."
            )
        except Exception:
            await bot.send_message(cb.message.chat.id, "Ошибка при загрузке выбранного элемента.")


@router.message(F.document)
async def handle_document(msg: Message, bot: Bot) -> None:
    pending = AWAITING_COOKIES.get(msg.from_user.id)
    if not pending:
        await msg.answer("Файл получен, но сейчас cookies не требуются.")
        return

    # Сохраняем cookies
    cookies_path = get_user_cookies_path(msg.from_user.id)
    try:
        await bot.download(msg.document, destination=cookies_path)
    except Exception:
        await msg.answer("Не удалось сохранить cookies.txt.")
        return

    await msg.answer("Cookies получены. Пробую снова...")

    url = pending.get("url")
    AWAITING_COOKIES.pop(msg.from_user.id, None)

    try:
        mode = decide_effective_mode(get_user_mode(msg.from_user.id), url)
        files = await download_media_to_temp(url, mode=mode, cookies_path=cookies_path)
        if not files:
            await msg.answer("Не удалось скачать даже с cookies. Скипаю.")
            return
        if mode == "audio":
            await send_audio_files(bot, msg.chat.id, files)
        else:
            await send_video_files(bot, msg.chat.id, files)
    except Exception:
        await msg.answer("Не удалось скачать даже с cookies. Скипаю.")


async def main() -> None:
    setup_logging()  # <= добавьте
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
