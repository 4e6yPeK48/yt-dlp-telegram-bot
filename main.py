"""
8221150427:AAGUBlaFCHQfKhSn3dpM9xsUDlhT6aOP0jA
"""


# main.py
import asyncio
import os
import math
import tempfile
import shutil
from contextlib import suppress
from urllib.parse import urlparse
from typing import Any, Dict, List, Optional, Tuple

from aiogram import Bot, Dispatcher, Router, F
from aiogram.filters import CommandStart, Command
from aiogram.types import (
    Message,
    CallbackQuery,
    InlineKeyboardButton,
    FSInputFile,
)
from aiogram.utils.keyboard import InlineKeyboardBuilder

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

# ========= Настройки =========
BOT_TOKEN = os.getenv("BOT_TOKEN")  # Установите переменную окружения BOT_TOKEN
MAX_RESULTS = 25
PAGE_SIZE = 5
CONCURRENT_DOWNLOADS = 2
AUDIO_EXTS = {".mp3", ".m4a", ".opus", ".webm", ".ogg", ".flac", ".wav"}

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
    kb.row(InlineKeyboardButton(text="Отмена", callback_data="cancel"))
    return kb


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


async def download_media_to_temp(
    url: str,
    convert_to_mp3: bool = True,
    cookies_path: Optional[str] = None,
) -> List[str]:
    """
    Скачивает трек или плейлист в temp\-директорию.
    Возвращает список путей к аудио файлам.
    """
    tmpdir = tempfile.mkdtemp(prefix="dl_")
    # Для плейлистов не запрещаем, yt\-dlp сам решит
    postprocessors = []
    if convert_to_mp3:
        postprocessors = [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }]

    ydl_opts: Dict[str, Any] = {
        "quiet": True,
        "format": "bestaudio/best",
        "outtmpl": os.path.join(tmpdir, "%(title)s [%(id)s].%(ext)s"),
        "noplaylist": False,
        "postprocessors": postprocessors,
        "nocheckcertificate": True,
    }
    if cookies_path and os.path.exists(cookies_path):
        ydl_opts["cookiefile"] = cookies_path

    # Выполняем в пуле потоков, под семафором
    async with download_sem:
        try:
            await ytdlp_extract(url, ydl_opts, download=True)
        except DownloadError as e:
            # Пробрасываем выше, чтобы обработчик попросил cookies
            raise e
        except Exception as e:
            raise e

    files = find_audio_files(tmpdir)
    if not files:
        # Если ничего не конвертировалось в mp3, попробуем взять исходники
        files = find_audio_files(tmpdir)
    if not files:
        # Чистим и сообщаем
        shutil.rmtree(tmpdir, ignore_errors=True)
        return []

    # Перенесём файлы в устойчивую temp, чтобы можно было удалить рабочую папку
    stable_dir = tempfile.mkdtemp(prefix="out_")
    moved: List[str] = []
    for f in files:
        dst = os.path.join(stable_dir, os.path.basename(f))
        with suppress(Exception):
            shutil.move(f, dst)
            moved.append(dst)

    shutil.rmtree(tmpdir, ignore_errors=True)
    return moved


async def send_audio_files(bot: Bot, chat_id: int, files: List[str]) -> None:
    for path in files:
        try:
            title = os.path.splitext(os.path.basename(path))[0]
            await bot.send_audio(
                chat_id=chat_id,
                audio=FSInputFile(path),
                caption=title,
            )
        finally:
            # Удаляем файл после отправки
            with suppress(Exception):
                os.remove(path)
            # Небольшая пауза, чтобы не упереться в лимиты
            await asyncio.sleep(0.3)


def remember_cookie_request(user_id: int, kind: str, url: str) -> None:
    AWAITING_COOKIES[user_id] = {"kind": kind, "url": url, "asked": True}


def get_user_cookies_path(user_id: int) -> str:
    return os.path.join(COOKIES_DIR, f"{user_id}_cookies.txt")


@router.message(CommandStart())
async def cmd_start(msg: Message) -> None:
    await msg.answer(
        "Отправьте ссылку на трек/плейлист — скачаю и пришлю аудио.\n"
        "Или отправьте название трека — покажу список из 25 результатов.\n"
        "Если трек требует cookies, пришлите файл `cookies.txt`."
    )


@router.message(Command("help"))
async def cmd_help(msg: Message) -> None:
    await msg.answer(
        "Как пользоваться:\n"
        "• Ссылка → скачивание аудио (mp3 по умолчанию).\n"
        "• Текст запроса → 25 результатов, 5 страниц по 5 кнопок.\n"
        "• Если просит cookies — отправьте `cookies.txt`."
    )


@router.message(F.text)
async def handle_text(msg: Message, bot: Bot) -> None:
    text = (msg.text or "").strip()

    if not text:
        await msg.answer("Пустой запрос.")
        return

    # Если это URL — качаем
    if is_url(text):
        await msg.answer("Скачиваю, подождите...")
        try:
            files = await download_media_to_temp(text, convert_to_mp3=True)
            if not files:
                await msg.answer("Не удалось получить аудио файлы.")
                return
            await send_audio_files(bot, msg.chat.id, files)
        except DownloadError:
            # Просим cookies и запоминаем задание
            remember_cookie_request(msg.from_user.id, kind="download", url=text)
            await msg.answer(
                "Источник требует cookies или произошла ошибка.\n"
                "Пришлите файл `cookies.txt` для повтора попытки."
            )
        except Exception:
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

        await cb.answer("Скачиваю выбранный трек...")
        try:
            files = await download_media_to_temp(url, convert_to_mp3=True)
            if not files:
                await bot.send_message(cb.message.chat.id, "Не удалось получить аудио файл.")
                return
            await send_audio_files(bot, cb.message.chat.id, files)
        except DownloadError:
            remember_cookie_request(cb.from_user.id, kind="pick", url=url)
            await bot.send_message(
                cb.message.chat.id,
                "Источник требует cookies или произошла ошибка.\n"
                "Пришлите файл `cookies.txt` для повтора попытки."
            )
        except Exception:
            await bot.send_message(cb.message.chat.id, "Ошибка при загрузке выбранного трека.")


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
    kind = pending.get("kind")
    # Один раз пробуем с cookies, затем забываем запрос
    AWAITING_COOKIES.pop(msg.from_user.id, None)

    try:
        files = await download_media_to_temp(url, convert_to_mp3=True, cookies_path=cookies_path)
        if not files:
            await msg.answer("Не удалось скачать даже с cookies. Скипаю.")
            return
        await send_audio_files(bot, msg.chat.id, files)
    except Exception:
        await msg.answer("Не удалось скачать даже с cookies. Скипаю.")


async def main() -> None:
    if not BOT_TOKEN:
        raise RuntimeError("Не задана переменная окружения BOT_TOKEN")
    bot = Bot(BOT_TOKEN, parse_mode="HTML")
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
