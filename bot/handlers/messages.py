import os
import asyncio
from asyncio import to_thread
from contextlib import suppress

from aiogram import Bot, F
from aiogram.types import (
    Message,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from yt_dlp import YoutubeDL  # type: ignore[import-untyped]
from yt_dlp.utils import DownloadError  # type: ignore[import-untyped]

from bot.dispatcher import router, logger
from utils.log_helpers import log_info, log_warning, log_exception
from bot.handlers.commands import cmd_start, cmd_help, cmd_settings, cmd_history
from bot.keyboards import build_download_choice_kb, build_results_kb
from config import (
    COOKIES_MAX_BYTES,
    ALLOWED_COOKIES_EXTS,
)
from services.telegram import (
    send_info_card,
)
from services.ytdlp import (
    decide_effective_mode,
    search_tracks,
)
from storage.state import (
    get_searches,
    set_searches,
    pop_searches,
    get_awaiting,
    pop_awaiting,
    get_user_mode,
    begin_user_download,
    remember_search_cookie_request,
    get_user_cookies_path,
    save_pending_url,
    set_download_task,
    pop_download_task,
)
from utils.text import sanitize_query, parse_main_button_intent
from utils.validators import is_url
from bot.handlers.downloads import perform_download
from bot.helpers import get_user_and_chat


@router.message(F.text)
async def handle_text(msg: Message, bot: Bot) -> None:
    """Обрабатывает текст: команды/кнопки, URL (меню скачивания) или поиск.

    Args:
        msg (Message): Входящее сообщение.
        bot (Bot): Экземпляр бота.

    Returns:
        None
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
    if intent == "history":
        await cmd_history(msg)
        return

    url = raw
    uid, _ = get_user_and_chat(msg)
    log_info(logger, "Запрос", user_id=uid, url=url)
    if not url:
        await msg.answer("⚠️ Пустой запрос.")
        return
    if is_url(url):
        log_info(logger, "Обнаружена ссылка. Показываю карточку выбора", user_id=uid, url=url)
        if uid is None:
            log_info(logger, "Не удалось определить пользователя для ссылки.")
            await msg.answer("⚠️ Не удалось определить пользователя.")
            return
        token = save_pending_url(uid, url)
        kb = build_download_choice_kb(uid, token)
        await send_info_card(bot, msg.chat.id, url, uid, reply_markup=kb.as_markup())
        return
    query = sanitize_query(url)
    if not query:
        log_info(logger, "Пустой или некорректный поисковый запрос", user_id=uid)
        await msg.answer("⚠️ Некорректный запрос.")
        return
    log_info(logger, "Начинаю поиск", user_id=uid, extra={"query": query})
    await msg.answer("🔎 Ищу")
    try:
        cookies_path = get_user_cookies_path(uid) if uid is not None else None
        results = await search_tracks(query, cookies_path=cookies_path)
        log_info(logger, "Поиск завершён", user_id=uid, extra={"found": len(results)})
        if uid is not None:
            set_searches(uid, {"results": results, "page": 0})
        if not results:
            log_info(logger, "Ничего не найдено", user_id=uid)
            await msg.answer("🙁 Ничего не найдено (или превышен лимит длительности).")
            return
        kb = build_results_kb(uid if uid is not None else 0)
        log_info(logger, "Показываю результаты поиска", user_id=uid)
        await msg.answer("📋 Результаты поиска:", reply_markup=kb.as_markup())
    except DownloadError as e:
        if uid is not None:
            remember_search_cookie_request(uid, query)
        log_info(logger, "Поиск требует cookies", user_id=uid, extra={"err": str(e)})
        await msg.answer(
            "🍪 Источник требует cookies или защиту (YouTube может просить вход).\n"
            "Пришлите файл cookies.txt — повторю поиск с cookies."
        )
    except Exception as e:
        log_exception(logger, "Ошибка поиска", user_id=uid, extra={"query": query, "err": str(e)})
        await msg.answer("❌ Ошибка поиска. Попробуйте позже.")


@router.message(F.document)
async def handle_document(msg: Message, bot: Bot) -> None:
    """Обрабатывает полученный документ (файл cookies).

    Args:
        msg (Message): Входящее сообщение.
        bot (Bot): Экземпляр бота.

    Returns:
        None
    """
    user_id, _ = get_user_and_chat(msg)
    if user_id is None:
        log_info(logger, "Получен файл, но не удалось определить пользователя.")
        await msg.answer("📄 Файл получен, но не удалось определить пользователя.")
        return
    pending = get_awaiting(user_id)
    if not pending:
        log_info(logger, "Получен файл, но cookies не требуются.", user_id=user_id)
        await msg.answer("📄 Файл получен, но сейчас cookies не требуются.")
        return

    cookies_path = get_user_cookies_path(user_id)
    doc = msg.document
    if doc is None:
        log_info(logger, "Не удалось прочитать файл cookies.", user_id=user_id)
        await msg.answer("❌ Не удалось прочитать файл.")
        return

    name_l = (doc.file_name or "").lower()
    ext = os.path.splitext(name_l)[1]
    size = doc.file_size or 0
    log_info(logger, "Получен файл cookies", user_id=user_id, extra={"file_name": doc.file_name, "size": size})
    if ext not in ALLOWED_COOKIES_EXTS:
        log_info(logger, "Некорректный формат файла cookies", user_id=user_id, extra={"ext": ext})
        await msg.answer("⚠️ Нужен файл cookies в формате Netscape: cookies.txt.")
        return
    if size and size > COOKIES_MAX_BYTES:
        lim_mb = COOKIES_MAX_BYTES / (1024 * 1024)
        cur_mb = size / (1024 * 1024)
        log_info(logger, "Слишком большой файл cookies", user_id=user_id, extra={"cur_mb": cur_mb, "lim_mb": lim_mb})
        await msg.answer(f"⚠️ Слишком большой cookies.txt ({cur_mb:.1f} МБ). Максимум {lim_mb:.0f} МБ.")
        return

    try:
        await bot.download(doc, destination=cookies_path)
        with suppress(Exception):
            real_size = await to_thread(os.path.getsize, cookies_path)
            log_info(logger, "Cookies сохранены", user_id=user_id, extra={"path": cookies_path, "size": real_size})
    except Exception as e:
        log_exception(logger, "Не удалось сохранить файл cookies", user_id=user_id, extra={"err": str(e)})
        await msg.answer("❌ Не удалось сохранить cookies.txt.")
        return

    with suppress(Exception):
        real_size = await to_thread(os.path.getsize, cookies_path)
        if real_size > COOKIES_MAX_BYTES:
            lim_mb = COOKIES_MAX_BYTES / (1024 * 1024)
            cur_mb = real_size / (1024 * 1024)
            with suppress(Exception):
                await to_thread(os.remove, cookies_path)
            log_info(
                logger,
                "Слишком большой сохранённый файл cookies",
                user_id=user_id,
                extra={"cur_mb": cur_mb, "lim_mb": lim_mb},
            )
            await msg.answer(f"⚠️ Слишком большой cookies.txt ({cur_mb:.1f} МБ). Максимум {lim_mb:.0f} МБ.")
            return

    log_info(logger, "Повтор операции с cookies", user_id=user_id)
    await msg.answer("🍪 Cookies получены. Пробую снова")

    pending_kind = (pending.get("kind") or "").lower()
    if pending_kind == "search":
        query_any = pending.get("query")
        if not isinstance(query_any, str) or not query_any.strip():
            log_info(logger, "Нет запроса для повтора поиска с cookies", user_id=user_id)
            await msg.answer("❌ Нет запроса для повтора поиска.")
            return
        query = sanitize_query(query_any)
        log_info(logger, "Повтор поиска с cookies", user_id=user_id, extra={"query": query})
        pop_awaiting(user_id)
        try:
            results = await search_tracks(query, cookies_path=cookies_path)
            log_info(logger, "Поиск с cookies завершён", user_id=user_id, extra={"found": len(results)})
            set_searches(user_id, {"results": results, "page": 0})
            if not results:
                log_info(logger, "Ничего не найдено с cookies", user_id=user_id)
                await msg.answer("🙁 Ничего не найдено даже с cookies.")
                return
            kb = build_results_kb(user_id)
            log_info(logger, "Показываю результаты поиска с cookies", user_id=user_id)
            await msg.answer("📋 Результаты поиска:", reply_markup=kb.as_markup())
        except Exception as e:
            log_exception(logger, "Ошибка поиска с cookies", user_id=user_id, extra={"query": query, "err": str(e)})
            await msg.answer("❌ Не удалось выполнить поиск даже с cookies.")
        return

    url_any = pending.get("url")
    if not isinstance(url_any, str) or not url_any:
        log_info(logger, "Нет URL для повтора загрузки с cookies", user_id=user_id)
        await msg.answer("❌ Нет URL для повтора.")
        return
    url = url_any

    pending_mode = pending.get("mode")
    if isinstance(pending_mode, str) and pending_mode in {
        "audio",
        "video",
        "video_nosound",
    }:
        mode = pending_mode
    elif pending_mode == "auto":
        mode = decide_effective_mode(get_user_mode(user_id), url)
    else:
        mode = decide_effective_mode(get_user_mode(user_id), url)

    log_info(logger, "Повтор загрузки с cookies", user_id=user_id, mode=mode, url=url)

    pop_awaiting(user_id)
    lock = await begin_user_download(user_id)
    if not lock:
        log_info(logger, "Не удалось начать загрузку с cookies: другая загрузка идёт", user_id=user_id)
        await msg.answer("⏳ Идёт другая загрузка. Дождитесь завершения.")
        return

    async def on_nothing():
        await msg.answer("😕 Не удалось скачать даже с cookies (возможно, превышен лимит длительности).")

    async def on_error():
        await msg.answer("❌ Не удалось скачать даже с cookies. Скипаю.")

    cancel_kb = InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(text="❌ Отмена", callback_data="download:cancel")]]
    )
    status_msg = await msg.answer("⏳ Скачиваю, подождите", reply_markup=cancel_kb)

    async def _run_and_cleanup():
        try:
            await perform_download(
                bot=bot,
                chat_id=msg.chat.id,
                user_id=user_id,
                url=url,
                mode=mode,
                lock=lock,
                cookies_path=cookies_path,
                on_nothing=on_nothing,
                on_error=on_error,
                status_message=status_msg,
            )
        finally:
            with suppress(Exception):
                await pop_download_task(user_id)

    task = asyncio.create_task(_run_and_cleanup())
    set_download_task(user_id, task)
