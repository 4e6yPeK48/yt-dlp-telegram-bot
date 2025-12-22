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
        await send_info_card(bot, msg.chat.id, url, uid, reply_markup=kb.as_markup())
        return
    query = sanitize_query(url)
    if not query:
        logger.info("Пустой или некорректный поисковый запрос (user=%s)", str(uid))
        await msg.answer("⚠️ Некорректный запрос.")
        return
    logger.info("Начинаю поиск (user=%s, query=%s)", str(uid), query[:120])
    await msg.answer("🔎 Ищу")
    try:
        cookies_path = get_user_cookies_path(uid) if uid is not None else None
        results = await search_tracks(query, cookies_path=cookies_path)
        logger.info("Поиск завершён: найдено %d (user=%s)", len(results), str(uid))
        if uid is not None:
            set_searches(uid, {"results": results, "page": 0})
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
        logger.info("Поиск требует cookies (user=%s): %s", str(uid), str(e))
        await msg.answer(
            "🍪 Источник требует cookies или защиту (YouTube может просить вход).\n"
            "Пришлите файл cookies.txt — повторю поиск с cookies."
        )
    except Exception as e:
        logger.exception('Ошибка поиска для "%s" (user=%s): %s', query, str(uid), e)
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
        logger.info("Получен файл, но не удалось определить пользователя.")
        await msg.answer("📄 Файл получен, но не удалось определить пользователя.")
        return
    pending = get_awaiting(user_id)
    if not pending:
        logger.info("Получен файл от %s, но cookies не требуются.", user_id)
        await msg.answer("📄 Файл получен, но сейчас cookies не требуются.")
        return

    cookies_path = get_user_cookies_path(user_id)
    doc = msg.document
    if doc is None:
        logger.info("Не удалось прочитать файл cookies от %s.", user_id)
        await msg.answer("❌ Не удалось прочитать файл.")
        return

    name_l = (doc.file_name or "").lower()
    ext = os.path.splitext(name_l)[1]
    size = doc.file_size or 0
    logger.info(
        "Получен файл cookies от %s: %s (%d байт)",
        user_id,
        doc.file_name,
        size,
    )
    if ext not in ALLOWED_COOKIES_EXTS:
        logger.info(
            "Некорректный формат файла cookies от %s: %s", user_id, ext
        )
        await msg.answer("⚠️ Нужен файл cookies в формате Netscape: cookies.txt.")
        return
    if size and size > COOKIES_MAX_BYTES:
        lim_mb = COOKIES_MAX_BYTES / (1024 * 1024)
        cur_mb = size / (1024 * 1024)
        logger.info(
            "Слишком большой файл cookies от %s: %.2f МБ (лимит %.0f МБ)",
            user_id,
            cur_mb,
            lim_mb,
        )
        await msg.answer(
            f"⚠️ Слишком большой cookies.txt ({cur_mb:.1f} МБ). Максимум {lim_mb:.0f} МБ."
        )
        return

    try:
        await bot.download(doc, destination=cookies_path)
        with suppress(Exception):
            real_size = await to_thread(os.path.getsize, cookies_path)
            logger.info(
                "Cookies сохранены для %s: %s (%d байт)",
                user_id,
                cookies_path,
                real_size,
            )
    except Exception as e:
        logger.exception("Не удалось сохранить файл cookies от %s: %s", user_id, e)
        await msg.answer("❌ Не удалось сохранить cookies.txt.")
        return

    with suppress(Exception):
        real_size = await to_thread(os.path.getsize, cookies_path)
        if real_size > COOKIES_MAX_BYTES:
            lim_mb = COOKIES_MAX_BYTES / (1024 * 1024)
            cur_mb = real_size / (1024 * 1024)
            with suppress(Exception):
                await to_thread(os.remove, cookies_path)
            logger.info(
                "Слишком большой сохранённый файл cookies от %s: %.2f МБ (лимит %.0f МБ)",
                user_id,
                cur_mb,
                lim_mb,
            )
            await msg.answer(
                f"⚠️ Слишком большой cookies.txt ({cur_mb:.1f} МБ). Максимум {lim_mb:.0f} МБ."
            )
            return

    logger.info("Повтор операции с cookies для %s.", user_id)
    await msg.answer("🍪 Cookies получены. Пробую снова")

    pending_kind = (pending.get("kind") or "").lower()
    if pending_kind == "search":
        query_any = pending.get("query")
        if not isinstance(query_any, str) or not query_any.strip():
            logger.info(
                "Нет запроса для повтора поиска с cookies от %s.", user_id
            )
            await msg.answer("❌ Нет запроса для повтора поиска.")
            return
        query = query_any.strip()
        logger.info(
            "Повтор поиска с cookies (user=%s, query=%s)", user_id, query[:120]
        )
        pop_awaiting(user_id)
        try:
            results = await search_tracks(query, cookies_path=cookies_path)
            logger.info(
                "Поиск с cookies: найдено %d (user=%s)", len(results), user_id
            )
            set_searches(user_id, {"results": results, "page": 0})
            if not results:
                logger.info("Ничего не найдено с cookies от %s.", user_id)
                await msg.answer("🙁 Ничего не найдено даже с cookies.")
                return
            kb = build_results_kb(user_id)
            logger.info(
                "Показываю результаты поиска с cookies (user=%s)", user_id
            )
            await msg.answer("📋 Результаты поиска:", reply_markup=kb.as_markup())
        except Exception as e:
            logger.exception("Ошибка поиска с cookies от %s (query=%s): %s", user_id, query, e)
            await msg.answer("❌ Не удалось выполнить поиск даже с cookies.")
        return

    url_any = pending.get("url")
    if not isinstance(url_any, str) or not url_any:
        logger.info("Нет URL для повтора загрузки с cookies от %s.", user_id)
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

    logger.info(
        "Повтор загрузки с cookies (user=%s, mode=%s, url=%s)",
        user_id,
        mode,
        url[:200],
    )

    pop_awaiting(user_id)
    lock = await begin_user_download(user_id)
    if not lock:
        logger.info(
            "Не удалось начать загрузку с cookies: другая загрузка идёт (user=%s)",
            user_id,
        )
        await msg.answer("⏳ Идёт другая загрузка. Дождитесь завершения.")
        return

    async def on_nothing():
        await msg.answer(
            "😕 Не удалось скачать даже с cookies (возможно, превышен лимит длительности)."
        )

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
