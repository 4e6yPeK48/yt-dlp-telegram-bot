import os
from contextlib import suppress

from aiogram import Bot, F
from aiogram.types import (
    Message,
)
from yt_dlp import YoutubeDL  # type: ignore[import-untyped]
from yt_dlp.utils import DownloadError  # type: ignore[import-untyped]

from ...bot.dispatcher import router, logger
from ...bot.handlers.commands import (
    cmd_start,
    cmd_help,
    cmd_settings
)
from ...bot.keyboards import (
    build_download_choice_kb,
    build_results_kb
)
from ...config import (
    COOKIES_MAX_BYTES,
    ALLOWED_COOKIES_EXTS,
)
from ...services.telegram import (
    send_info_card,
    send_by_mode
)
from ...services.ytdlp import (
    decide_effective_mode,
    search_tracks,
    download_media_to_temp,
)
from ...storage.state import (
    USER_SEARCHES,
    AWAITING_COOKIES,
    get_user_mode,
    begin_user_download,
    end_user_download,
    remember_search_cookie_request,
    get_user_cookies_path,
    save_pending_url
)
from ...utils.text import sanitize_query, parse_main_button_intent
from ...utils.validators import is_url


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
            logger.info("Слишком большой сохранённый файл cookies от %s: %.2f МБ (лимит %.0f МБ)", msg.from_user.id,
                        cur_mb, lim_mb)
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
        logger.info("Загрузка с cookies завершена: файлов к отправке %d (user=%s, mode=%s)", len(files),
                    msg.from_user.id, mode)
        await send_by_mode(bot, msg.chat.id, mode, files)
        logger.info("Отправка (cookies) завершена: отправлено %d файлов (user=%s, mode=%s)", len(files),
                    msg.from_user.id, mode)
    except Exception:
        logger.info("Ошибка при загрузке с cookies (user=%s, mode=%s)", msg.from_user.id, mode)
        await msg.answer("❌ Не удалось скачать даже с cookies. Скипаю.")
    finally:
        await end_user_download(lock)