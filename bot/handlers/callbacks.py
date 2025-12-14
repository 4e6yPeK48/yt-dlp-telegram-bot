from contextlib import suppress
from typing import Any, Dict, List

from aiogram import Bot, F
from aiogram.types import (
    Message,
    CallbackQuery,
)
from yt_dlp import YoutubeDL  # type: ignore[import-untyped]
from yt_dlp.utils import DownloadError  # type: ignore[import-untyped]

from ...bot.dispatcher import router, logger
from ...bot.keyboards import (
    build_settings_kb,
    build_download_choice_kb,
    build_results_kb
)
from ...config import (
    PAGE_SIZE,
)
from ...services.telegram import (
    send_info_card,
    get_cb_chat_id,
    try_cb_answer,
    send_by_mode
)
from ...services.ytdlp import (
    decide_effective_mode,
    download_media_to_temp,
)
from ...storage.state import (
    USER_SEARCHES,
    AWAITING_COOKIES,
    PENDING_DOWNLOADS,
    get_user_mode,
    set_user_mode,
    begin_user_download,
    end_user_download,
    slice_page,
    remember_cookie_request,
    get_user_cookies_path,
    save_pending_url
)
from downloads import perform_download

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
        await end_user_download(lock)
        await try_cb_answer(cb)
        return

    await try_cb_answer(cb)
    await bot.send_message(chat_id, "⏳ Скачиваю, подождите...")

    cookies_path = get_user_cookies_path(user_id)

    async def on_cookies_required():
        remember_cookie_request(user_id, kind="download", url=url, mode=mode)
        await bot.send_message(
            chat_id,
            "🍪 Источник требует cookies или произошла ошибка.\nПришлите файл cookies.txt для повтора попытки.",
        )

    async def on_nothing():
        await bot.send_message(
            chat_id,
            "😕 Нечего отправлять. Возможно, превышен лимит длительности (30 минут).",
        )

    async def on_error():
        await bot.send_message(chat_id, "❌ Произошла ошибка при загрузке. Попробуйте позже.")

    await perform_download(
        bot=bot,
        chat_id=chat_id,
        user_id=user_id,
        url=url,
        mode=mode,
        lock=lock,
        cookies_path=cookies_path,
        on_cookies_required=on_cookies_required,
        on_nothing=on_nothing,
        on_error=on_error,
    )


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
