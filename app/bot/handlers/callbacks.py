from contextlib import suppress
from typing import Any, Dict, List, Optional
import asyncio

from aiogram import Bot, F
from aiogram.types import (
    Message,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)

from app.bot.dispatcher import router, logger
from app.bot.keyboards import (
    build_settings_kb,
    build_download_choice_kb,
    build_results_kb,
    build_history_kb,
)
from app.config import (
    PAGE_SIZE,
)
from app.services.telegram import (
    send_info_card,
)
from app.services.ytdlp import (
    decide_effective_mode,
    find_server_cookies_for_url,
)
from app.storage.state import (
    get_searches,
    get_awaiting,
    get_pending,
    get_user_mode,
    set_user_mode,
    begin_user_download,
    end_user_download,
    slice_page,
    remember_cookie_request,
    get_user_cookies_path,
    save_pending_url,
    pop_searches,
    set_searches,
    pop_awaiting,
    get_history,
    set_history_page,
    get_history_page,
    reset_history_page,
    set_download_task,
    pop_download_task,
    cancel_download_task,
)
from app.bot.handlers.downloads import perform_download
from app.bot.helpers import (
    safe_answer,
    safe_edit_markup,
    safe_delete_msg,
    get_user_and_chat,
)
from app.utils.log_helpers import log_info


@router.callback_query(F.data == "settings:open")
async def cb_settings_open(cb: CallbackQuery) -> None:
    """Callback открытия настроек.

    Args:
        cb (CallbackQuery): Запрос.

    Returns:
        None
    """
    await safe_answer(cb)
    user_id, _ = get_user_and_chat(cb)
    if user_id is None:
        await safe_answer(cb, "⚠️ Не удалось определить пользователя.")
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

    Returns:
        None
    """
    await safe_answer(cb)
    message: Optional[Message] = cb.message if isinstance(cb.message, Message) else None
    await safe_delete_msg(message)
    await safe_edit_markup(message, None)


@router.callback_query(F.data.startswith("setmode:"))
async def cb_set_mode(cb: CallbackQuery) -> None:
    """Выбор режима скачивания.

    Args:
        cb (CallbackQuery): Запрос с режимом.

    Returns:
        None
    """
    data = cb.data or ""
    if not data.startswith("setmode:"):
        await safe_answer(cb, "⚠️ Некорректные данные.")
        return
    mode = data.split(":", 1)[1]
    if mode not in {"auto", "audio", "video", "video_nosound"}:
        await safe_answer(cb, "⚠️ Неизвестный режим.")
        return
    user_id, _ = get_user_and_chat(cb)
    if user_id is None:
        await safe_answer(cb, "⚠️ Не удалось определить пользователя.")
        return
    set_user_mode(cb.from_user.id, mode)
    log_info(logger, "Режим пользователя изменён", user_id=cb.from_user.id, mode=mode)
    kb = build_settings_kb(cb.from_user.id)
    message: Optional[Message] = cb.message if isinstance(cb.message, Message) else None
    await safe_edit_markup(message, kb)
    await safe_answer(cb, "✅ Режим обновлён.")


@router.callback_query(F.data.startswith("dl:"))
async def cb_download_choice(cb: CallbackQuery, bot: Bot) -> None:
    """Обрабатывает выбор режима скачивания для сохранённого URL.

    Args:
        cb (CallbackQuery): Callback с данными вида dl:<mode>:<token>.
        bot (Bot): Экземпляр бота.

    Returns:
        None
    """
    data = cb.data or ""
    parts = data.split(":")
    if len(parts) != 3:
        await safe_answer(cb, "⚠️ Некорректные данные.")
        return
    _, mode_sel, token = parts
    if mode_sel not in {"audio", "video", "auto"}:
        await safe_answer(cb, "⚠️ Неизвестный режим.")
        return
    pend = get_pending(token)
    if not pend:
        await safe_answer(cb, "ℹ️ Ссылка устарела. Отправьте её снова.")
        return
    user_id = pend.get("user_id")
    url = pend.get("url")
    if not isinstance(user_id, int) or not isinstance(url, str):
        await safe_answer(cb, "⚠️ Ошибка данных.")
        return

    if mode_sel == "auto":
        mode = decide_effective_mode(get_user_mode(user_id), url)
    else:
        mode = mode_sel

    log_info(logger, "Выбор скачивания", user_id=user_id, mode=mode, url=url)

    message: Optional[Message] = cb.message if isinstance(cb.message, Message) else None
    await safe_edit_markup(message, None)

    lock = await begin_user_download(user_id)
    if not lock:
        await safe_answer(cb, "⏳ Идёт другая загрузка.")
        return

    _, chat_id = get_user_and_chat(cb)
    if chat_id is None:
        await end_user_download(lock)
        await safe_answer(cb, "⚠️ Не удалось определить чат.")
        return

    await safe_answer(cb)

    cancel_kb = build_cancel_kb()
    status_msg = await bot.send_message(
        chat_id, "⏳ Скачиваю, подождите", reply_markup=cancel_kb
    )

    cookies_path = get_user_cookies_path(user_id)

    async def on_cookies_required() -> None:
        remember_cookie_request(user_id, kind="download", url=url)
        kb = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="🍪 Пришлю cookies.txt",
                        callback_data="ask_cookies",
                    ),
                    InlineKeyboardButton(
                        text="⚙️ Попытка с серверными cookies",
                        callback_data=f"download:server:{token}",
                    ),
                ]
            ]
        )
        if chat_id is not None:
            await bot.send_message(
                chat_id,
                "🍪 Источник требует cookies или произошла ошибка.\nПришлите файл cookies.txt для повтора попытки — или попробуйте загрузку с серверными cookies (если доступно).",
                reply_markup=kb,
            )

    async def on_nothing() -> None:
        await bot.send_message(
            chat_id,
            "😕 Нечего отправлять. Возможно, превышен лимит длительности (30 минут).",
        )

    async def on_error() -> None:
        await bot.send_message(
            chat_id, "❌ Произошла ошибка при загрузке. Попробуйте позже."
        )

    async def _run_and_cleanup() -> None:
        try:
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
                status_message=status_msg,
            )
        finally:
            with suppress(Exception):
                await pop_download_task(user_id)

    task = asyncio.create_task(_run_and_cleanup())
    set_download_task(user_id, task)


@router.callback_query(F.data == "ask_cookies")
async def cb_ask_cookies(cb: CallbackQuery) -> None:
    """
    Запрашивает у пользователя файл cookies.txt.

    Args:
        cb (CallbackQuery): Запрос.

    Returns:
        None
    """
    await safe_answer(cb)
    user_id, _ = get_user_and_chat(cb)
    if user_id is None:
        return
    pending = get_awaiting(user_id)
    if not pending:
        return
    if cb.message is not None and isinstance(cb.message, Message):
        try:
            await cb.message.answer(
                "🍪 Пришлите файл `cookies.txt` в формате Netscape как документ (макс. 5 МБ). "
                "После получения бот автоматически повторит операцию."
            )
        except Exception:
            pass


@router.callback_query(F.data.startswith("download:server:"))  # ПОКА НЕ РАБОТАЕТ
async def cb_download_server(cb: CallbackQuery, bot: Bot) -> None:
    """
    Начинает загрузку с использованием серверных cookies.

    Args:
        cb (CallbackQuery): Запрос.
        bot (Bot): Экземпляр бота.

    Returns:
        None
    """
    await safe_answer(cb)
    data = cb.data or ""
    log_info(
        logger,
        "cb_download_server invoked",
        user_id=getattr(cb.from_user, "id", None),
        extra={"data": data},
    )

    parts = data.split(":", 2)
    if len(parts) != 3:
        await safe_answer(cb, "⚠️ Некорректные данные.")
        return
    _, _, token = parts
    pend = get_pending(token)
    if not pend:
        await safe_answer(cb, "ℹ️ Ссылка устарела. Отправьте её снова.")
        return
    user_id = pend.get("user_id")
    url = pend.get("url")
    if not isinstance(user_id, int) or not isinstance(url, str):
        await safe_answer(cb, "⚠️ Ошибка данных.")
        return

    server_cookies = await find_server_cookies_for_url(url)
    if not server_cookies:
        await safe_answer(cb, "ℹ️ Серверные cookies для этого источника недоступны.")
        return

    mode = decide_effective_mode(get_user_mode(user_id), url)

    log_info(
        logger,
        "Попытка загрузки с серверными cookies",
        user_id=user_id,
        mode=mode,
        url=url,
    )

    message: Optional[Message] = cb.message if isinstance(cb.message, Message) else None
    await safe_edit_markup(message, None)

    lock = await begin_user_download(user_id)
    if not lock:
        await safe_answer(cb, "⏳ Идёт другая загрузка.")
        return

    _, chat_id = get_user_and_chat(cb)
    if chat_id is None:
        await end_user_download(lock)
        await safe_answer(cb, "⚠️ Не удалось определить чат.")
        return

    await safe_answer(cb)

    cancel_kb = build_cancel_kb()
    status_msg = await bot.send_message(
        chat_id, "⏳ Скачиваю (серверные cookies), подождите", reply_markup=cancel_kb
    )

    async def on_cookies_required_inner() -> None:
        remember_cookie_request(user_id, kind="download", url=url)
        await bot.send_message(
            chat_id,
            "🍪 Серверные cookies не помогли. Пришлите файл cookies.txt для повтора попытки.",
        )

    async def on_nothing_inner() -> None:
        await bot.send_message(
            chat_id,
            "😕 Нечего отправлять. Возможно, превышен лимит длительности (30 минут).",
        )

    async def on_error_inner() -> None:
        await bot.send_message(
            chat_id, "❌ Произошла ошибка при загрузке. Попробуйте позже."
        )

    async def _run_and_cleanup() -> None:
        try:
            await perform_download(
                bot=bot,
                chat_id=chat_id,
                user_id=user_id,
                url=url,
                mode=mode,
                lock=lock,
                cookies_path=server_cookies,
                on_cookies_required=on_cookies_required_inner,
                on_nothing=on_nothing_inner,
                on_error=on_error_inner,
                status_message=status_msg,
            )
        finally:
            with suppress(Exception):
                await pop_download_task(user_id)

    task = asyncio.create_task(_run_and_cleanup())
    set_download_task(user_id, task)


@router.callback_query(F.data == "noop")
async def handle_noop(cb: CallbackQuery) -> None:
    """Пустой callback.

    Args:
        cb (CallbackQuery): Запрос.

    Returns:
        None
    """
    await safe_answer(cb)


@router.callback_query(F.data == "cancel")
async def handle_cancel(cb: CallbackQuery) -> None:
    """Отмена списка результатов и ожидания cookies.

    Args:
        cb (CallbackQuery): Запрос.

    Returns:
        None
    """
    user_id, _ = get_user_and_chat(cb)
    if user_id is not None:
        pop_searches(user_id)
        pop_awaiting(user_id)
    message: Optional[Message] = cb.message if isinstance(cb.message, Message) else None
    await safe_edit_markup(message, None)
    await safe_answer(cb, "❌ Отменено.")


@router.callback_query(F.data == "page:next")
async def handle_next_page(cb: CallbackQuery) -> None:
    """Переход к следующей странице результатов.

    Args:
        cb (CallbackQuery): Запрос.

    Returns:
        None
    """
    user_id, _ = get_user_and_chat(cb)
    if user_id is None:
        await safe_answer(cb, "⚠️ Не удалось определить пользователя.")
        return
    state = get_searches(user_id)
    if not state:
        await safe_answer(cb, "ℹ️ Нет активного списка.")
        return
    results = state["results"]
    page = state.get("page", 0)
    _, pages = slice_page(results, page, PAGE_SIZE)
    state["page"] = (page + 1) % pages
    set_searches(user_id, state)
    kb = build_results_kb(user_id)
    message: Optional[Message] = cb.message if isinstance(cb.message, Message) else None
    await safe_edit_markup(message, kb)
    await safe_answer(cb)


@router.callback_query(F.data == "page:prev")
async def handle_prev_page(cb: CallbackQuery) -> None:
    """Переход к предыдущей странице результатов.

    Args:
        cb (CallbackQuery): Запрос.

    Returns:
        None
    """
    user_id, _ = get_user_and_chat(cb)
    if user_id is None:
        await safe_answer(cb, "⚠️ Не удалось определить пользователя.")
        return
    state = get_searches(user_id)
    if not state:
        await safe_answer(cb, "ℹ️ Нет активного списка.")
        return
    results = state["results"]
    page = state.get("page", 0)
    _, pages = slice_page(results, page, PAGE_SIZE)
    state["page"] = (page - 1 + pages) % pages
    set_searches(user_id, state)
    kb = build_results_kb(user_id)
    message: Optional[Message] = cb.message if isinstance(cb.message, Message) else None
    await safe_edit_markup(message, kb)
    await safe_answer(cb)


@router.callback_query(F.data.startswith("pick:"))
async def handle_pick(cb: CallbackQuery, bot: Bot) -> None:
    """Начинает загрузку выбранного результата.

    Args:
        cb (CallbackQuery): Запрос с выбором.
        bot (Bot): Экземпляр бота.

    Returns:
        None
    """
    data = cb.data or ""
    if ":" not in data:
        await safe_answer(cb, "⚠️ Некорректные данные.")
        return
    idx_str = data.split(":", 1)[1]
    with suppress(ValueError):
        idx = int(idx_str)
        user_id, _ = get_user_and_chat(cb)
        if user_id is None:
            await safe_answer(cb, "ℹ️ Не удалось определить пользователя.")
            return
        state = get_searches(user_id)
        if not state:
            await safe_answer(cb, "ℹ️ Список результатов устарел.")
            return
        results: List[Dict[str, Any]] = state["results"]
        if idx < 0 or idx >= len(results):
            await safe_answer(cb, "⚠️ Некорректный выбор.")
            return
        url = results[idx].get("url")
        if not url:
            await safe_answer(cb, "⚠️ Нет URL для выбранного трека.")
            return

        log_info(
            logger,
            "Выбор результата",
            user_id=cb.from_user.id,
            url=url,
            extra={"choice_idx": idx},
        )

        token = save_pending_url(cb.from_user.id, url)
        kb = build_download_choice_kb(cb.from_user.id, token)

        await safe_answer(cb)

        with suppress(Exception):
            pop_searches(cb.from_user.id)
        # narrow message for typed helpers
        message: Optional[Message] = (
            cb.message if isinstance(cb.message, Message) else None
        )
        with suppress(Exception):
            await safe_delete_msg(message)
        await safe_edit_markup(message, None)

        _, chat_id = get_user_and_chat(cb)
        if (
            chat_id is not None
            and cb.message is not None
            and isinstance(cb.message, Message)
        ):
            await send_info_card(
                bot,
                chat_id,
                url,
                cb.from_user.id,
                reply_markup=kb.as_markup(),
            )
        return


@router.callback_query(F.data == "history:open")
async def cb_history_open(cb: CallbackQuery) -> None:
    """
    Открывает историю загрузок пользователя.

    Args:
        cb (CallbackQuery): Запрос.

    Returns:
        None
    """
    await safe_answer(cb)
    user_id, _ = get_user_and_chat(cb)
    if user_id is None:
        await safe_answer(cb, "⚠️ Не удалось определить пользователя.")
        return
    items = get_history(user_id) or []
    if not items:
        await safe_answer(cb, "ℹ️ История пуста.")
        return
    set_history_page(user_id, 0)
    kb = build_history_kb(user_id)
    if cb.message is not None and isinstance(cb.message, Message):
        with suppress(Exception):
            await cb.message.answer(
                "📜 Ваша история загрузок:", reply_markup=kb.as_markup()
            )


@router.callback_query(F.data == "history:close")
async def cb_history_close(cb: CallbackQuery) -> None:
    """
    Закрывает историю загрузок пользователя.

    Args:
        cb (CallbackQuery): Запрос.

    Returns:
        None
    """
    await safe_answer(cb)
    message: Optional[Message] = cb.message if isinstance(cb.message, Message) else None
    await safe_delete_msg(message)
    await safe_edit_markup(message, None)
    user_id, _ = get_user_and_chat(cb)
    if user_id is not None:
        reset_history_page(user_id)


@router.callback_query(F.data == "history:page:next")
async def cb_history_next(cb: CallbackQuery) -> None:
    """
    Переходит к следующей странице истории загрузок пользователя.

    Args:
        cb (CallbackQuery): Запрос.

    Returns:
        None
    """
    await safe_answer(cb)
    user_id, _ = get_user_and_chat(cb)
    if user_id is None:
        await safe_answer(cb, "⚠️ Не удалось определить пользователя.")
        return
    items = get_history(user_id) or []
    if not items:
        await safe_answer(cb, "ℹ️ Нет истории.")
        return
    page = get_history_page(user_id)
    _, pages = slice_page(items, page, PAGE_SIZE)
    page = (page + 1) % pages
    set_history_page(user_id, page)
    kb = build_history_kb(user_id)
    message: Optional[Message] = cb.message if isinstance(cb.message, Message) else None
    await safe_edit_markup(message, kb)
    await safe_answer(cb)


@router.callback_query(F.data == "history:page:prev")
async def cb_history_prev(cb: CallbackQuery) -> None:
    """
    Переходит к предыдущей странице истории загрузок пользователя.

    Args:
        cb (CallbackQuery): Запрос.

    Returns:
        None
    """
    await safe_answer(cb)
    user_id, _ = get_user_and_chat(cb)
    if user_id is None:
        await safe_answer(cb, "⚠️ Не удалось определить пользователя.")
        return
    items = get_history(user_id) or []
    if not items:
        await safe_answer(cb, "ℹ️ Нет истории.")
        return
    page = get_history_page(user_id)
    _, pages = slice_page(items, page, PAGE_SIZE)
    page = (page - 1 + pages) % pages
    set_history_page(user_id, page)
    kb = build_history_kb(user_id)
    message: Optional[Message] = cb.message if isinstance(cb.message, Message) else None
    await safe_edit_markup(message, kb)
    await safe_answer(cb)


@router.callback_query(F.data.startswith("history:show:"))
async def cb_history_show(cb: CallbackQuery) -> None:
    """
    Показывает детали выбранной записи из истории загрузок пользователя.

    Args:
        cb (CallbackQuery): Запрос.

    Returns:
        None
    """
    await safe_answer(cb)
    data = cb.data or ""
    if ":" not in data:
        await safe_answer(cb, "⚠️ Некорректные данные.")
        return
    with suppress(ValueError):
        idx = int(data.split(":", 2)[2])
        user_id, _ = get_user_and_chat(cb)
        if user_id is None:
            await safe_answer(cb, "⚠️ Не удалось определить пользователя.")
            return
        items = get_history(user_id) or []
        if idx < 0 or idx >= len(items):
            await safe_answer(cb, "⚠️ Некорректный выбор.")
            return
        entry = items[idx]
        title = entry.get("title") or "Без названия"
        url = entry.get("url") or "—"
        mode = entry.get("mode") or "—"
        duration = entry.get("duration")
        dur_str = "—"
        if isinstance(duration, (int, float)):
            m, s = divmod(int(duration), 60)
            dur_str = f"{m}:{s:02d}"
        t = entry.get("time")
        from datetime import datetime

        time_str = (
            datetime.fromtimestamp(t).isoformat(sep=" ", timespec="minutes")
            if t
            else "—"
        )
        text = (
            f"📦 История загрузки:\n\n"
            f"Название: {title}\n"
            f"Режим: {mode}\n"
            f"Длительность: {dur_str}\n"
            f"URL: {url}\n"
            f"Время: {time_str}"
        )
        _, chat_id = get_user_and_chat(cb)
        if (
            chat_id is not None
            and cb.message is not None
            and isinstance(cb.message, Message)
        ):
            await cb.message.answer(text)
        return


@router.callback_query(F.data == "download:cancel")
async def cb_download_cancel(cb: CallbackQuery) -> None:
    """
    Отмена текущей загрузки пользователя.

    Args:
        cb (CallbackQuery): Запрос.

    Returns:
        None
    """
    await safe_answer(cb)
    user_id, _ = get_user_and_chat(cb)
    if user_id is None:
        await safe_answer(cb, "⚠️ Не удалось определить пользователя.")
        return
    cancelled = cancel_download_task(user_id)
    log_info(
        logger,
        "Пользователь запросил отмену загрузки",
        user_id=user_id,
        extra={"result": "cancelled" if cancelled else "no_task"},
    )
    if cancelled:
        message: Optional[Message] = (
            cb.message if isinstance(cb.message, Message) else None
        )
        await safe_edit_markup(message, None)
        await safe_delete_msg(message)
        await safe_answer(cb, "❌ Загрузка отменена.")
    else:
        await safe_answer(cb, "ℹ️ Нет активной загрузки.")


def build_cancel_kb() -> InlineKeyboardMarkup:
    """
    Строит клавиатуру отмены.

    Returns:
        InlineKeyboardMarkup: Клавиатура с кнопкой отмены.
    """
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="❌ Отмена", callback_data="download:cancel")]
        ]
    )
