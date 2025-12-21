from contextlib import suppress
from typing import Any, Dict, List

from aiogram import Bot, F
from aiogram.types import (
    Message,
    CallbackQuery,
)
from yt_dlp import YoutubeDL  # type: ignore[import-untyped]
from yt_dlp.utils import DownloadError  # type: ignore[import-untyped]

from bot.dispatcher import router, logger
from bot.keyboards import (
    build_settings_kb,
    build_download_choice_kb,
    build_results_kb,
    build_history_kb,
)
from config import (
    PAGE_SIZE,
)
from services.telegram import (
    send_info_card,
)
from services.ytdlp import (
    decide_effective_mode,
    download_media_to_temp,
)
from storage.state import (
    get_searches,
    get_awaiting,
    get_pending,
    pop_pending,
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
)
from bot.handlers.downloads import perform_download
from bot.helpers import (
    safe_answer,
    safe_edit_markup,
    safe_delete_msg,
    get_user_and_chat,
)


@router.callback_query(F.data == "settings:open")
async def cb_settings_open(cb: CallbackQuery) -> None:
    """Callback открытия настроек.

    Args:
        cb (CallbackQuery): Запрос.
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
    """
    await safe_answer(cb)
    await safe_delete_msg(cb.message)
    await safe_edit_markup(cb.message, None)


@router.callback_query(F.data.startswith("setmode:"))
async def cb_set_mode(cb: CallbackQuery) -> None:
    """Выбор режима скачивания.

    Args:
        cb (CallbackQuery): Запрос с режимом.
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
    logger.info("Режим пользователя %s изменён на %s", cb.from_user.id, mode)
    kb = build_settings_kb(cb.from_user.id)
    await safe_edit_markup(cb.message, kb)
    await safe_answer(cb, "✅ Режим обновлён.")


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

    with suppress(Exception):
        pop_pending(token)

    if mode_sel == "auto":
        mode = decide_effective_mode(get_user_mode(user_id), url)
    else:
        mode = mode_sel

    logger.info(
        "Выбор скачивания: user=%s, mode=%s, url=%s", str(user_id), mode, url[:200]
    )

    await safe_edit_markup(cb.message, None)

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
    await bot.send_message(chat_id, "⏳ Скачиваю, подождите")

    cookies_path = get_user_cookies_path(user_id)

    async def on_cookies_required():
        remember_cookie_request(user_id, kind="download", url=url)
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
        await bot.send_message(
            chat_id, "❌ Произошла ошибка при загрузке. Попробуйте позже."
        )

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
    await safe_answer(cb)


@router.callback_query(F.data == "cancel")
async def handle_cancel(cb: CallbackQuery) -> None:
    """Отмена списка результатов и ожидания cookies.

    Args:
        cb (CallbackQuery): Запрос.
    """
    user_id, _ = get_user_and_chat(cb)
    if user_id is not None:
        pop_searches(user_id)
        pop_awaiting(user_id)
    await safe_edit_markup(cb.message, None)
    await safe_answer(cb, "❌ Отменено.")


@router.callback_query(F.data == "page:next")
async def handle_next_page(cb: CallbackQuery) -> None:
    """Переход к следующей странице результатов.

    Args:
        cb (CallbackQuery): Запрос.
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
    await safe_edit_markup(cb.message, kb)
    await safe_answer(cb)


@router.callback_query(F.data == "page:prev")
async def handle_prev_page(cb: CallbackQuery) -> None:
    """Переход к предыдущей странице результатов.

    Args:
        cb (CallbackQuery): Запрос.
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
    await safe_edit_markup(cb.message, kb)
    await safe_answer(cb)


@router.callback_query(F.data.startswith("pick:"))
async def handle_pick(cb: CallbackQuery, bot: Bot) -> None:
    """Начинает загрузку выбранного результата."""
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

        logger.info(
            "Выбор результата #%d пользователем %s: %s",
            idx,
            cb.from_user.id,
            (url or "")[:200],
        )

        token = save_pending_url(cb.from_user.id, url)
        kb = build_download_choice_kb(cb.from_user.id, token)

        await safe_answer(cb)

        with suppress(Exception):
            pop_searches(cb.from_user.id)
        await safe_delete_msg(cb.message)
        await safe_edit_markup(cb.message, None)

        _, chat_id = get_user_and_chat(cb)
        if chat_id is not None and cb.message is not None and isinstance(cb.message, Message):
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
    await safe_answer(cb)
    await safe_delete_msg(cb.message)
    await safe_edit_markup(cb.message, None)
    user_id, _ = get_user_and_chat(cb)
    if user_id is not None:
        reset_history_page(user_id)


@router.callback_query(F.data == "history:page:next")
async def cb_history_next(cb: CallbackQuery) -> None:
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
    await safe_edit_markup(cb.message, kb)
    await safe_answer(cb)


@router.callback_query(F.data == "history:page:prev")
async def cb_history_prev(cb: CallbackQuery) -> None:
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
    await safe_edit_markup(cb.message, kb)
    await safe_answer(cb)


@router.callback_query(F.data.startswith("history:show:"))
async def cb_history_show(cb: CallbackQuery) -> None:
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
        if chat_id is not None and cb.message is not None and isinstance(cb.message, Message):
            await cb.message.answer(text)
        return
