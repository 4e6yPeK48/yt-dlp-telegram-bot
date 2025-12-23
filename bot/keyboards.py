from typing import Dict, List, Any

from aiogram.types import InlineKeyboardButton, ReplyKeyboardMarkup, KeyboardButton
from aiogram.utils.keyboard import InlineKeyboardBuilder

from config import PAGE_SIZE, BTN_MENU, BTN_HELP, BTN_SETTINGS, BTN_HISTORY
from storage.state import (
    slice_page,
    get_searches,
    get_user_mode,
    get_history,
    get_history_page,
)


def build_results_kb(user_id: int) -> InlineKeyboardBuilder:
    """Строит инлайн-клавиатуру результатов поиска с пагинацией.

    Args:
        user_id (int): Идентификатор пользователя.

    Returns:
        InlineKeyboardBuilder: Сконструированный билдер.
    """
    state = get_searches(user_id) or {}
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


def build_main_reply_kb() -> ReplyKeyboardMarkup:
    """Строит основную reply-клавиатуру.

    Returns:
        ReplyKeyboardMarkup: Клавиатура с основными командами.
    """
    return ReplyKeyboardMarkup(
        keyboard=[
            [
                KeyboardButton(text=BTN_MENU),
                KeyboardButton(text=BTN_HELP),
                KeyboardButton(text=BTN_SETTINGS),
                KeyboardButton(text=BTN_HISTORY),
            ],
        ],
        resize_keyboard=True,
        is_persistent=True,
    )


def build_history_kb(user_id: int) -> InlineKeyboardBuilder:
    items = get_history(user_id) or []
    page = get_history_page(user_id)
    current, pages = slice_page(items, page, PAGE_SIZE)
    kb = InlineKeyboardBuilder()

    if not items:
        kb.button(text="История пуста", callback_data="noop")
        kb.adjust(1)
    else:
        for idx, entry in enumerate(current):
            global_index = page * PAGE_SIZE + idx
            title = entry.get("title") or entry.get("url") or "Без названия"
            if len(title) > 64:
                title = title[:61] + "..."
            dur = entry.get("duration")
            suffix = ""
            if isinstance(dur, (int, float)):
                m, s = divmod(int(dur), 60)
                suffix = f" [{m}:{s:02d}]"
            kb.button(text=f"{title}{suffix}", callback_data=f"history:show:{global_index}")
        kb.adjust(1)

        kb.row(
            InlineKeyboardButton(text="« Назад", callback_data="history:page:prev"),
            InlineKeyboardButton(text=f"{page + 1}/{pages}", callback_data="noop"),
            InlineKeyboardButton(text="Вперёд »", callback_data="history:page:next"),
        )

    kb.row(InlineKeyboardButton(text="Закрыть", callback_data="history:close"))
    return kb
