from contextlib import suppress
from typing import Any, Optional, Tuple, Union

from aiogram.types import CallbackQuery, Message


async def safe_answer(cb: CallbackQuery, text: Optional[str] = None) -> None:
    """
    Безопасно отвечает на CallbackQuery.

    Args:
        cb (CallbackQuery): Объект CallbackQuery.
        text (Optional[str]): Текст ответа.
    Returns:
        None
    """
    with suppress(Exception):
        await cb.answer(text)


async def safe_edit_markup(msg: Optional[Message], kb: Optional[Any]) -> None:
    """
    Безопасно редактирует разметку сообщения.

    Args:
        msg (Optional[Message]): Сообщение для редактирования.
        kb (Optional[Any]): Новая клавиатура или разметка.
    Returns:
        None
    """
    if msg is None or not isinstance(msg, Message):
        return
    with suppress(Exception):
        markup = None
        if kb is not None:
            try:
                markup = kb.as_markup()
            except Exception:
                markup = kb
        await msg.edit_reply_markup(reply_markup=markup)


async def safe_delete_msg(msg: Optional[Message]) -> None:
    """
    Безопасно удаляет сообщение.

    Args:
        msg (Optional[Message]): Сообщение для удаления.
    Returns:
        None
    """
    if msg is None or not isinstance(msg, Message):
        return
    with suppress(Exception):
        await msg.delete()


def get_user_and_chat(cb_or_msg: Union[CallbackQuery, Message]) -> Tuple[Optional[int], Optional[int]]:
    """
    Извлекает user_id и chat_id из CallbackQuery или Message.

    Args:
        cb_or_msg (Union[CallbackQuery, Message]): Объект CallbackQuery или Message
    Returns:
        Tuple[Optional[int], Optional[int]]: Кортеж (user_id, chat_id
    """
    if isinstance(cb_or_msg, Message):
        user_id = cb_or_msg.from_user.id if cb_or_msg.from_user is not None else None
        chat_id = cb_or_msg.chat.id if getattr(cb_or_msg, "chat", None) is not None else None
        return user_id, chat_id

    cb = cb_or_msg
    user_id = cb.from_user.id if cb.from_user is not None else None
    if cb.message is not None and isinstance(cb.message, Message):
        chat_id = cb.message.chat.id
    else:
        chat_id = user_id
    return user_id, chat_id
