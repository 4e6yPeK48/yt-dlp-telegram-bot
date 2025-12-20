import asyncio
import os
import shutil
from contextlib import suppress
from typing import Any, Dict, List, Optional, Tuple

from aiogram import Bot
from aiogram.types import FSInputFile, CallbackQuery, Message

from services import telethon_client
from services.telethon_client import request_alternate_delivery_and_send, get_username
from services.ytdlp import extract_basic_info
from bot.dispatcher import logger
from config import TG_MAX_UPLOAD_BYTES, TELETHON_FALLBACK_ENABLED
from storage.state import get_user_cookies_path
from utils.text import make_caption, format_duration_hms, make_multiline_caption
from utils.validators import is_youtube_url


async def _send_via_bot_or_fallback(bot, chat_id, media_path, thumb_path, caption, method, media_arg, extra=None):
    """Пробует отправить файл через Bot API, при необходимости выполняет fallback через Telethon.

    Сначала пытается отправить файлы стандартным методом бота. Если файл превышает
    лимит Telegram или Bot API возвращает ошибку и включён TELETHON_FALLBACK_ENABLED,
    инициирует процедуру альтернативной доставки через авторизованный пользовательский аккаунт.

    Args:
        bot: Объект aiogram.Bot.
        chat_id: Идентификатор чата/пользователя.
        media_path: Путь к медиафайлу.
        thumb_path: Путь к миниатюре (или None).
        caption: Подпись к файлу.
        method: Имя метода Bot API (например, "send_audio"/"send_video").
        media_arg: Аргумент для файла ("audio" или "video").
        extra: Дополнительные параметры для метода.

    Returns:
        bool: True если отправлено успешно (через бота или Telethon), False в противном случае.
    """
    try:
        size = os.path.getsize(media_path)
    except Exception:
        size = 0

    if size and size > TG_MAX_UPLOAD_BYTES and TELETHON_FALLBACK_ENABLED and telethon_client.get_client():
        return await request_alternate_delivery_and_send(
            bot, chat_id, media_path,
            caption=caption,
            thumb=thumb_path,
            supports_streaming=(media_arg == "video"),
        )

    kwargs = {
        "chat_id": chat_id,
        "caption": caption,
        "parse_mode": None,
        media_arg: FSInputFile(media_path),
    }
    if thumb_path and os.path.exists(thumb_path):
        kwargs["thumbnail"] = FSInputFile(thumb_path)
    if extra:
        kwargs.update(extra)

    try:
        await getattr(bot, method)(**kwargs)
        return True
    except Exception as bot_exc:
        logger.exception("Ошибка отправки через Bot API: %s", bot_exc)
        if TELETHON_FALLBACK_ENABLED and telethon_client.get_client():
            return await request_alternate_delivery_and_send(
                bot, chat_id, media_path,
                caption=caption,
                thumb=thumb_path,
                supports_streaming=(media_arg == "video"),
            )
        return False


async def send_media_files(
    bot: Bot,
    chat_id: int,
    items: List[Tuple[str, Optional[str]]],
    method: str,
    media_arg: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Отправляет файлы по одному.

    Args:
        bot (Bot): Экземпляр бота.
        chat_id (int): ID чата.
        items (List[Tuple[str, Optional[str]]]): Список медиа.
        method (str): Метод Telegram API.
        media_arg (str): Аргумент ('audio'|'video').
        extra (Optional[Dict[str, Any]]): Дополнительные параметры.
    """
    for media_path, thumb_path in items:
        try:
            title = os.path.splitext(os.path.basename(media_path))[0]
            caption = make_caption(title)

            sent = await _send_via_bot_or_fallback(
                bot, chat_id, media_path, thumb_path, caption, method, media_arg, extra
            )
            if not sent:
                try:
                    await bot.send_message(chat_id,
                                           f"⚠️ Нельзя отправить файл '{title}'. Возможно, он слишком большой или произошла ошибка.")
                except Exception:
                    pass
                continue
        finally:
            if media_path:
                with suppress(Exception):
                    await asyncio.to_thread(os.remove, media_path)
            if thumb_path:
                with suppress(Exception):
                    await asyncio.to_thread(os.remove, thumb_path)
            await asyncio.sleep(0.3)

    parents = {os.path.dirname(p) for p, _ in items}
    for d in parents:
        base = os.path.basename(d)
        if base.startswith("out_"):
            with suppress(Exception):
                await asyncio.to_thread(shutil.rmtree, d, True)


async def send_audio_files(
    bot: Bot, chat_id: int, items: List[Tuple[str, Optional[str]]]
) -> None:
    """Отправляет аудиофайлы.

    Args:
        bot (Bot): Экземпляр бота.
        chat_id (int): ID чата.
        items (List[Tuple[str, Optional[str]]]): Список медиа.
    """
    await send_media_files(bot, chat_id, items, method="send_audio", media_arg="audio")


async def send_video_files(
    bot: Bot, chat_id: int, items: List[Tuple[str, Optional[str]]]
) -> None:
    """Отправляет видеофайлы.

    Args:
        bot (Bot): Экземпляр бота.
        chat_id (int): ID чата.
        items (List[Tuple[str, Optional[str]]]): Список медиа.
    """
    await send_media_files(
        bot,
        chat_id,
        items,
        method="send_video",
        media_arg="video",
        extra={"supports_streaming": True},
    )


async def send_by_mode(
    bot: Bot, chat_id: int, mode: str, items: List[Tuple[str, Optional[str]]]
) -> None:
    """Выбирает способ отправки по режиму.

    Args:
        bot (Bot): Экземпляр бота.
        chat_id (int): ID чата.
        mode (str): Режим.
        items (List[Tuple[str, Optional[str]]]): Медиа.
    """
    if mode == "audio":
        await send_audio_files(bot, chat_id, items)
    else:
        await send_video_files(bot, chat_id, items)


async def try_cb_answer(cb: CallbackQuery, text: Optional[str] = None) -> None:
    """Безопасно отправляет ответ на callback.

    Args:
        cb (CallbackQuery): Callback-запрос.
        text (Optional[str]): Текст уведомления.
    """
    with suppress(Exception):
        await cb.answer(text)


def get_cb_chat_id(cb: CallbackQuery) -> Optional[int]:
    """Получает chat_id из CallbackQuery.

    Args:
        cb (CallbackQuery): Объект запроса.

    Returns:
        Optional[int]: Идентификатор чата или None.
    """
    msg_obj = cb.message
    if msg_obj is not None and isinstance(msg_obj, Message):
        return msg_obj.chat.id
    if cb.from_user is not None:
        return cb.from_user.id
    return None


async def send_info_card(
    bot: Bot,
    chat_id: int,
    url: str,
    user_id: int,
    reply_markup: Optional[Any] = None,
) -> None:
    """Отправляет карточку найденного файла."""
    caption_fallback = "🎧 Файл найден:\n\nВыберите, что скачать для этой ссылки:"
    try:
        logger.info(
            "Показываю карточку информации (user=%s, url=%s)", str(user_id), url[:200]
        )
        info = await extract_basic_info(
            url, cookies_path=get_user_cookies_path(user_id)
        )
        title = str(info.get("title") or "Без названия")
        dur_s = info.get("duration")
        dur_str = format_duration_hms(dur_s)
        channel = str(info.get("channel") or "")
        show_channel = is_youtube_url(url) and bool(channel)
        parts = [
            "🎧 Файл найден:",
            "",
            f"Название: {title}",
        ]
        if show_channel:
            parts.append(f"Канал: {channel}")
        parts.append(f"Длительность: {dur_str}")
        parts.append("")
        parts.append("Выберите, что скачать для этой ссылки:")
        caption = make_multiline_caption("\n".join(parts))
        thumb_url = info.get("thumbnail")
        if isinstance(thumb_url, str) and thumb_url.strip():
            with suppress(Exception):
                await bot.send_photo(
                    chat_id=chat_id,
                    photo=thumb_url.strip(),
                    caption=caption,
                    parse_mode=None,
                    reply_markup=reply_markup,
                )
                return
        await bot.send_message(
            chat_id,
            caption,
            parse_mode=None,
            reply_markup=reply_markup,
        )
    except Exception:
        await bot.send_message(
            chat_id,
            caption_fallback,
            parse_mode=None,
            reply_markup=reply_markup,
        )
