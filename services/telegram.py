import asyncio
import os
import shutil
from contextlib import suppress
from typing import Any, Dict, List, Optional, Tuple

from aiogram import Bot
from aiogram.types import FSInputFile, CallbackQuery, Message

from services import telethon_client
from services.ytdlp import extract_basic_info
from bot.dispatcher import logger
from config import TG_MAX_UPLOAD_BYTES, TELETHON_FALLBACK_ENABLED
from storage.state import get_user_cookies_path
from utils.text import make_caption, format_duration_hms, make_multiline_caption
from utils.validators import is_youtube_url


async def _send_via_bot_or_fallback(bot, chat_id, media_path, thumb_path, caption, method, media_arg, extra=None):
    """Try bot send first, fallback to Telethon if allowed and necessary."""
    try:
        size = os.path.getsize(media_path)
    except Exception:
        size = 0

    # If size exceeds Bot API limit, attempt fallback
    if size and size > TG_MAX_UPLOAD_BYTES and TELETHON_FALLBACK_ENABLED and telethon_client.get_client():
        username = telethon_client.get_username() or "user account"
        try:
            await bot.send_message(chat_id, f"⚠️ File is large — will try alternate delivery. Please send any message to @{username} (the alternate account) within 120 seconds.")
        except Exception:
            logger.exception("Failed to notify user about fallback delivery.")

        # Wait for user incoming message to the telethon account as handshake
        got = await telethon_client.wait_for_user_message(chat_id, timeout=120)
        if not got:
            try:
                await bot.send_message(chat_id, "⌛ Timeout waiting for your message to the alternate account. Cannot deliver the large file.")
            except Exception:
                pass
            return False

        # Send via Telethon
        try:
            await telethon_client.send_file_via_user(chat_id, media_path, caption=caption, thumb=thumb_path, supports_streaming=(media_arg == "video"))
            try:
                await bot.send_message(chat_id, "✅ File delivered via alternate account.")
            except Exception:
                pass
            return True
        except Exception:
            logger.exception("Telethon fallback send failed.")
            try:
                await bot.send_message(chat_id, "❌ Alternate delivery failed (permissions or internal error). Ensure you started a chat with the alternate account and haven't blocked it.")
            except Exception:
                pass
            return False

    # Default: attempt to send via Bot API
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
        logger.exception("Bot API send failed: %s", bot_exc)
        # If file likely too large or other error and fallback available, try Telethon
        if TELETHON_FALLBACK_ENABLED and telethon_client.get_client():
            username = telethon_client.get_username() or "user account"
            try:
                await bot.send_message(chat_id, f"⚠️ Бот не смог отправить файл. Попытка альтернативной доставки. Пожалуйста, отправьте любое сообщение @{username} (альтернативному аккаунту) в течение 120 секунд.")
            except Exception:
                pass
            got = await telethon_client.wait_for_user_message(chat_id, timeout=120)
            if not got:
                try:
                    await bot.send_message(chat_id, "⌛ Таймаут ожидания вашего сообщения альтернативному аккаунту. Невозможно доставить файл.")
                except Exception:
                    pass
                return False
            try:
                await telethon_client.send_file_via_user(chat_id, media_path, caption=caption, thumb=thumb_path, supports_streaming=(media_arg == "video"))
                try:
                    await bot.send_message(chat_id, "✅ Файл доставлен через альтернативный аккаунт.")
                except Exception:
                    pass
                return True
            except Exception:
                logger.exception("Telethon fallback send failed after Bot API error.")
                try:
                    await bot.send_message(chat_id, "❌ Альтернативная доставка не удалась (возможно, проблемы с правами или внутренняя ошибка). Убедитесь, что вы начали чат с альтернативным аккаунтом и не заблокировали его.")
                except Exception:
                    pass
                return False
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
