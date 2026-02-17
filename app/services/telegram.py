import asyncio
import os
import shutil
from contextlib import suppress
from typing import Any, Dict, List, Optional, Tuple, Callable, Awaitable

from aiogram import Bot
from aiogram.types import FSInputFile

from app.services import telethon_client
from app.services.telethon_client import request_alternate_delivery_and_send
from app.services.ytdlp import extract_basic_info
from app.bot.dispatcher import logger
from app.utils.log_helpers import log_info, log_exception
from app.config import TG_MAX_UPLOAD_BYTES, TELETHON_FALLBACK_ENABLED


class TelegramSender:
    """
    Класс для отправки медиафайлов через Bot API с возможностью fallback через Telethon.
    """

    def __init__(
        self,
        tg_max_bytes: int = TG_MAX_UPLOAD_BYTES,
        telethon_enabled: bool = TELETHON_FALLBACK_ENABLED,
    ):
        self.tg_max_bytes = tg_max_bytes
        self.telethon_enabled = telethon_enabled

    async def _send_via_bot_or_fallback(
        self,
        bot: Bot,
        chat_id: int,
        media_path: str,
        thumb_path: Optional[str],
        caption: str,
        method: str,
        media_arg: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> bool:
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
            size = await asyncio.to_thread(os.path.getsize, media_path)  # mypy: ignore
        except Exception:
            size = 0

        if (
            size
            and self.tg_max_bytes
            and size > self.tg_max_bytes
            and self.telethon_enabled
            and telethon_client.get_client()
        ):
            return await request_alternate_delivery_and_send(
                bot,
                chat_id,
                media_path,
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
        if thumb_path and await asyncio.to_thread(
            os.path.exists, thumb_path
        ):  # mypy: ignore
            kwargs["thumbnail"] = FSInputFile(thumb_path)
        if extra:
            kwargs.update(extra)

        try:
            await getattr(bot, method)(**kwargs)
            return True
        except Exception as bot_exc:
            log_exception(
                logger,
                "Ошибка отправки через Bot API",
                chat_id=chat_id,
                extra={"media": media_path, "method": method, "err": str(bot_exc)},
            )
            if self.telethon_enabled and telethon_client.get_client():
                return await request_alternate_delivery_and_send(
                    bot,
                    chat_id,
                    media_path,
                    caption=caption,
                    thumb=thumb_path,
                    supports_streaming=(media_arg == "video"),
                )
            return False

    async def send_media_files(
        self,
        bot: Bot,
        chat_id: int,
        items: List[Tuple[str, Optional[str]]],
        method: str,
        media_arg: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        for media_path, thumb_path in items:
            try:
                title = os.path.splitext(os.path.basename(media_path))[0]
                caption = title
                sent = await self._send_via_bot_or_fallback(
                    bot,
                    chat_id,
                    media_path,
                    thumb_path,
                    caption,
                    method,
                    media_arg,
                    extra,
                )
                if not sent:
                    try:
                        await bot.send_message(
                            chat_id,
                            f"⚠️ Нельзя отправить файл '{title}'. Возможно, он слишком большой или произошла ошибка.",
                        )
                    except Exception as e:
                        log_exception(
                            logger,
                            "Не удалось уведомить пользователя о проблеме отправки файла",
                            chat_id=chat_id,
                            extra={"title": title, "media": media_path, "err": str(e)},
                        )
                    continue
            finally:
                if media_path:
                    with suppress(Exception):
                        await asyncio.to_thread(os.remove, media_path)  # mypy: ignore
                if thumb_path:
                    with suppress(Exception):
                        await asyncio.to_thread(os.remove, thumb_path)  # mypy: ignore
                await asyncio.sleep(0.3)

        parents = {os.path.dirname(p) for p, _ in items}
        for d in parents:
            base = os.path.basename(d)
            if base.startswith("out_"):
                with suppress(Exception):
                    await asyncio.to_thread(shutil.rmtree, d, True)  # mypy: ignore

    async def send_audio_files(
        self, bot: Bot, chat_id: int, items: List[Tuple[str, Optional[str]]]
    ) -> None:
        await self.send_media_files(
            bot, chat_id, items, method="send_audio", media_arg="audio"
        )

    async def send_video_files(
        self, bot: Bot, chat_id: int, items: List[Tuple[str, Optional[str]]]
    ) -> None:
        await self.send_media_files(
            bot,
            chat_id,
            items,
            method="send_video",
            media_arg="video",
            extra={"supports_streaming": True},
        )

    async def send_by_mode(
        self, bot: Bot, chat_id: int, mode: str, items: List[Tuple[str, Optional[str]]]
    ) -> None:
        if mode == "audio":
            await self.send_audio_files(bot, chat_id, items)
        else:
            await self.send_video_files(bot, chat_id, items)

    async def send_info_card(
        self,
        bot: Bot,
        chat_id: int,
        url: str,
        user_id: int,
        reply_markup: Optional[Any] = None,
        extract_basic_info_fn: Optional[
            Callable[[str], Awaitable[Dict[str, Any]]]
        ] = None,
    ) -> None:
        extract_fn = extract_basic_info_fn or extract_basic_info
        caption_fallback = "🎧 Файл найден:\n\nВыберите, что скачать для этой ссылки:"
        try:
            log_info(logger, "Показываю карточку информации", user_id=user_id, url=url)
            info = await extract_fn(url)
            title = str(info.get("title") or "Без названия")
            dur_s = info.get("duration")
            from app.utils.text import make_multiline_caption, format_duration_hms

            dur_str = format_duration_hms(dur_s)
            channel = str(info.get("channel") or "")
            from app.utils.validators import is_youtube_url

            show_channel = is_youtube_url(url) and bool(channel)
            parts = ["🎧 Файл найден:", "", f"Название: {title}"]
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
                chat_id, caption, parse_mode=None, reply_markup=reply_markup
            )
        except Exception as e:
            log_exception(
                logger,
                "Ошибка при отправке карточки информации",
                chat_id=chat_id,
                user_id=user_id,
                url=url,
                extra={"err": str(e)},
            )
            await bot.send_message(
                chat_id, caption_fallback, parse_mode=None, reply_markup=reply_markup
            )


_sender = TelegramSender()


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
    await _sender.send_media_files(bot, chat_id, items, method, media_arg, extra)


async def send_audio_files(
    bot: Bot, chat_id: int, items: List[Tuple[str, Optional[str]]]
) -> None:
    """Отправляет аудиофайлы.

    Args:
        bot (Bot): Экземпляр бота.
        chat_id (int): ID чата.
        items (List[Tuple[str, Optional[str]]]): Список медиа.
    """
    await _sender.send_audio_files(bot, chat_id, items)


async def send_video_files(
    bot: Bot, chat_id: int, items: List[Tuple[str, Optional[str]]]
) -> None:
    """Отправляет видеофайлы.

    Args:
        bot (Bot): Экземпляр бота.
        chat_id (int): ID чата.
        items (List[Tuple[str, Optional[str]]]): Список медиа.
    """
    await _sender.send_video_files(bot, chat_id, items)


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
    await _sender.send_by_mode(bot, chat_id, mode, items)


async def send_info_card(
    bot: Bot, chat_id: int, url: str, user_id: int, reply_markup: Optional[Any] = None
) -> None:
    """Отправляет карточку найденного файла.

    Args:
        bot (Bot): Экземпляр бота.
        chat_id (int): ID чата.
        url (str): Ссылка на источник.
        user_id (int): ID пользователя.
        reply_markup (Optional[Any]): Клавиатура для ответа.
    Returns:
        None
    """
    await _sender.send_info_card(bot, chat_id, url, user_id, reply_markup=reply_markup)
