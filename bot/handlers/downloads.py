from asyncio import CancelledError

from yt_dlp.utils import DownloadError

from bot.dispatcher import logger
from utils.log_helpers import log_info, log_warning, log_exception
from services.ytdlp import download_media_to_temp, extract_basic_info, FileTooLargeError
from services.telegram import send_by_mode
from storage.state import (
    end_user_download,
    remember_cookie_request,
    add_download_history,
)
from contextlib import suppress

from typing import Optional, Callable, Awaitable, Any


async def perform_download(
    bot: Any,
    chat_id: int,
    user_id: int,
    url: str,
    mode: str,
    lock: Any,
    cookies_path: Optional[str] = None,
    on_cookies_required: Optional[Callable[[], Awaitable[None]]] = None,
    on_nothing: Optional[Callable[[], Awaitable[None]]] = None,
    on_error: Optional[Callable[[], Awaitable[None]]] = None,
    status_message: Optional[Any] = None,
) -> None:
    """Универсальная функция для скачивания и отправки медиафайлов.

    Выполняет следующие шаги:
        1. Пытается извлечь базовую информацию о ресурсе.
        2. Скачивает медиа с помощью yt-dlp во временную папку.
        3. Отправляет полученные файлы через Bot API или делает fallback через Telethon.
        4. Записывает историю загрузки и очищает временные файлы.

    Args:
        bot: Экземпляр aiogram.Bot.
        chat_id: ID чата для отправки сообщений.
        user_id: ID пользователя, инициировавшего загрузку.
        url: Ссылка на источник.
        mode: Режим скачивания ('audio'|'video'|'video_nosound'|'auto').
        lock: Блокировка/семафор для отслеживания загрузки.
        cookies_path: Необязательный путь к cookies.txt.
        on_cookies_required: Корутин, вызываемый, если требуются cookies.
        on_nothing: Корутин, вызываемый, если нечего отправлять.
        on_error: Корутин, вызываемый при общей ошибке.
        status_message: Сообщение статуса для удаления по завершении.

    Returns:
        None
    """
    try:
        try:
            info = await extract_basic_info(url, cookies_path=cookies_path)
            title = info.get("title")
            duration = info.get("duration")
        except Exception:
            title = None
            duration = None

        try:
            files = await download_media_to_temp(
                url, mode=mode, cookies_path=cookies_path
            )
        except FileTooLargeError:
            log_warning(
                logger,
                "Загрузка прервана: файл превышает максимально допустимый размер (превышает лимит сервера, 2 ГБ).",
                user_id=user_id,
                mode=mode,
                extra={"reason": "too_large"},
            )
            try:
                await bot.send_message(
                    chat_id,
                    "❌ Файл слишком большой (превышает лимит сервера, 2 ГБ). Нельзя доставить через бота.",
                )
                from services import telethon_client

                if telethon_client.get_client():
                    username = telethon_client.get_username() or "alternate account"
                    await bot.send_message(
                        chat_id,
                        f"⚠️ Можно попытаться доставить через альтернативный аккаунт @{username}. Отправьте любое сообщение этому аккаунту и попробуйте снова.",
                    )
            except Exception as e:
                log_exception(
                    logger,
                    "Не удалось уведомить пользователя о слишком большом файле",
                    user_id=user_id,
                    chat_id=chat_id,
                    url=url,
                    mode=mode,
                    extra={"err": str(e)},
                )
            return

        if not files:
            log_info(
                logger,
                "Загрузка завершена: нечего отправлять",
                user_id=user_id,
                mode=mode,
            )
            if on_nothing:
                await on_nothing()
            else:
                await bot.send_message(
                    chat_id,
                    "😕 Нечего отправлять. Возможно, превышен лимит длительности (30 минут).",
                )
            return
        log_info(
            logger,
            f"Загрузка завершена: файлов к отправке {len(files)}",
            user_id=user_id,
            mode=mode,
        )

        await send_by_mode(bot, chat_id, mode, files)
        log_info(
            logger,
            f"Отправка завершена: отправлено {len(files)} файлов",
            user_id=user_id,
            mode=mode,
        )

        if status_message is not None:
            with suppress(Exception):
                mid = getattr(status_message, "message_id", None)
                if mid:
                    await bot.delete_message(chat_id, mid)

        try:
            add_download_history(
                user_id,
                {"url": url, "mode": mode, "title": title or "", "duration": duration},
            )
        except Exception:
            log_warning(
                logger,
                "Не удалось записать историю загрузки",
                user_id=user_id,
            )

    except DownloadError:
        log_info(
            logger,
            "Загрузка требует cookies",
            user_id=user_id,
            mode=mode,
        )
        if on_cookies_required:
            await on_cookies_required()
        else:
            remember_cookie_request(user_id, kind="download", url=url, mode=mode)
            await bot.send_message(
                chat_id,
                "🍪 Источник требует cookies или произошла ошибка.\nПришлите файл cookies.txt для повтора попытки.",
            )
    except CancelledError:
        log_info(
            logger,
            "Загрузка отменена",
            user_id=user_id,
            mode=mode,
            url=url,
        )
        raise
    except Exception as e:
        log_exception(
            logger,
            "Ошибка при загрузке",
            user_id=user_id,
            mode=mode,
            url=url,
            extra={"err": str(e)},
        )
        if on_error:
            await on_error()
        else:
            try:
                await bot.send_message(
                    chat_id, "❌ Произошла ошибка при загрузке. Попробуйте позже."
                )
            except Exception as e2:
                log_exception(
                    logger,
                    "Не удалось уведомить пользователя о внутренней ошибке загрузки",
                    user_id=user_id,
                    chat_id=chat_id,
                    extra={"err": str(e2)},
                )
    finally:
        await end_user_download(lock)
