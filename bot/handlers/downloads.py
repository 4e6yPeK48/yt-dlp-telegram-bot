# File: final_project/bot/handlers/downloads.py
from yt_dlp.utils import DownloadError  # type: ignore[import-untyped]

from bot.dispatcher import logger, download_sem
from services.ytdlp import download_media_to_temp, extract_basic_info, FileTooLargeError
from services.telegram import send_by_mode
from storage.state import (
    end_user_download,
    remember_cookie_request,
    add_download_history,
)


async def perform_download(
    bot,
    chat_id,
    user_id,
    url,
    mode,
    lock,
    cookies_path=None,
    on_cookies_required=None,
    on_nothing=None,
    on_error=None,
):
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

    Returns:
        None
    """
    try:
        async with download_sem:
            try:
                info = await extract_basic_info(url, cookies_path=cookies_path)
                title = info.get("title")
                duration = info.get("duration")
            except Exception:
                title = None
                duration = None

            try:
                files = await download_media_to_temp(url, mode=mode, cookies_path=cookies_path)
            except FileTooLargeError:
                logger.warning("Загрузка прервана: файл превышает максимально допустимый размер (user=%s, mode=%s)",
                               str(user_id), mode)
                try:
                    await bot.send_message(chat_id,
                                           "❌ Файл слишком большой (превышает лимит сервера, 2 ГБ). Нельзя доставить через бота.")
                    from services import telethon_client
                    if telethon_client.get_client():
                        username = telethon_client.get_username() or "alternate account"
                        await bot.send_message(chat_id,
                                               f"⚠️ Можно попытаться доставить через альтернативный аккаунт @{username}. Отправьте любое сообщение этому аккаунту и попробуйте снова.")
                except Exception:
                    pass
                return

        if not files:
            logger.info(
                "Загрузка завершена: нечего отправлять (user=%s, mode=%s)",
                str(user_id),
                mode,
            )
            if on_nothing:
                await on_nothing()
            else:
                await bot.send_message(
                    chat_id,
                    "😕 Нечего отправлять. Возможно, превышен лимит длительности (30 минут).",
                )
            return
        logger.info(
            "Загрузка завершена: файлов к отправке %d (user=%s, mode=%s)",
            len(files),
            str(user_id),
            mode,
        )
        await send_by_mode(bot, chat_id, mode, files)
        logger.info(
            "Отправка завершена: отправлено %d файлов (user=%s, mode=%s)",
            len(files),
            str(user_id),
            mode,
        )

        try:
            add_download_history(
                user_id,
                {"url": url, "mode": mode, "title": title or "", "duration": duration},
            )
        except Exception:
            logger.warning(
                "Не удалось записать историю загрузки для user=%s", str(user_id)
            )

    except DownloadError:
        logger.info("Загрузка требует cookies (user=%s, mode=%s)", str(user_id), mode)
        if on_cookies_required:
            await on_cookies_required()
        else:
            remember_cookie_request(user_id, kind="download", url=url, mode=mode)
            await bot.send_message(
                chat_id,
                "🍪 Источник требует cookies или произошла ошибка.\nПришлите файл cookies.txt для повтора попытки.",
            )
    except Exception:
        logger.info("Ошибка при загрузке (user=%s, mode=%s)", str(user_id), mode)
        if on_error:
            await on_error()
        else:
            await bot.send_message(
                chat_id, "❌ Произошла ошибка при загрузке. Попробуйте позже."
            )
    finally:
        await end_user_download(lock)
