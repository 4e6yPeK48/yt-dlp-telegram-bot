from contextlib import suppress
from yt_dlp.utils import DownloadError  # type: ignore[import-untyped]

from ...bot.dispatcher import logger
from ...services.ytdlp import download_media_to_temp
from ...services.telegram import send_by_mode
from ...storage.state import (
    end_user_download,
    remember_cookie_request,
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
    """
    Универсальная функция скачивания и отправки медиа.
    """
    try:
        files = await download_media_to_temp(url, mode=mode, cookies_path=cookies_path)
        if not files:
            logger.info("Загрузка завершена: нечего отправлять (user=%s, mode=%s)", str(user_id), mode)
            if on_nothing:
                await on_nothing()
            else:
                await bot.send_message(
                    chat_id,
                    "😕 Нечего отправлять. Возможно, превышен лимит длительности (30 минут).",
                )
            return
        logger.info("Загрузка завершена: файлов к отправке %d (user=%s, mode=%s)", len(files), str(user_id), mode)
        await send_by_mode(bot, chat_id, mode, files)
        logger.info("Отправка завершена: отправлено %d файлов (user=%s, mode=%s)", len(files), str(user_id), mode)
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
            await bot.send_message(chat_id, "❌ Произошла ошибка при загрузке. Попробуйте позже.")
    finally:
        await end_user_download(lock)

