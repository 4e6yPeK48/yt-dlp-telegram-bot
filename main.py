import asyncio

from aiogram import Bot
from aiogram.client.default import DefaultBotProperties
from yt_dlp import YoutubeDL  # type: ignore[import-untyped]
from yt_dlp.utils import DownloadError  # type: ignore[import-untyped]

from bot.dispatcher import dp, logger
from config import (
    BOT_TOKEN, TELETHON_FALLBACK_ENABLED,
)
from services import telethon_client
from utils.logging import setup_logging
from bot.handlers import commands, messages, callbacks, downloads  # noqa: F401


async def main() -> None:
    """Точка входа приложения: настройка логирования и старт поллинга.

    Raises:
        RuntimeError: Если отсутствует BOT_TOKEN.
    """
    setup_logging()
    if not BOT_TOKEN:
        raise RuntimeError("Не задана переменная окружения BOT_TOKEN")

    try:
        await telethon_client.ensure_client_started()
    except Exception as e:
        if TELETHON_FALLBACK_ENABLED:
            logger.error("Включён Telethon-fallback, но не удалось запустить клиент: %s", e)
            raise
        else:
            logger.warning("Продолжаю без Telethon-fallback: %s", e)

    bot = Bot(
        BOT_TOKEN,
        default=DefaultBotProperties(parse_mode="HTML"),
    )
    logger.info("Старт поллинга")
    try:
        await dp.start_polling(bot)
    finally:
        try:
            await telethon_client.disconnect_client()
        except Exception:
            logger.exception("Ошибка при отключении Telethon-клиента при завершении")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
