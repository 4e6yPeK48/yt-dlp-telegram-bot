import asyncio

from aiogram import Bot
from aiogram.client.default import DefaultBotProperties
from yt_dlp import YoutubeDL  # type: ignore[import-untyped]
from yt_dlp.utils import DownloadError  # type: ignore[import-untyped]

from bot.dispatcher import dp, logger
from config import (
    BOT_TOKEN,
)
from utils.logging import setup_logging


async def main() -> None:
    """Точка входа приложения: настройка логирования и старт поллинга.

    Raises:
        RuntimeError: Если отсутствует BOT_TOKEN.
    """
    setup_logging()
    if not BOT_TOKEN:
        raise RuntimeError("Не задана переменная окружения BOT_TOKEN")
    bot = Bot(
        BOT_TOKEN,
        default=DefaultBotProperties(parse_mode="HTML"),
    )
    logger.info("Старт поллинга")
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
