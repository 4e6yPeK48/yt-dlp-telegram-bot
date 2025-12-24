import asyncio

from aiogram import Bot
from aiogram.client.default import DefaultBotProperties

from bot.dispatcher import dp, logger
from utils.log_helpers import log_info, log_warning, log_exception
from config import (
    BOT_TOKEN,
    TELETHON_FALLBACK_ENABLED,
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

    if TELETHON_FALLBACK_ENABLED:
        try:
            await telethon_client.ensure_client_started()
            telethon_ready = bool(telethon_client.get_client())
            if telethon_ready:
                log_info(logger, "Telethon-fallback успешно инициализирован")
            else:
                log_warning(logger, "Telethon-fallback включён, но клиент не инициализирован")
        except RuntimeError as e:
            log_warning(
                logger,
                "Telethon-fallback включён, но не настроен. Запустите `python scripts/telethon_login.py` локально, чтобы авторизовать или отключить резервный вариант. Продолжаю без Telethon.",
                extra={"err": str(e)},
            )
        except Exception as e:
            log_exception(
                logger,
                "Ошибка при инициализации Telethon-fallback. Продолжаю без Telethon.",
                extra={"err": str(e)},
            )
    else:
        log_warning(logger, "Продолжаю без Telethon-fallback")

    bot = Bot(
        BOT_TOKEN,
        default=DefaultBotProperties(parse_mode="HTML"),
    )
    log_info(logger, "Старт поллинга")
    try:
        await dp.start_polling(bot)
    finally:
        log_info(logger, "Остановка поллинга: завершаю работу бота")
        try:
            await telethon_client.disconnect_client()
        except Exception as e:
            log_exception(logger, "Ошибка при отключении Telethon-клиента при завершении работы", extra={"err": str(e)})
        log_info(logger, "Бот завершил работу")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
