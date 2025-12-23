import asyncio
import logging
import sys

from config import SERVER_COOKIES_SOURCES
from services.server_cookies import refresh_server_cookies_once
from utils.log_helpers import log_info, log_exception
from utils.logging import setup_logging

try:
    from bot.dispatcher import logger
except ImportError:
    setup_logging()
    logger = logging.getLogger("bot")

async def _main():
    if not SERVER_COOKIES_SOURCES:
        log_info(logger, "Не заданы SERVER_COOKIES_SOURCES; пропускаю.")
        return 0
    try:
        results = await refresh_server_cookies_once()
        success = all(results.get(k, False) for k in SERVER_COOKIES_SOURCES.keys())
        log_info(logger, "fetch_server_cookies: результаты", extra={"results": results})
        return 0 if success else 2
    except Exception:
        log_exception(logger, "fetch_server_cookies: неожиданная ошибка")
        return 3

if __name__ == "__main__":
    exit_code = asyncio.run(_main())
    sys.exit(exit_code)

# Небольшой CLI-скрипт для однократной загрузки cookies (подходит для cron/systemd timers).
