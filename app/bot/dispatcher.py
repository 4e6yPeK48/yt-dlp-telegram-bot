import asyncio
import logging

from aiogram import Router, Dispatcher

from app.config import CONCURRENT_DOWNLOADS

router: Router = Router()
dp: Dispatcher = Dispatcher()
dp.include_router(router)
download_sem: asyncio.Semaphore = asyncio.Semaphore(CONCURRENT_DOWNLOADS)
telethon_sem: asyncio.Semaphore = asyncio.Semaphore(CONCURRENT_DOWNLOADS)
logger: logging.Logger = logging.getLogger("bot")
