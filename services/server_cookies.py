import asyncio
import logging
import os
import tempfile
from typing import Dict

import aiohttp

from config import (
    SERVER_COOKIES_DIR,
    SERVER_COOKIES_SOURCES,
    SERVER_COOKIES_REFRESH_INTERVAL_SEC,
    COOKIES_MAX_BYTES,
)
from utils.log_helpers import log_info, log_warning, log_exception
from bot.dispatcher import logger
from contextlib import suppress

async def _download_to_path(session: aiohttp.ClientSession, url: str, dest_path: str, max_bytes: int) -> bool:
    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(dest_path), prefix=".tmp_cookie_")
    os.close(tmp_fd)
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
            if resp.status != 200:
                log_warning(logger, "server_cookies: неожиданный статус", url=url, extra={"status": resp.status})
                return False
            size = 0
            with open(tmp_path, "wb") as f:
                async for chunk in resp.content.iter_chunked(1024 * 16):
                    if not chunk:
                        break
                    f.write(chunk)
                    size += len(chunk)
                    if max_bytes and size > max_bytes:
                        log_warning("server_cookies: файл превышает максимальный размер", url=url, extra={"max_bytes": max_bytes})
                        return False
            if size == 0:
                log_warning(logger, "server_cookies: загружен пустой файл", url=url)
                return False
        os.replace(tmp_path, dest_path)
        try:
            os.chmod(dest_path, 0o600)
        except Exception:
            pass
        log_info(logger, "server_cookies: сохранено", extra={"size": size, "path": dest_path})
        return True
    except Exception as e:
        log_exception(logger, "server_cookies: ошибка при загрузке", url=url, extra={"err": str(e)})
        with suppress(Exception):
            os.remove(tmp_path)
        return False


async def refresh_server_cookies_once(sources: Dict[str, str] = None) -> Dict[str, bool]:
    out: Dict[str, bool] = {}
    sources = sources or SERVER_COOKIES_SOURCES or {}
    if not sources:
        log_info(logger, "server_cookies: не заданы источники для загрузки")
        return out
    os.makedirs(SERVER_COOKIES_DIR, exist_ok=True)
    async with aiohttp.ClientSession() as session:
        for fname, url in sources.items():
            if not url:
                out[fname] = False
                continue
            dest = os.path.join(SERVER_COOKIES_DIR, fname)
            try:
                ok = await _download_to_path(session, url, dest, COOKIES_MAX_BYTES)
                out[fname] = bool(ok)
            except Exception:
                log_exception(logger , "server_cookies: неожиданная ошибка", extra={"filename": fname})
                out[fname] = False
    return out


async def start_periodic_refresher(interval: int = None, stop_event: asyncio.Event = None) -> None:
    interval = interval or SERVER_COOKIES_REFRESH_INTERVAL_SEC
    stop_event = stop_event or asyncio.Event()
    log_info(logger, "server_cookies: рефрешер запущен", extra={"interval_sec": interval})
    try:
        while not stop_event.is_set():
            try:
                await refresh_server_cookies_once()
            except Exception:
                log_exception(logger , "server_cookies: refresh loop error")
            await asyncio.wait([stop_event.wait()], timeout=interval)
    except asyncio.CancelledError:
        log_exception(logger , "server_cookies: рефрешер отменён")
    finally:
        log_info(logger, "server_cookies: рефрешер остановлен")
