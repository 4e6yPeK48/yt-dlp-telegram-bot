import asyncio
import os
import tempfile
from typing import Dict, Optional
from contextlib import suppress

import aiohttp

from app.config import (
    SERVER_COOKIES_DIR,
    SERVER_COOKIES_SOURCES,
    SERVER_COOKIES_REFRESH_INTERVAL_SEC,
    COOKIES_MAX_BYTES,
    SERVER_COOKIES_MAP,
)
from app.utils.log_helpers import log_info, log_warning, log_exception
from app.bot.dispatcher import logger


async def _download_to_path(
    session: aiohttp.ClientSession, url: str, dest_path: str, max_bytes: int
) -> bool:
    """
    Загружает файл по URL и сохраняет его во временный файл, затем перемещает в dest_path.
    Проверяет размер файла и логирует события.

    Args:
        session (aiohttp.ClientSession): Сессия для выполнения HTTP-запросов.
        url (str): URL для загрузки.
        dest_path (str): Путь для сохранения загруженного файла.
        max_bytes (int): Максимально допустимый размер файла в байтах.

    Returns:
        bool: True, если загрузка и сохранение прошли успешно, иначе False.
    """
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=os.path.dirname(dest_path), prefix=".tmp_cookie_"
    )
    os.close(tmp_fd)
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
            if resp.status != 200:
                log_warning(
                    logger,
                    "server_cookies: неожиданный статус",
                    url=url,
                    extra={"status": resp.status},
                )
                return False
            size = 0
            with open(tmp_path, "wb") as f:
                async for chunk in resp.content.iter_chunked(1024 * 16):
                    if not chunk:
                        break
                    f.write(chunk)
                    size += len(chunk)
                    if max_bytes and size > max_bytes:
                        log_warning(
                            logger,
                            "server_cookies: файл превышает максимальный размер",
                            url=url,
                            extra={"max_bytes": max_bytes},
                        )
                        with suppress(Exception):
                            os.remove(tmp_path)
                        return False
            if size == 0:
                log_warning(logger, "server_cookies: загружен пустой файл", url=url)
                with suppress(Exception):
                    os.remove(tmp_path)
                return False
        os.replace(tmp_path, dest_path)
        try:
            os.chmod(dest_path, 0o600)
        except Exception:
            pass
        log_info(
            logger, "server_cookies: сохранено", extra={"size": size, "path": dest_path}
        )
        return True
    except Exception as e:
        log_exception(
            logger,
            "server_cookies: ошибка при загрузке",
            url=url,
            extra={"err": str(e)},
        )
        with suppress(Exception):
            os.remove(tmp_path)
        return False


async def refresh_server_cookies_once(
    sources: Optional[Dict[str, str]] = None,
) -> Dict[str, bool]:
    """
    Однократное обновление серверных cookies из заданных источников.

    Args:
        sources (Dict[str, str], optional): Словарь источников для загрузки cookies
            в формате {ключ: URL}. Ключ может быть доменом или именем файла.
            Если не задан, используются SERVER_COOKIES_SOURCES из конфигурации.

    Returns:
        Dict[str, bool]: Словарь с результатами загрузки в формате {имя_файла: успешно (bool)}.
    """
    out: Dict[str, bool] = {}
    sources = sources or SERVER_COOKIES_SOURCES or {}
    if not sources:
        log_info(logger, "server_cookies: не заданы источники для загрузки")
        return out
    os.makedirs(SERVER_COOKIES_DIR, exist_ok=True)
    async with aiohttp.ClientSession() as session:
        handled_fnames = set()
        for domain_key, fname in SERVER_COOKIES_MAP.items():
            url = None
            if domain_key in sources:
                url = sources[domain_key]
            elif fname in sources:
                url = sources[fname]
            if not url:
                continue
            dest = os.path.join(SERVER_COOKIES_DIR, fname)
            try:
                ok = await _download_to_path(session, url, dest, COOKIES_MAX_BYTES)
                out[fname] = bool(ok)
            except Exception:
                log_exception(
                    logger,
                    "server_cookies: неожиданная ошибка",
                    extra={"filename": fname},
                )
                out[fname] = False
            handled_fnames.add(fname)

        for key, url in sources.items():
            if key in SERVER_COOKIES_MAP:
                continue
            mapped_fname = key
            if mapped_fname in handled_fnames:
                continue
            dest = os.path.join(SERVER_COOKIES_DIR, mapped_fname)
            try:
                ok = await _download_to_path(session, url, dest, COOKIES_MAX_BYTES)
                out[mapped_fname] = bool(ok)
            except Exception:
                log_exception(
                    logger,
                    "server_cookies: неожиданная ошибка",
                    extra={"filename": mapped_fname},
                )
                out[mapped_fname] = False

    return out


async def start_periodic_refresher(
    interval: Optional[int] = None, stop_event: Optional[asyncio.Event] = None
) -> None:
    """
    Запускает периодический рефрешер серверных cookies.

    Args:
        interval (int, optional): Интервал в секундах между обновлениями.
        stop_event (asyncio.Event, optional): Событие для остановки рефрешера.

    Returns:
        None
    """
    interval = interval or SERVER_COOKIES_REFRESH_INTERVAL_SEC
    stop_event = stop_event or asyncio.Event()
    log_info(
        logger, "server_cookies: рефрешер запущен", extra={"interval_sec": interval}
    )
    try:
        while not stop_event.is_set():
            try:
                await refresh_server_cookies_once()
            except Exception:
                log_exception(logger, "server_cookies: refresh loop error")
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval)
            except asyncio.TimeoutError:
                log_exception(logger, "server_cookies: рефрешер: таймаут ожидания")
    except asyncio.CancelledError:
        log_exception(logger, "server_cookies: рефрешер отменён")
    finally:
        log_info(logger, "server_cookies: рефрешер остановлен")
