import asyncio
import math
import os
import secrets
from typing import Any, Dict, List, Optional, Tuple

from ..config import COOKIES_DIR, PAGE_SIZE


USER_SEARCHES: Dict[int, Dict[str, Any]] = {}
AWAITING_COOKIES: Dict[int, Dict[str, Any]] = {}
USER_SETTINGS: Dict[int, Dict[str, str]] = {}
USER_LOCKS: Dict[int, asyncio.Lock] = {}
PENDING_DOWNLOADS: Dict[str, Dict[str, Any]] = {}


def get_user_mode(user_id: int) -> str:
    """Возвращает текущий режим пользователя.

    Args:
        user_id (int): Идентификатор пользователя Telegram.

    Returns:
        str: Один из: 'auto', 'audio', 'video', 'video_nosound'.
    """
    return (USER_SETTINGS.get(user_id) or {}).get("mode", "auto")


def set_user_mode(user_id: int, mode: str) -> None:
    """Сохраняет выбранный режим пользователя.

    Args:
        user_id (int): Идентификатор пользователя.
        mode (str): Режим ('auto'|'audio'|'video'|'video_nosound').
    """
    if user_id not in USER_SETTINGS:
        USER_SETTINGS[user_id] = {}
    USER_SETTINGS[user_id]["mode"] = mode


def get_user_lock(user_id: int) -> asyncio.Lock:
    """Получает или создаёт Lock для пользователя.

    Args:
        user_id (int): Идентификатор пользователя.

    Returns:
        asyncio.Lock: Lock пользователя.
    """
    if user_id not in USER_LOCKS:
        USER_LOCKS[user_id] = asyncio.Lock()
    return USER_LOCKS[user_id]


async def begin_user_download(user_id: int) -> Optional[asyncio.Lock]:
    """Пытается захватить пользовательский Lock перед загрузкой.

    Args:
        user_id (int): Идентификатор пользователя.

    Returns:
        Optional[asyncio.Lock]: Захваченный Lock или None если занят.
    """
    lock = get_user_lock(user_id)
    if lock.locked():
        return None
    return lock


async def end_user_download(lock: Optional[asyncio.Lock]) -> None:
    """Освобождает захваченный Lock.

    Args:
        lock (Optional[asyncio.Lock]): Объект блокировки.
    """
    if lock is not None and lock.locked():
        lock.release()


def slice_page(
    items: List[Any],
    page: int,
    page_size: int = PAGE_SIZE,
) -> Tuple[List[Any], int]:
    """Возвращает элементы указанной страницы и общее число страниц.

    Args:
        items (List[Any]): Полный список элементов.
        page (int): Номер страницы (0-индексация).
        page_size (int): Размер страницы.

    Returns:
        Tuple[List[Any], int]: Элементы текущей страницы и всего страниц.
    """
    pages = max(1, math.ceil(len(items) / page_size))
    page = max(0, min(page, pages - 1))
    start = page * page_size
    end = start + page_size
    return items[start:end], pages


def remember_cookie_request(user_id: int, kind: str, url: Optional[str] = None, mode: Optional[str] = None) -> None:
    """Сохраняет ожидание cookies.

    Args:
        user_id (int): Пользователь.
        kind (str): Тип ('download'|'search').
        url (Optional[str]): URL для повтора.
        mode (Optional[str]): Режим ('audio'|'video'|'video_nosound'|'auto').
    """
    payload: Dict[str, Any] = {"kind": kind, "asked": True}
    if url:
        payload["url"] = url
    if mode:
        payload["mode"] = mode
    AWAITING_COOKIES[user_id] = payload


def remember_search_cookie_request(
    user_id: int,
    query: str,
) -> None:
    """Сохраняет ожидание cookies для поиска.

    Args:
        user_id (int): Пользователь.
        query (str): Поисковый запрос.
    """
    AWAITING_COOKIES[user_id] = {"kind": "search", "query": query, "asked": True}


def get_user_cookies_path(user_id: int) -> str:
    """Возвращает путь к cookies.txt пользователя.

    Args:
        user_id (int): Идентификатор пользователя.

    Returns:
        str: Путь к cookies.txt.
    """
    return os.path.join(COOKIES_DIR, f"{user_id}_cookies.txt")


def make_dl_token() -> str:
    """Генерирует уникальный токен для отложенного скачивания.

    Returns:
        str: Токен (10 символов [A-Za-z0-9]).
    """
    t = ""
    for _ in range(5):
        t = secrets.token_urlsafe(6).replace("-", "").replace("_", "")[:10]
        if t not in PENDING_DOWNLOADS:
            break
    return t


def save_pending_url(user_id: int, url: str) -> str:
    """Сохраняет URL для последующего выбора режима отправки.

    Args:
        user_id (int): Идентификатор пользователя.
        url (str): Сохранённый URL.

    Returns:
        str: Токен сохранения.
    """
    token = make_dl_token()
    PENDING_DOWNLOADS[token] = {
        "user_id": user_id,
        "url": url,
    }
    return token
