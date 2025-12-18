import asyncio
import math
import os
import secrets
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import COOKIES_DIR, PAGE_SIZE


class StateStore:
    def __init__(self, cookies_dir: str = COOKIES_DIR) -> None:
        self._searches: Dict[int, Dict[str, Any]] = {}
        self._awaiting: Dict[int, Dict[str, Any]] = {}
        self._settings: Dict[int, Dict[str, str]] = {}
        self._locks: Dict[int, asyncio.Lock] = {}
        self._pending: Dict[str, Dict[str, Any]] = {}
        self._cookies_dir = Path(cookies_dir)
        self._cookies_dir.mkdir(parents=True, exist_ok=True)

    def get_user_mode(self, user_id: int) -> str:
        return (self._settings.get(user_id) or {}).get("mode", "auto")

    def set_user_mode(self, user_id: int, mode: str) -> None:
        if user_id not in self._settings:
            self._settings[user_id] = {}
        self._settings[user_id]["mode"] = mode

    def _get_lock(self, user_id: int) -> asyncio.Lock:
        if user_id not in self._locks:
            self._locks[user_id] = asyncio.Lock()
        return self._locks[user_id]

    async def begin_user_download(self, user_id: int) -> Optional[asyncio.Lock]:
        lock = self._get_lock(user_id)
        if lock.locked():
            return None
        await lock.acquire()
        return lock

    async def end_user_download(self, lock: Optional[asyncio.Lock]) -> None:
        if lock and lock.locked():
            lock.release()

    def get_searches(self, user_id: int) -> Optional[Dict[str, Any]]:
        return self._searches.get(user_id)

    def set_searches(self, user_id: int, payload: Dict[str, Any]) -> None:
        self._searches[user_id] = payload

    def pop_searches(self, user_id: int) -> Optional[Dict[str, Any]]:
        return self._searches.pop(user_id, None)


    def remember_cookie_request(self, user_id: int, kind: str, url: Optional[str] = None,
                                mode: Optional[str] = None) -> None:
        payload: Dict[str, Any] = {"kind": kind, "asked": True}
        if url:
            payload["url"] = url
        if mode:
            payload["mode"] = mode
        self._awaiting[user_id] = payload

    def remember_search_cookie_request(self, user_id: int, query: str) -> None:
        self._awaiting[user_id] = {"kind": "search", "query": query, "asked": True}

    def get_awaiting(self, user_id: int) -> Optional[Dict[str, Any]]:
        return self._awaiting.get(user_id)

    def pop_awaiting(self, user_id: int) -> Optional[Dict[str, Any]]:
        return self._awaiting.pop(user_id, None)

    def get_user_cookies_path(self, user_id: int) -> str:
        return str(self._cookies_dir / f"{user_id}_cookies.txt")

    def slice_page(
            self,
            items: List[Any],
            page: int,
            page_size: int = PAGE_SIZE,
    ) -> Tuple[List[Any], int]:
        pages = max(1, math.ceil(len(items) / page_size))
        page = max(0, min(page, pages - 1))
        start = page * page_size
        end = start + page_size
        return items[start:end], pages

    def make_dl_token(self) -> str:
        for _ in range(10):
            t = secrets.token_urlsafe(12)
            if t not in self._pending:
                return t
        return secrets.token_hex(12)

    def save_pending_url(self, user_id: int, url: str) -> str:
        token = self.make_dl_token()
        self._pending[token] = {
            "user_id": user_id,
            "url": url,
        }
        return token

    def get_pending(self, token: str) -> Optional[Dict[str, Any]]:
        return self._pending.get(token)

    def pop_pending(self, token: str) -> Optional[Dict[str, Any]]:
        return self._pending.pop(token, None)


_store = StateStore()


def get_user_mode(user_id: int) -> str:
    """Возвращает текущий режим пользователя.

    Args:
        user_id (int): Идентификатор пользователя Telegram.

    Returns:
        str: Один из: 'auto', 'audio', 'video', 'video_nosound'.
    """
    return _store.get_user_mode(user_id)


def set_user_mode(user_id: int, mode: str) -> None:
    """Сохраняет выбранный режим пользователя.

    Args:
        user_id (int): Идентификатор пользователя.
        mode (str): Режим ('auto'|'audio'|'video'|'video_nosound').
    """
    _store.set_user_mode(user_id, mode)


def get_user_lock(user_id: int) -> asyncio.Lock:
    """Получает или создаёт Lock для пользователя.

    Args:
        user_id (int): Идентификатор пользователя.

    Returns:
        asyncio.Lock: Lock пользователя.
    """
    return _store._get_lock(user_id)  # noqa


async def begin_user_download(user_id: int) -> Optional[asyncio.Lock]:
    """Пытается захватить пользовательский Lock перед загрузкой.

    Args:
        user_id (int): Идентификатор пользователя.

    Returns:
        Optional[asyncio.Lock]: Захваченный Lock или None если занят.
    """
    return await _store.begin_user_download(user_id)


async def end_user_download(lock: Optional[asyncio.Lock]) -> None:
    """Освобождает захваченный Lock.

    Args:
        lock (Optional[asyncio.Lock]): Объект блокировки.
    """
    await _store.end_user_download(lock)


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
    return _store.slice_page(items, page, page_size)


def remember_cookie_request(user_id: int, kind: str, url: Optional[str] = None, mode: Optional[str] = None) -> None:
    """Сохраняет ожидание cookies.

    Args:
        user_id (int): Пользователь.
        kind (str): Тип ('download'|'search').
        url (Optional[str]): URL для повтора.
        mode (Optional[str]): Режим ('audio'|'video'|'video_nosound'|'auto').
    """
    _store.remember_cookie_request(user_id, kind, url, mode)


def remember_search_cookie_request(
        user_id: int,
        query: str,
) -> None:
    """Сохраняет ожидание cookies для поиска.

    Args:
        user_id (int): Пользователь.
        query (str): Поисковый запрос.
    """
    _store.remember_search_cookie_request(user_id, query)


def get_user_cookies_path(user_id: int) -> str:
    """Возвращает путь к cookies.txt пользователя.

    Args:
        user_id (int): Идентификатор пользователя.

    Returns:
        str: Путь к cookies.txt.
    """
    return _store.get_user_cookies_path(user_id)


def make_dl_token() -> str:
    """Генерирует уникальный токен для отложенного скачивания.

    Returns:
        str: Токен (10 символов [A-Za-z0-9]).
    """
    return _store.make_dl_token()


def save_pending_url(user_id: int, url: str) -> str:
    """Сохраняет URL для последующего выбора режима отправки.

    Args:
        user_id (int): Идентификатор пользователя.
        url (str): Сохранённый URL.

    Returns:
        str: Токен сохранения.
    """
    return _store.save_pending_url(user_id, url)


def get_pending(token: str) -> Optional[Dict[str, Any]]:
    return _store.get_pending(token)


def pop_pending(token: str) -> Optional[Dict[str, Any]]:
    return _store.pop_pending(token)


def get_searches(user_id: int) -> Optional[Dict[str, Any]]:
    return _store.get_searches(user_id)


def set_searches(user_id: int, payload: Dict[str, Any]) -> None:
    _store.set_searches(user_id, payload)


def pop_searches(user_id: int) -> Optional[Dict[str, Any]]:
    return _store.pop_searches(user_id)


def get_awaiting(user_id: int) -> Optional[Dict[str, Any]]:
    return _store.get_awaiting(user_id)


def pop_awaiting(user_id: int) -> Optional[Dict[str, Any]]:
    return _store.pop_awaiting(user_id)
