import asyncio
import math
import secrets
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from bot.dispatcher import logger
from utils.log_helpers import log_info, log_exception
from config import COOKIES_DIR, PAGE_SIZE


class StateStore:
    """
    Хранилище состояния пользователей.
    """

    def __init__(self, cookies_dir: str = COOKIES_DIR) -> None:
        self._searches: Dict[int, Dict[str, Any]] = {}
        self._awaiting: Dict[int, Dict[str, Any]] = {}
        self._settings: Dict[int, Dict[str, str]] = {}
        self._locks: Dict[int, asyncio.Lock] = {}
        self._pending: Dict[str, Dict[str, Any]] = {}
        self._history: Dict[int, List[Dict[str, Any]]] = {}
        self._hist_pages: Dict[int, int] = {}
        self._download_tasks: Dict[int, asyncio.Task[Any]] = {}
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

    def remember_cookie_request(
        self,
        user_id: int,
        kind: str,
        url: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> None:
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

    def add_download_history(self, user_id: int, entry: Dict[str, Any], max_items: int = 200) -> None:
        if user_id not in self._history:
            self._history[user_id] = []
        self._history[user_id].insert(0, {"time": int(time.time()), **entry})
        if len(self._history[user_id]) > max_items:
            self._history[user_id] = self._history[user_id][:max_items]

    def get_history(self, user_id: int) -> List[Dict[str, Any]]:
        return list(self._history.get(user_id, []))

    def clear_history(self, user_id: int) -> None:
        self._history.pop(user_id, None)
        self._hist_pages.pop(user_id, None)

    def get_history_page(self, user_id: int) -> int:
        return self._hist_pages.get(user_id, 0)

    def set_history_page(self, user_id: int, page: int) -> None:
        self._hist_pages[user_id] = int(page)

    def reset_history_page(self, user_id: int) -> None:
        self._hist_pages.pop(user_id, None)

    def set_download_task(self, user_id: int, task: asyncio.Task[Any]) -> None:
        self._download_tasks[user_id] = task

    def get_download_task(self, user_id: int) -> Optional[asyncio.Task[Any]]:
        return self._download_tasks.get(user_id)

    def pop_download_task(self, user_id: int) -> Optional[asyncio.Task[Any]]:
        return self._download_tasks.pop(user_id, None)

    def cancel_download_task(self, user_id: int) -> bool:
        t = self._download_tasks.pop(user_id, None)
        if t:
            log_info(logger, "Отмена запроса: попытка отменить выполняемую задачу загрузки", user_id=user_id)
            try:
                t.cancel()
                log_info(logger, "Задача загрузки отменена", user_id=user_id)
            except Exception as e:
                log_exception(
                    logger,
                    "Не удалось отменить задачу загрузки для пользователя",
                    user_id=user_id,
                    extra={"err": str(e)},
                )
            return True
        log_info(logger, "Отмена запроса: активных задач загрузки не найдено", user_id=user_id)
        return False


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
    """
    Получает сохранённый URL по токену.

    Args:
        token (str): Токен сохранённого URL.

    Returns:
        Optional[Dict[str, Any]]: Данные сохранённого URL или None.
    """
    return _store.get_pending(token)


def pop_pending(token: str) -> Optional[Dict[str, Any]]:
    """
    Извлекает и удаляет сохранённый URL по токену.

    Args:
        token (str): Токен сохранённого URL.

    Returns:
        Optional[Dict[str, Any]]: Данные сохранённого URL или None.
    """
    return _store.pop_pending(token)


def get_searches(user_id: int) -> Optional[Dict[str, Any]]:
    """
    Получает сохранённые результаты поиска пользователя.

    Args:
        user_id (int): Идентификатор пользователя.

    Returns:
        Optional[Dict[str, Any]]: Данные поиска или None.
    """
    return _store.get_searches(user_id)


def set_searches(user_id: int, payload: Dict[str, Any]) -> None:
    """
    Сохраняет результаты поиска пользователя.

    Args:
        user_id (int): Идентификатор пользователя.
        payload (Dict[str, Any]): Данные поиска.

    Returns:
        None
    """
    _store.set_searches(user_id, payload)


def pop_searches(user_id: int) -> Optional[Dict[str, Any]]:
    """
    Извлекает и удаляет сохранённые результаты поиска пользователя.

    Args:
        user_id (int): Идентификатор пользователя.

    Returns:
        Optional[Dict[str, Any]]: Данные поиска или None.
    """
    return _store.pop_searches(user_id)


def get_awaiting(user_id: int) -> Optional[Dict[str, Any]]:
    """
    Получает данные ожидания cookies от пользователя.

    Args:
        user_id (int): Идентификатор пользователя.

    Returns:
        Optional[Dict[str, Any]]: Данные ожидания или None.
    """
    return _store.get_awaiting(user_id)


def pop_awaiting(user_id: int) -> Optional[Dict[str, Any]]:
    """
    Извлекает и удаляет данные ожидания cookies от пользователя.

    Args:
        user_id (int): Идентификатор пользователя.

    Returns:
        Optional[Dict[str, Any]]: Данные ожидания или None.
    """
    return _store.pop_awaiting(user_id)


def add_download_history(user_id: int, entry: Dict[str, Any], max_items: int = 200) -> None:
    """
    Добавляет запись в историю загрузок пользователя.

    Args:
        user_id (int): Идентификатор пользователя.
        entry (Dict[str, Any]): Запись истории.
        max_items (int): Максимальное число записей в истории.

    Returns:
        None
    """
    _store.add_download_history(user_id, entry, max_items)


def get_history(user_id: int) -> List[Dict[str, Any]]:
    """
    Получает историю загрузок пользователя.

    Args:
        user_id (int): Идентификатор пользователя.

    Returns:
        List[Dict[str, Any]]: Список записей истории.
    """
    return _store.get_history(user_id)


def clear_history(user_id: int) -> None:
    """
    Очищает историю загрузок пользователя.

    Args:
        user_id (int): Идентификатор пользователя.

    Returns:
        None
    """
    _store.clear_history(user_id)


def get_history_page(user_id: int) -> int:
    """
    Получает текущую страницу истории загрузок пользователя.

    Args:
        user_id (int): Идентификатор пользователя.

    Returns:
        int: Номер текущей страницы истории.
    """
    return _store.get_history_page(user_id)


def set_history_page(user_id: int, page: int) -> None:
    """
    Устанавливает текущую страницу истории загрузок пользователя.

    Args:
        user_id (int): Идентификатор пользователя.
        page (int): Номер страницы.

    Returns:
        None
    """
    _store.set_history_page(user_id, page)


def reset_history_page(user_id: int) -> None:
    """
    Сбрасывает текущую страницу истории загрузок пользователя.

    Args:
        user_id (int): Идентификатор пользователя.

    Returns:
        None
    """
    _store.reset_history_page(user_id)


def set_download_task(user_id: int, task: asyncio.Task[Any]) -> None:
    """
    Сохраняет задачу загрузки для пользователя.

    Args:
        user_id (int): Идентификатор пользователя.
        task (asyncio.Task): Задача загрузки.

    Returns:
        None
    """
    _store.set_download_task(user_id, task)


def get_download_task(user_id: int) -> Optional[asyncio.Task[Any]]:
    """
    Получает задачу загрузки пользователя.

    Args:
        user_id (int): Идентификатор пользователя.

    Returns:
        Optional[asyncio.Task]: Задача загрузки или None.
    """
    return _store.get_download_task(user_id)


def pop_download_task(user_id: int) -> Optional[asyncio.Task[Any]]:
    """
    Извлекает и удаляет задачу загрузки пользователя.

    Args:
        user_id (int): Идентификатор пользователя.

    Returns:
        Optional[asyncio.Task]: Задача загрузки или None.
    """
    return _store.pop_download_task(user_id)


def cancel_download_task(user_id: int) -> bool:
    """
    Отменяет задачу загрузки пользователя, если она существует.

    Args:
        user_id (int): Идентификатор пользователя.

    Returns:
        bool: True если задача была отменена, False если не найдена.
    """
    return _store.cancel_download_task(user_id)
