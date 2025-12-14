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
    """Get user's current download mode.

    Args:
        user_id: Telegram user ID.

    Returns:
        Mode string ('auto', 'audio', 'video').
    """
    return (USER_SETTINGS.get(user_id) or {}).get("mode", "auto")


def set_user_mode(user_id: int, mode: str) -> None:
    """Set user's download mode.

    Args:
        user_id: Telegram user ID.
        mode: Mode to set ('auto', 'audio', 'video').
    """
    if user_id not in USER_SETTINGS:
        USER_SETTINGS[user_id] = {}
    USER_SETTINGS[user_id]["mode"] = mode


def get_user_lock(user_id: int) -> asyncio.Lock:
    """Get or create an asyncio lock for a user.

    Args:
        user_id: Telegram user ID.

    Returns:
        User's asyncio.Lock instance.
    """
    if user_id not in USER_LOCKS:
        USER_LOCKS[user_id] = asyncio.Lock()
    return USER_LOCKS[user_id]


async def begin_user_download(user_id: int) -> Optional[asyncio.Lock]:
    """Attempt to acquire user's download lock (non-blocking).

    Args:
        user_id: Telegram user ID.

    Returns:
        Lock instance if acquired, None if already locked.
    """
    lock = get_user_lock(user_id)
    if lock.locked():
        return None
    return lock


async def end_user_download(lock: Optional[asyncio.Lock]) -> None:
    """Release user's download lock if held.

    Args:
        lock: Lock instance to release (or None).
    """
    if lock is not None and lock.locked():
        lock.release()


def slice_page(
    items: List[Any],
    page: int,
    page_size: int = PAGE_SIZE,
) -> Tuple[List[Any], int]:
    """Get a slice of items for pagination.

    Args:
        items: Full list of items.
        page: Current page number (0-indexed).
        page_size: Items per page.

    Returns:
        Tuple of (page_items, total_pages, current_page).
    """
    pages = max(1, math.ceil(len(items) / page_size))
    page = max(0, min(page, pages - 1))
    start = page * page_size
    end = start + page_size
    return items[start:end], pages


def remember_cookie_request(user_id: int, kind: str, url: Optional[str] = None, mode: Optional[str] = None) -> None:
    """Store pending cookie request for a direct URL download.

    Args:
        user_id: Telegram user ID.
        kind (str): Тип ('download'|'search').
        url: URL that requires cookies.
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
    """Store pending cookie request for a search query.

    Args:
        user_id: Telegram user ID.
        query: Search query that requires cookies.
    """
    AWAITING_COOKIES[user_id] = {"kind": "search", "query": query, "asked": True}


def get_user_cookies_path(user_id: int) -> str:
    """Get path to user's cookies file.

    Args:
        user_id: Telegram user ID.

    Returns:
        Path string to cookies file.
    """
    return os.path.join(COOKIES_DIR, f"{user_id}_cookies.txt")


def make_dl_token() -> str:
    """Generate a unique download token.

    Returns:
        Random hex token string.
    """
    t = ""
    for _ in range(5):
        t = secrets.token_urlsafe(6).replace("-", "").replace("_", "")[:10]
        if t not in PENDING_DOWNLOADS:
            break
    return t


def save_pending_url(user_id: int, url: str) -> str:
    """Save URL for pending download and return token.

    Args:
        user_id: Telegram user ID.
        url: URL to save.

    Returns:
        Token to retrieve the pending download.
    """
    token = make_dl_token()
    PENDING_DOWNLOADS[token] = {
        "user_id": user_id,
        "url": url,
    }
    return token
