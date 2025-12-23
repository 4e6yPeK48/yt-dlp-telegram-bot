import logging
from sys import exc_info
from typing import Any, Dict, Optional

from config import TRUNC_URL


def _truncate(s: Optional[str], n: int = TRUNC_URL) -> Optional[str]:
    """
    Обрезает строку до заданной длины, добавляя "..." в конце, если необходимо.

    Args:
        s (Optional[str]): Исходная строка.
        n (int): Максимальная длина строки.

    Returns:
        Optional[str]: Обрезанная строка или None, если входная строка была None.
    """
    if s is None:
        return None
    return s if len(s) <= n else s[: n - 3] + "..."


def _format_ctx(
    user_id: Optional[int] = None,
    chat_id: Optional[int] = None,
    mode: Optional[str] = None,
    url: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    parts: list[str] = []
    if user_id is not None:
        parts.append(f"user={user_id}")
    if chat_id is not None:
        parts.append(f"chat={chat_id}")
    if mode:
        parts.append(f"mode={mode}")
    if url:
        parts.append(f"url={_truncate(url)}")
    if extra:
        kv = " ".join(f"{k}={v}" for k, v in extra.items())
        parts.append(kv)
    return " | " + " ".join(parts) if parts else ""


def log_info(
    logger: logging.Logger,
    msg: str,
    *args,
    user_id: Optional[int] = None,
    chat_id: Optional[int] = None,
    mode: Optional[str] = None,
    url: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> None:
    logger.info(msg + _format_ctx(user_id, chat_id, mode, url, extra), *args, **kwargs)


def log_warning(
    logger: logging.Logger,
    msg: str,
    *args,
    user_id: Optional[int] = None,
    chat_id: Optional[int] = None,
    mode: Optional[str] = None,
    url: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> None:
    logger.warning(msg + _format_ctx(user_id, chat_id, mode, url, extra), *args, **kwargs)


def log_error(
    logger: logging.Logger,
    msg: str,
    *args,
    user_id: Optional[int] = None,
    chat_id: Optional[int] = None,
    mode: Optional[str] = None,
    url: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> None:
    logger.error(msg + _format_ctx(user_id, chat_id, mode, url, extra), *args, **kwargs)


def log_debug(
    logger: logging.Logger,
    msg: str,
    *args,
    user_id: Optional[int] = None,
    chat_id: Optional[int] = None,
    mode: Optional[str] = None,
    url: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> None:
    logger.debug(msg + _format_ctx(user_id, chat_id, mode, url, extra), *args, **kwargs)


def _current_exception_marked() -> Optional[BaseException]:
    """
    Возвращает текущее исключение из стека вызовов.

    Returns:
        Optional[BaseException]: Текущее исключение или None.
    """
    exc = exc_info()[1]
    return exc


def log_exception(
    logger: logging.Logger,
    msg: str,
    *args,
    user_id: Optional[int] = None,
    chat_id: Optional[int] = None,
    mode: Optional[str] = None,
    url: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> None:
    exc = _current_exception_marked()
    ctx = _format_ctx(user_id, chat_id, mode, url, extra)

    if exc is not None and getattr(exc, "_logged", False):
        logger.error(msg + ctx, *args, **kwargs)
        return

    if exc is not None:
        try:
            setattr(exc, "_logged", True)
        except Exception:
            pass

    logger.exception(msg + ctx, *args, **kwargs)
