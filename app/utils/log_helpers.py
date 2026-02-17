import logging
from sys import exc_info
from typing import Any, Dict, Optional

from app.config import TRUNC_URL


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
    """
    Форматирует контекст для логирования.

    Args:
        user_id (Optional[int]): Идентификатор пользователя.
        chat_id (Optional[int]): Идентификатор чата.
        mode (Optional[str]): Режим работы.
        url (Optional[str]): URL-адрес.
        extra (Optional[Dict[str, Any]]): Дополнительные параметры.

    Returns:
        str: Отформатированная строка контекста.
    """
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
    *args: Any,
    user_id: Optional[int] = None,
    chat_id: Optional[int] = None,
    mode: Optional[str] = None,
    url: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> None:
    """
    Логирует информационное сообщение с контекстом.

    Args:
        logger (logging.Logger): Логгер для записи сообщений.
        msg (str): Сообщение для логирования.
        *args: Дополнительные позиционные аргументы для логгера.
        user_id (Optional[int]): Идентификатор пользователя.
        chat_id (Optional[int]): Идентификатор чата.
        mode (Optional[str]): Режим работы.
        url (Optional[str]): URL-адрес.
        extra (Optional[Dict[str, Any]]): Дополнительные параметры.
        **kwargs: Дополнительные именованные аргументы для логгера.

    Returns:
        None
    """
    logger.info(msg + _format_ctx(user_id, chat_id, mode, url, extra), *args, **kwargs)


def log_warning(
    logger: logging.Logger,
    msg: str,
    *args: Any,
    user_id: Optional[int] = None,
    chat_id: Optional[int] = None,
    mode: Optional[str] = None,
    url: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> None:
    """
    Логирует предупреждающее сообщение с контекстом.

    Args:
        logger (logging.Logger): Логгер для записи сообщений.
        msg (str): Сообщение для логирования.
        *args: Дополнительные позиционные аргументы для логгера.
        user_id (Optional[int]): Идентификатор пользователя.
        chat_id (Optional[int]): Идентификатор чата.
        mode (Optional[str]): Режим работы.
        url (Optional[str]): URL-адрес.
        extra (Optional[Dict[str, Any]]): Дополнительные параметры.
        **kwargs: Дополнительные именованные аргументы для логгера.

    Returns:
        None
    """
    logger.warning(
        msg + _format_ctx(user_id, chat_id, mode, url, extra), *args, **kwargs
    )


def log_error(
    logger: logging.Logger,
    msg: str,
    *args: Any,
    user_id: Optional[int] = None,
    chat_id: Optional[int] = None,
    mode: Optional[str] = None,
    url: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> None:
    """
    Логирует сообщение об ошибке с контекстом.

    Args:
        logger (logging.Logger): Логгер для записи сообщений.
        msg (str): Сообщение для логирования.
        *args: Дополнительные позиционные аргументы для логгера.
        user_id (Optional[int]): Идентификатор пользователя.
        chat_id (Optional[int]): Идентификатор чата.
        mode (Optional[str]): Режим работы.
        url (Optional[str]): URL-адрес.
        extra (Optional[Dict[str, Any]]): Дополнительные параметры.
        **kwargs: Дополнительные именованные аргументы для логгера.

    Returns:
        None
    """
    logger.error(msg + _format_ctx(user_id, chat_id, mode, url, extra), *args, **kwargs)


def log_debug(
    logger: logging.Logger,
    msg: str,
    *args: Any,
    user_id: Optional[int] = None,
    chat_id: Optional[int] = None,
    mode: Optional[str] = None,
    url: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> None:
    """
    Логирует отладочное сообщение с контекстом.

    Args:
        logger (logging.Logger): Логгер для записи сообщений.
        msg (str): Сообщение для логирования.
        *args: Дополнительные позиционные аргументы для логгера.
        user_id (Optional[int]): Идентификатор пользователя.
        chat_id (Optional[int]): Идентификатор чата.
        mode (Optional[str]): Режим работы.
        url (Optional[str]): URL-адрес.
        extra (Optional[Dict[str, Any]]): Дополнительные параметры.
        **kwargs: Дополнительные именованные аргументы для логгера.

    Returns:
        None
    """
    logger.debug(msg + _format_ctx(user_id, chat_id, mode, url, extra), *args, **kwargs)


def log_exception(
    logger: logging.Logger,
    msg: str,
    *args: Any,
    user_id: Optional[int] = None,
    chat_id: Optional[int] = None,
    mode: Optional[str] = None,
    url: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> None:
    """
    Логирует сообщение об исключении с контекстом.

    Args:
        logger (logging.Logger): Логгер для записи сообщений.
        msg (str): Сообщение для логирования.
        *args: Дополнительные позиционные аргументы для логгера.
        user_id (Optional[int]): Идентификатор пользователя.
        chat_id (Optional[int]): Идентификатор чата.
        mode (Optional[str]): Режим работы.
        url (Optional[str]): URL-адрес.
        extra (Optional[Dict[str, Any]]): Дополнительные параметры.
        **kwargs: Дополнительные именованные аргументы для логгера.

    Returns:
        None
    """
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


def _current_exception_marked() -> Optional[BaseException]:
    """
    Возвращает текущее исключение из стека вызовов.

    Returns:
        Optional[BaseException]: Текущее исключение или None.
    """
    exc = exc_info()[1]
    return exc
