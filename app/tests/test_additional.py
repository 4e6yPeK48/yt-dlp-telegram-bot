import asyncio
import logging

import pytest

from app.utils.logging import OnlyLoggerFilter
from app.utils.text import parse_main_button_intent
from app.storage.state import (
    begin_user_download,
    end_user_download,
    get_download_task,
    cancel_download_task,
    set_download_task,
    get_user_cookies_path,
    make_dl_token,
)


@pytest.mark.asyncio
async def test_user_download_lock_behavior():
    """begin_user_download / end_user_download: захват и освобождение блокировки."""
    uid = 1234567890
    # первый begin должен захватить блокировку
    lock1 = await begin_user_download(uid)
    assert lock1 is not None
    # второй begin должен вернуть None (уже заблокировано)
    lock2 = await begin_user_download(uid)
    assert lock2 is None
    # освобождаем и проверяем, что можем захватить снова
    await end_user_download(lock1)
    lock3 = await begin_user_download(uid)
    assert lock3 is not None
    await end_user_download(lock3)


@pytest.mark.asyncio
async def test_set_get_and_cancel_download_task():
    """set/get/cancel: цикл для asyncio.Task, сохранённой в состоянии."""
    uid = 42424242

    async def sleeper():
        # простая долгоживущая корутина, которую можно отменить
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            # повторно возбуждаем, чтобы ожидающие вызовы могли увидеть отмену
            raise

    task = asyncio.create_task(sleeper())
    set_download_task(uid, task)
    got = get_download_task(uid)
    assert got is task

    cancelled = cancel_download_task(uid)
    assert cancelled is True

    # ожидание отменённой задачи должно вызвать CancelledError
    with pytest.raises(asyncio.CancelledError):
        await task


def test_only_logger_filter():
    """OnlyLoggerFilter пропускает только записи с заданным префиксом имени логгера."""
    f = OnlyLoggerFilter("bot")
    rec1 = logging.LogRecord(
        name="bot.handlers",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="x",
        args=(),
        exc_info=None,
    )
    rec2 = logging.LogRecord(
        name="other",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="x",
        args=(),
        exc_info=None,
    )
    assert f.filter(rec1) is True
    assert f.filter(rec2) is False


def test_parse_main_button_intent_additional_cases():
    """parse_main_button_intent: русские варианты, пунктуация и разные регистры."""
    assert parse_main_button_intent("/start") == "menu"
    assert parse_main_button_intent("меню") == "menu"
    assert parse_main_button_intent("Помощь, пожалуйста") == "help"
    assert parse_main_button_intent("настройки") == "settings"
    assert parse_main_button_intent("история") == "history"
    # пунктуация / смешанный регистр
    assert parse_main_button_intent(" /Help ") == "help"


def test_make_dl_token_uniqueness_and_cookies_path():
    """make_dl_token генерирует уникальные токены; путь к cookies содержит идентификатор пользователя."""
    tokens = {make_dl_token() for _ in range(200)}
    assert len(tokens) == 200
    uid = 777
    path = get_user_cookies_path(uid)
    # путь должен содержать суффикс с id пользователя и _cookies.txt
    assert path.endswith(f"{uid}_cookies.txt")
