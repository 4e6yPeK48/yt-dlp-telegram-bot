import asyncio
import os
from contextlib import suppress
from typing import Optional, Callable, Awaitable

from telethon import TelegramClient, events, errors
from telethon.tl.types import InputPeerUser

from config import TELETHON_API_ID, TELETHON_API_HASH, TELETHON_SESSION, TELETHON_FALLBACK_ENABLED
from bot.dispatcher import download_sem, logger

_client: Optional[TelegramClient] = None
_client_lock = asyncio.Lock()
_me_cache: Optional[dict] = None


async def ensure_client_started() -> None:
    """Инициализировать и подключить синглтон Telethon-клиента.

    Выполняет ленивую инициализацию клиента. Если учётные данные не заданы,
    ничего не делает. Если сессия не авторизована и TELETHON_FALLBACK_ENABLED
    включён, возбуждает RuntimeError.

    Raises:
        RuntimeError: Если сессия не авторизована и TELETHON_FALLBACK_ENABLED == True.
        Exception: Пробрасывает исключения при ошибках подключения.
    """
    global _client, _me_cache
    if _client is not None and _client.is_connected():
        return
    if not TELETHON_API_ID or not TELETHON_API_HASH:
        logger.warning("Учётные данные Telethon не заданы; fallback Telethon отключён.")
        return
    async with _client_lock:
        if _client is not None:
            return
        _client = TelegramClient(TELETHON_SESSION, TELETHON_API_ID, TELETHON_API_HASH)
        try:
            await _client.connect()
            is_auth = await _client.is_user_authorized()
            if not is_auth:
                msg = "Сессия Telethon не авторизована. Интерактивный вход недоступен в боте."
                logger.error(msg)
                if TELETHON_FALLBACK_ENABLED:
                    raise RuntimeError(msg)
                else:
                    await _client.disconnect()
                    _client = None
                    return
            me = await _client.get_me()
            _me_cache = {"id": getattr(me, "id", None), "username": getattr(me, "username", None), "title": str(me)}
            logger.info("Telethon-клиент подключён как %s (id=%s)", _me_cache.get("username") or _me_cache.get("title"),
                        _me_cache.get("id"))
        except Exception:
            logger.exception("Не удалось запустить Telethon-клиент.")
            if _client:
                with suppress(Exception):
                    await _client.disconnect()
                _client = None
            raise


async def disconnect_client() -> None:
    """Отключить Telethon-клиент при завершении работы.

    Не удаляет файл сессии, только аккуратно отключается.

    Returns:
        None
    """
    global _client
    global _me_cache
    if _client is None:
        return
    try:
        await _client.disconnect()
        logger.info("Telethon-клиент отключён.")
    except Exception:
        logger.exception("Ошибка при отключении Telethon-клиента.")
    finally:
        _client = None
        _me_cache = None


def get_client() -> Optional[TelegramClient]:
    """Получить текущий экземпляр Telethon-клиента (или None).
    Returns:
        Optional[TelegramClient]: Экземпляр клиента или None.
    """
    return _client


def get_username() -> Optional[str]:
    """Вернуть username авторизованного аккаунта, если доступен.

    Returns:
        Optional[str]: username или None.
    """
    return _me_cache.get("username") if _me_cache else None


async def wait_for_user_message(user_id: int, timeout: int = 120) -> bool:
    """Ожидать любое входящее сообщение от пользователя в указанный таймаут.

    Args:
        user_id (int): Идентификатор пользователя/чата, от которого ожидается сообщение.
        timeout (int): Таймаут в секундах.

    Returns:
        bool: True если сообщение получено, False при таймауте или ошибке.
    """
    client = get_client()
    if not client:
        logger.info("wait_for_user_message: Telethon-клиент недоступен.")
        return False

    fut = asyncio.get_event_loop().create_future()

    @client.on(events.NewMessage(from_users=user_id))
    async def _handler(event):
        if not fut.done():
            fut.set_result(True)
        client.remove_event_handler(_handler, events.NewMessage)

    try:
        return await asyncio.wait_for(fut, timeout=timeout)
    except asyncio.TimeoutError:
        try:
            client.remove_event_handler(_handler, events.NewMessage)
        except Exception:
            pass
        return False
    except Exception:
        logger.exception("Ошибка при ожидании сообщения пользователя через Telethon.")
        try:
            client.remove_event_handler(_handler, events.NewMessage)
        except Exception:
            pass
        return False


async def send_file_via_user(
    chat_id: int,
    file_path: str,
    *,
    caption: Optional[str] = None,
    thumb: Optional[str] = None,
    supports_streaming: bool = False,
    notify: Optional[Callable[[str], Awaitable[None]]] = None,
) -> None:
    """Отправить файл через авторизованный пользовательский аккаунт Telethon.

    Args:
        chat_id (int): Идентификатор получателя (user/chat id).
        file_path (str): Путь к файлу для отправки.
        caption (Optional[str]): Подпись к файлу.
        thumb (Optional[str]): Путь к миниатюре.
        supports_streaming (bool): Флаг для поддержки стриминга (для видео).
        notify (Optional[Callable[[str], Awaitable[None]]]): Необязательный корутин для
            отправки статусов (например, "загрузка", "повтор", "готово") обратно боту/пользователю.

    Raises:
        RuntimeError: Если Telethon-клиент не инициализирован.
        Exception: Пробрасывает исключения при фатальных ошибках отправки.
    """
    client = get_client()
    if not client:
        raise RuntimeError("Telethon-клиент не инициализирован")
    async with download_sem:
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                if notify:
                    try:
                        await notify("⏳ Альтернативная доставка: подготовка...")
                    except Exception:
                        logger.debug("notify не удался (подготовка)")

                try:
                    entity = await client.get_entity(chat_id)
                except ValueError as ve:
                    logger.warning("get_entity не удался для %s на попытке %d: %s", str(chat_id), attempt, ve)
                    if notify:
                        with suppress(Exception):
                            await notify("⚠️ Альтернативная доставка: пользователь не найден в сессии; обновляю диалоги...")
                    try:
                        await client.get_dialogs(limit=20)
                    except Exception:
                        pass
                    entity = await client.get_entity(chat_id)

                kwargs = {}
                if caption:
                    kwargs["caption"] = caption
                if thumb and os.path.exists(thumb):
                    kwargs["thumb"] = thumb
                if supports_streaming:
                    kwargs["supports_streaming"] = True

                if notify:
                    try:
                        await notify("⏳ Альтернативная доставка: начинаю загрузку...")
                    except Exception:
                        logger.debug("notify не удался (начало загрузки)")

                await asyncio.wait_for(
                    client.send_file(entity, file_path, **kwargs),
                    timeout=300
                )

                if notify:
                    try:
                        await notify("✅ Альтернативная доставка: загрузка завершена.")
                    except Exception:
                        logger.debug("notify не удался (завершено)")

                logger.info("Отправлено через Telethon пользователю %s: %s", str(chat_id), file_path)
                return
            except errors.FloodWaitError as e:
                wait = int(getattr(e, "seconds", 5))
                logger.warning("Telethon FloodWait %s секунд; ожидаю...", wait)
                if notify:
                    with suppress(Exception):
                        await notify(f"⚠️ Альтернативная доставка: ожидание из-за лимита {wait} с...")
                await asyncio.sleep(wait + 1)
            except asyncio.TimeoutError:
                logger.warning("Таймаут отправки Telethon на попытке %d для %s", attempt, file_path)
                if notify:
                    with suppress(Exception):
                        await notify(f"⚠️ Альтернативная доставка: загрузка превысила время (попытка {attempt}). Повтор...")
            except Exception as e:
                logger.exception("Попытка отправки Telethon №%d завершилась ошибкой: %s", attempt, str(e))
                if notify:
                    with suppress(Exception):
                        await notify(f"⚠️ Ошибка альтернативной доставки (попытка {attempt}): {str(e)}")
            await asyncio.sleep(1 * attempt)
        if notify:
            with suppress(Exception):
                await notify("❌ Альтернативная доставка не удалась после повторных попыток.")
        raise RuntimeError("Не удалось отправить файл через Telethon после повторных попыток")
