import asyncio
import os
from contextlib import suppress
from typing import Optional, Callable, Awaitable

from telethon import TelegramClient, events, errors

from config import TELETHON_API_ID, TELETHON_API_HASH, TELETHON_SESSION, TELETHON_FALLBACK_ENABLED, \
    TELETHON_UPLOAD_TIMEOUT, CONCURRENT_DOWNLOADS
from bot.dispatcher import logger

try:
    from bot.dispatcher import telethon_sem  # type: ignore
except Exception:
    telethon_sem = asyncio.Semaphore(CONCURRENT_DOWNLOADS)

NotifyCallable = Optional[Callable[[str], Awaitable[None]]]


class TelethonManager:
    """Менеджер синглтон Telethon-клиента для альтернативной доставки файлов."""
    def __init__(self, session: str = TELETHON_SESSION, api_id: Optional[int] = TELETHON_API_ID,
                 api_hash: Optional[str] = TELETHON_API_HASH) -> None:
        self._session = session
        self._api_id = api_id
        self._api_hash = api_hash
        self._client: Optional[TelegramClient] = None
        self._client_lock = asyncio.Lock()
        self._me_cache: Optional[dict] = None

    async def ensure_started(self) -> None:
        if not TELETHON_FALLBACK_ENABLED:
            logger.info("Telethon-fallback отключён; пропускаю инициализацию клиента.")
            return

        if self._client is not None and getattr(self._client, "is_connected", lambda: True)():
            return

        if not self._api_id or not self._api_hash:
            logger.warning("Учётные данные Telethon не заданы; fallback Telethon отключён.")
            return

        async with self._client_lock:
            if self._client is not None:
                return
            self._client = TelegramClient(self._session, self._api_id, self._api_hash)
            try:
                await self._client.connect()
                is_auth = await self._client.is_user_authorized()
                if not is_auth:
                    msg = "Сессия Telethon не авторизована. Интерактивный вход недоступен в боте."
                    logger.error(msg)
                    raise RuntimeError(msg)
                me = await self._client.get_me()
                self._me_cache = {
                    "id": getattr(me, "id", None),
                    "username": getattr(me, "username", None),
                    "title": str(me),
                }
                logger.info("Telethon-клиент подключён как %s (id=%s)",
                            self._me_cache.get("username") or self._me_cache.get("title"), self._me_cache.get("id"))
            except Exception:
                logger.exception("Не удалось запустить Telethon-клиент.")
                if self._client:
                    with suppress(Exception):
                        await self._client.disconnect()
                    self._client = None
                raise

    async def disconnect(self) -> None:
        if not self._client:
            return
        try:
            await self._client.disconnect()
            logger.info("Telethon-клиент отключён.")
        except Exception:
            logger.exception("Ошибка при отключении Telethon-клиента.")
        finally:
            self._client = None
            self._me_cache = None

    def get_client(self) -> Optional[TelegramClient]:
        return self._client

    def get_username(self) -> Optional[str]:
        return self._me_cache.get("username") if self._me_cache else None

    async def wait_for_user_message(self, user_id: int, timeout: int = 120) -> bool:
        client = self.get_client()
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
            except Exception as e:
                logger.exception(
                    "Failed to remove Telethon event handler after timeout (user_id=%s): %s",
                    user_id,
                    e,
                )
            return False
        except Exception as e:
            logger.exception(
                "Ошибка при ожидании сообщения пользователя через Telethon (user_id=%s): %s",
                user_id,
                e,
            )
            try:
                client.remove_event_handler(_handler, events.NewMessage)
            except Exception as e2:
                logger.exception(
                    "Failed to remove Telethon event handler after error (user_id=%s): %s",
                    user_id,
                    e2,
                )
            return False

    async def send_file_via_user(
            self,
            chat_id: int,
            file_path: str,
            *,
            caption: Optional[str] = None,
            thumb: Optional[str] = None,
            supports_streaming: bool = False,
            notify: NotifyCallable = None,
    ) -> None:
        client = self.get_client()
        if not client:
            raise RuntimeError("Telethon-клиент не инициализирован")
        async with telethon_sem:
            max_retries = 3
            for attempt in range(1, max_retries + 1):
                try:
                    if notify:
                        with suppress(Exception):
                            logger.info("Уведомление пользователя о подготовке альтернативной доставки.")
                            await notify("⏳ Альтернативная доставка: подготовка...")
                    try:
                        entity = await client.get_entity(chat_id)
                    except ValueError as ve:
                        logger.warning("get_entity не удался для %s на попытке %d: %s", str(chat_id), attempt, str(ve))
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
                        with suppress(Exception):
                            logger.info("Уведомление пользователя о начале загрузки альтернативной доставки.")
                            await notify("⏳ Альтернативная доставка: начинаю загрузку...")

                    await asyncio.wait_for(client.send_file(entity, file_path, **kwargs), timeout=TELETHON_UPLOAD_TIMEOUT)

                    if notify:
                        with suppress(Exception):
                            logger.info("Уведомление пользователя о завершении альтернативной доставки.")
                            await notify("✅ Альтернативная доставка: загрузка завершена.")
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
                            await notify(
                                f"⚠️ Альтернативная доставка: загрузка превысила время (попытка {attempt}). Повтор...")
                except Exception as e:
                    logger.exception("Попытка отправки Telethon №%d завершилась ошибкой: %s", attempt, str(e))
                    if notify:
                        with suppress(Exception):
                            await notify(f"⚠️ Ошибка альтернативной доставки (попытка {attempt}): {str(e)}")
                await asyncio.sleep(2 * attempt)
            if notify:
                with suppress(Exception):
                    await notify("❌ Альтернативная доставка не удалась после повторных попыток.")
            raise RuntimeError("Не удалось отправить файл через Telethon после повторных попыток")

    async def request_alternate_delivery_and_send(
            self,
            bot,
            chat_id: int,
            file_path: str,
            caption: Optional[str] = None,
            thumb: Optional[str] = None,
            supports_streaming: bool = False,
            timeout: int = 120,
    ) -> bool:
        if not self.get_client():
            logger.info("request_alternate_delivery_and_send: Telethon-клиент не инициализирован.")
            return False
        username = self.get_username() or "alternate account"
        try:
            logger.info("Попытка альтернативной доставки файла %s пользователю %s через Telethon.", file_path, str(chat_id))
            try:
                await bot.send_message(chat_id,
                                       f"⚠️ Файл большой — будет попытка альтернативной доставки. Пожалуйста, отправьте любое сообщение @{username} (альтернативному аккаунту) в течение 120 секунд.")
            except Exception:
                logger.exception("Не удалось уведомить пользователя о переходе на альтернативную доставку.")
            got = await self.wait_for_user_message(chat_id, timeout=timeout)
            if not got:
                try:
                    await bot.send_message(chat_id,
                                           "⌛ Таймаут ожидания сообщения альтернативному аккаунту. Нельзя доставить большой файл.")
                except Exception:
                    pass
                return False

            try:
                await bot.send_message(chat_id,
                                       "✅ Налажено соединение. Начинаю альтернативную доставку через авторизованный аккаунт...")
            except Exception:
                logger.exception("Не удалось уведомить пользователя о полученном рукопожатии.")

            async def _notify_to_user(text: str) -> None:
                try:
                    await bot.send_message(chat_id, text)
                except Exception:
                    logger.debug("Не удалось отправить уведомление пользователю: %s", text)

            try:
                await self.send_file_via_user(chat_id, file_path, caption=caption, thumb=thumb,
                                              supports_streaming=supports_streaming, notify=_notify_to_user)
                try:
                    await bot.send_message(chat_id, "✅ Файл доставлен через альтернативный аккаунт.")
                except Exception as e:
                    logger.exception(
                        "Не удалось уведомить пользователя об успешной альтернативной доставке (chat_id=%s, file=%s): %s",
                        chat_id,
                        file_path,
                        e,
                    )
                return True
            except Exception as e:
                logger.exception(
                    "Альтернативная доставка через Telethon не удалась (chat_id=%s, file=%s): %s",
                    chat_id,
                    file_path,
                    e,
                )
                try:
                    await bot.send_message(chat_id,
                                           "❌ Альтернативная доставка не удалась (проблемы с правами или внутренняя ошибка). Убедитесь, что вы начали диалог с альтернативным аккаунтом и не заблокировали его.")
                except Exception as e2:
                    logger.exception(
                        "Не удалось уведомить пользователя о неудачной альтернативной доставке (chat_id=%s, file=%s): %s",
                        chat_id,
                        file_path,
                        e2,
                    )
                return False
        except Exception:
            logger.exception("Ошибка в request_alternate_delivery_and_send (chat_id=%s, file=%s): %s", chat_id, file_path, e)
            return False


telethon_manager = TelethonManager()


async def ensure_client_started() -> None:
    """Инициализировать и подключить синглтон Telethon-клиента.

    Выполняет ленивую инициализацию клиента. Если учётные данные не заданы,
    ничего не делает. Если сессия не авторизована и TELETHON_FALLBACK_ENABLED
    включён, возбуждает RuntimeError.

    Returns:
        None
    """
    return await telethon_manager.ensure_started()


async def disconnect_client() -> None:
    """Отключить Telethon-клиент при завершении работы.

    Не удаляет файл сессии, только аккуратно отключается.

    Returns:
        None
    """
    return await telethon_manager.disconnect()


def get_client() -> Optional[TelegramClient]:
    """Получить текущий экземпляр Telethon-клиента (или None).
    Returns:
        Optional[TelegramClient]: Экземпляр клиента или None.
    """
    return telethon_manager.get_client()


def get_username() -> Optional[str]:
    """Вернуть username авторизованного аккаунта, если доступен.

    Returns:
        Optional[str]: username или None.
    """
    return telethon_manager.get_username()


async def wait_for_user_message(user_id: int, timeout: int = 120) -> bool:
    """Ожидать любое входящее сообщение от пользователя в указанный таймаут.

    Args:
        user_id (int): Идентификатор пользователя/чата, от которого ожидается сообщение.
        timeout (int): Таймаут в секундах.

    Returns:
        bool: True если сообщение получено, False при таймауте или ошибке.
    """
    return await telethon_manager.wait_for_user_message(user_id, timeout=timeout)


async def send_file_via_user(
        chat_id: int,
        file_path: str,
        *,
        caption: Optional[str] = None,
        thumb: Optional[str] = None,
        supports_streaming: bool = False,
        notify: NotifyCallable = None,
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
    return await telethon_manager.send_file_via_user(chat_id, file_path, caption=caption, thumb=thumb,
                                                     supports_streaming=supports_streaming, notify=notify)


async def request_alternate_delivery_and_send(
        bot,
        chat_id: int,
        file_path: str,
        caption: Optional[str] = None,
        thumb: Optional[str] = None,
        supports_streaming: bool = False,
        timeout: int = 120,
) -> bool:
    """
    Запросить у пользователя подтверждение для альтернативной доставки
    через авторизованный Telethon-аккаунт и отправить файл.
    Args:
        bot: Экземпляр бота для отправки сообщений.
        chat_id (int): Идентификатор чата пользователя.
        file_path (str): Путь к файлу для отправки.
        caption (Optional[str]): Подпись к файлу.
        thumb (Optional[str]): Путь к миниатюре.
        supports_streaming (bool): Флаг поддержки стриминга.
        timeout (int): Таймаут ожидания сообщения от пользователя.

    Returns:
        bool: True если доставка успешна, False в случае ошибки или таймаута.
    """
    return await telethon_manager.request_alternate_delivery_and_send(bot, chat_id, file_path, caption=caption,
                                                                      thumb=thumb,
                                                                      supports_streaming=supports_streaming,
                                                                      timeout=timeout)
