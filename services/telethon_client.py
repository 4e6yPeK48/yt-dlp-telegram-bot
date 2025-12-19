import asyncio
import os
from contextlib import suppress
from typing import Optional

from telethon import TelegramClient, events, errors
from telethon.tl.types import InputPeerUser

from config import TELETHON_API_ID, TELETHON_API_HASH, TELETHON_SESSION, TELETHON_FALLBACK_ENABLED
from bot.dispatcher import download_sem, logger

_client: Optional[TelegramClient] = None
_client_lock = asyncio.Lock()
_me_cache: Optional[dict] = None


async def ensure_client_started() -> None:
    """Initialize and connect the Telethon client singleton. Fatal if unauthorized when fallback enabled."""
    global _client, _me_cache
    if _client is not None and _client.is_connected():
        return
    if not TELETHON_API_ID or not TELETHON_API_HASH:
        logger.warning("Telethon API credentials not provided; Telethon fallback disabled.")
        return
    async with _client_lock:
        if _client is not None:
            return
        _client = TelegramClient(TELETHON_SESSION, TELETHON_API_ID, TELETHON_API_HASH)
        try:
            await _client.connect()
            is_auth = await _client.is_user_authorized()
            if not is_auth:
                msg = "Telethon session is not authorized. No interactive login allowed in bot."
                logger.error(msg)
                if TELETHON_FALLBACK_ENABLED:
                    raise RuntimeError(msg)
                else:
                    await _client.disconnect()
                    _client = None
                    return
            me = await _client.get_me()
            _me_cache = {"id": getattr(me, "id", None), "username": getattr(me, "username", None), "title": str(me)}
            logger.info("Telethon client connected as %s (id=%s)", _me_cache.get("username") or _me_cache.get("title"),
                        _me_cache.get("id"))
        except Exception:
            logger.exception("Failed to start Telethon client.")
            if _client:
                with suppress(Exception):
                    await _client.disconnect()
                _client = None
            raise


async def disconnect_client() -> None:
    """Disconnect Telethon client on shutdown; do not delete session."""
    global _client
    global _me_cache
    if _client is None:
        return
    try:
        await _client.disconnect()
        logger.info("Telethon client disconnected.")
    except Exception:
        logger.exception("Error while disconnecting Telethon client.")
    finally:
        _client = None
        _me_cache = None


def get_client() -> Optional[TelegramClient]:
    return _client


def get_username() -> Optional[str]:
    return _me_cache.get("username") if _me_cache else None


async def wait_for_user_message(user_id: int, timeout: int = 120) -> bool:
    """Wait for any incoming message from user_id to the Telethon client within timeout seconds."""
    client = get_client()
    if not client:
        logger.info("wait_for_user_message: Telethon client not available.")
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
        logger.exception("Error waiting for user message via Telethon.")
        try:
            client.remove_event_handler(_handler, events.NewMessage)
        except Exception:
            pass
        return False


async def send_file_via_user(chat_id: int, file_path: str, *, caption: Optional[str] = None,
                             thumb: Optional[str] = None, supports_streaming: bool = False) -> None:
    """Send file via the authorized user account. Handles FloodWait and basic retries. Cleans up on completion/error."""
    client = get_client()
    if not client:
        raise RuntimeError("Telethon client is not initialized")
    async with download_sem:
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                try:
                    entity = await client.get_entity(chat_id)
                except ValueError as ve:
                    logger.warning("get_entity failed for %s on attempt %d: %s", str(chat_id), attempt, ve)
                    try:
                        await client.get_dialogs(limit=10)
                    except Exception:
                        pass
                    entity = await client.get_entity(chat_id)

                kwargs = {}
                if caption:
                    kwargs["caption"] = caption
                if thumb and os.path.exists(thumb):
                    kwargs["thumb"] = thumb

                await asyncio.wait_for(
                    client.send_file(entity, file_path, **kwargs),
                    timeout=300
                )
                logger.info("Sent file via Telethon to %s: %s", str(chat_id), file_path)
                return
            except errors.FloodWaitError as e:
                wait = int(getattr(e, "seconds", 5))
                logger.warning("Telethon FloodWait %s seconds; sleeping...", wait)
                await asyncio.sleep(wait + 1)
            except asyncio.TimeoutError:
                logger.warning("Telethon send timeout on attempt %d for %s", attempt, file_path)
            except Exception as e:
                logger.exception("Telethon send attempt %d failed: %s", attempt, str(e))
            await asyncio.sleep(1 * attempt)
        raise RuntimeError("Failed to send file via Telethon after retries")
