import asyncio
from telethon import TelegramClient

from config import TELETHON_API_ID, TELETHON_API_HASH, TELETHON_SESSION


async def main() -> None:
    if not TELETHON_API_ID or not TELETHON_API_HASH:
        print("Установите сначала переменные окружения TELETHON_API_ID и TELETHON_API_HASH.")
        return
    client = TelegramClient(TELETHON_SESSION, TELETHON_API_ID, TELETHON_API_HASH)
    await client.start()
    me = await client.get_me()
    print("Вход выполнен как:", me.stringify())
    await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())

# ЗАПУСК ЛОКАЛЬНО: python scripts/telethon_login.py
