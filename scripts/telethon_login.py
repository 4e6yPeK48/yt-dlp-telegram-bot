import asyncio
import os
from telethon import TelegramClient

from config import TELETHON_API_ID, TELETHON_API_HASH, TELETHON_SESSION

async def main():
    if not TELETHON_API_ID or not TELETHON_API_HASH:
        print("Set TELETHON_API_ID and TELETHON_API_HASH environment variables first.")
        return
    client = TelegramClient(TELETHON_SESSION, TELETHON_API_ID, TELETHON_API_HASH)
    await client.start()  # interactive: phone + code
    me = await client.get_me()
    print("Logged in as:", me.stringify())
    await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())

# RUN LOCALLY: python scripts/telethon_login.py
# Afterwards, ensure TELETHON_SESSION is protected (.gitignore).
