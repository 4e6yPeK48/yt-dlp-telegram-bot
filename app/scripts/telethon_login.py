from telethon import TelegramClient
from dotenv import load_dotenv
import os

load_dotenv()

client = TelegramClient(
    "telethon",
    int(os.getenv("TELETHON_API_ID")),
    os.getenv("TELETHON_API_HASH"),
)


async def main() -> None:
    await client.start(phone=os.getenv("PHONE"))
    print("✅ Telethon логин выполнен")


with client:
    client.loop.run_until_complete(main())
