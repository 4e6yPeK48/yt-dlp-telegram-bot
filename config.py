import os
from typing import Optional, Set, Tuple

from dotenv import load_dotenv

load_dotenv()

# Bot token from environment
# BOT_TOKEN: Optional[str] = os.getenv("BOT_TOKEN")
BOT_TOKEN: Optional[str] = "8524336951:AAFG9KLCfuSRK6igDPbP2RLnkikYMKlXxyk"

# Search and pagination
MAX_RESULTS: int = 25
PAGE_SIZE: int = 5

# Download limits
CONCURRENT_DOWNLOADS: int = 2
MAX_PLAYLIST_ITEMS: int = 10
DURATION_LIMIT_SEC: int = 30 * 60
YTDLP_THREAD_TIMEOUT: int = 5 * 60  # seconds

# Query limits
MAX_QUERY_LEN: int = 120

# File extensions
AUDIO_EXTS: Set[str] = {".mp3", ".m4a", ".opus", ".webm", ".ogg", ".flac", ".wav"}
IMAGE_EXTS: Set[str] = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTS: Set[str] = {".mp4", ".mkv", ".webm", ".mov", ".m4v"}

# Thumbnail settings
THUMB_SIZE: Tuple[int, int] = (320, 320)
THUMB_MAX_BYTES: int = 200 * 1024

# Telegram limits
CAPTION_MAX_LEN: int = 1000
TG_MAX_UPLOAD_BYTES: int = int(os.getenv("TG_MAX_UPLOAD_MB", "50")) * 1024 * 1024

# Cookies settings
COOKIES_MAX_BYTES: int = int(os.getenv("COOKIES_MAX_MB", "5")) * 1024 * 1024
ALLOWED_COOKIES_EXTS: Set[str] = {".txt"}
COOKIES_DIR: str = os.path.join(os.getcwd(), "cookies")

# Ensure cookies directory exists
os.makedirs(COOKIES_DIR, exist_ok=True)

# Button labels
BTN_MENU: str = "🏠 Меню (/start, /menu)"
BTN_HELP: str = "❓ Помощь (/help)"
BTN_SETTINGS: str = "⚙️ Настройки (/settings)"
