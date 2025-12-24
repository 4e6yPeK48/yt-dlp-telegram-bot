import json
import os
from typing import Optional, Set, Tuple, Dict

from dotenv import load_dotenv

load_dotenv()

# Bot token from environment
BOT_TOKEN: Optional[str] = os.getenv("BOT_TOKEN")

# Telethon fallback config
TELETHON_API_ID: Optional[int] = int(os.getenv("TELETHON_API_ID", "0")) or None
TELETHON_API_HASH: Optional[str] = os.getenv("TELETHON_API_HASH") or None
TELETHON_SESSION: str = os.getenv("TELETHON_SESSION", "telethon.session")
TELETHON_FALLBACK_ENABLED: bool = os.getenv("TELETHON_FALLBACK_ENABLED", "True").lower() in ("1", "true", "yes")


# Search and pagination
MAX_RESULTS: int = 10
PAGE_SIZE: int = 5

# Download limits
CONCURRENT_DOWNLOADS: int = 3
MAX_PLAYLIST_ITEMS: int = 10
DURATION_LIMIT_SEC: int = 30 * 60
YTDLP_THREAD_TIMEOUT: int = 10 * 60  # seconds
TELETHON_UPLOAD_TIMEOUT: int = 10 * 60  # seconds, default 10 minutes

# Safety cap for single downloaded file (default 2 GiB)
MAX_FILE_MB: int = int(os.getenv("MAX_FILE_MB", "2048"))  # megabytes
MAX_FILE_BYTES: int = MAX_FILE_MB * 1024 * 1024

# Query limits
MAX_QUERY_LEN: int = 120
TRUNC_URL = 200

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

# Server-side cookies dir & refresh settings
SERVER_COOKIES_DIR: str = os.path.join(os.getcwd(), "server_cookies")
SERVER_COOKIES_MAP: Dict[str, str] = {
    "youtube.": "server_youtube_cookies.txt",
    "youtu.be": "server_youtube_cookies.txt",
    "vimeo.": "server_vimeo_cookies.txt",
    "tiktok.": "server_tiktok_cookies.txt",
}
SERVER_COOKIES_SOURCES: Dict[str, str] = {}
try:
    _env = os.getenv("SERVER_COOKIES_SOURCES_JSON")
    if _env:
        SERVER_COOKIES_SOURCES = json.loads(_env)
except Exception:
    SERVER_COOKIES_SOURCES = SERVER_COOKIES_SOURCES

# Refresh interval in seconds (default 1 hour)
SERVER_COOKIES_REFRESH_INTERVAL_SEC: int = int(os.getenv("SERVER_COOKIES_REFRESH_INTERVAL_SEC", "3600"))

# Ensure dir exists (already present in file)
os.makedirs(SERVER_COOKIES_DIR, exist_ok=True)

# Button labels
BTN_MENU: str = "🏠 Меню"
BTN_HELP: str = "❓ Помощь"
BTN_SETTINGS: str = "⚙️ Настройки"
BTN_HISTORY: str = "📜 История"
