from contextlib import suppress
from urllib.parse import urlparse


def is_url(text: str) -> bool:
    """Проверяет, является ли строка URL со схемой http/https.

    Args:
        text (str): Исходная строка.

    Returns:
        bool: True, если строка похожа на URL, иначе False.
    """
    with suppress(Exception):
        u = urlparse(text.strip())
        return u.scheme in {"http", "https"} and bool(u.netloc)
    return False


def is_youtube_url(url: str) -> bool:
    """Определяет, относится ли URL к YouTube/YouTube Music.

    Args:
        url (str): Проверяемый URL.

    Returns:
        bool: True, если URL относится к YouTube.
    """
    try:
        host = (urlparse(url).netloc or "").lower()
    except Exception:
        return False
    return any(h in host for h in ("youtube.", "youtu.be", "music.youtube."))


def is_audio_platform(url: str) -> bool:
    """Эвристически определяет, что ресурс аудио-ориентирован.

    Args:
        url (str): URL ресурса.

    Returns:
        bool: True, если сайт похоже аудио-площадка.
    """
    try:
        u = urlparse(url)
        host = (u.netloc or "").lower()
        path = (u.path or "").lower()
    except Exception:
        return False
    audio_hosts = [
        "music.youtube.",
        "soundcloud.com",
        "bandcamp.com",
        "mixcloud.com",
        "audius.co",
        "hearthis.at",
        "promodj.com",
        "music.yandex.",
        "yandex.ru/music",
        "deezer.com",
        "napster.com",
    ]
    return any(h in host for h in audio_hosts) or "/music" in path
