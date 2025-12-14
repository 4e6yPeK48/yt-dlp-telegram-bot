"""URL and platform validation utilities."""

from contextlib import suppress
from urllib.parse import urlparse


def is_url(text: str) -> bool:
    """Check if a string is a valid HTTP/HTTPS URL.

    Args:
        text: Input string to check.

    Returns:
        True if the string is a valid URL with http/https scheme.
    """
    with suppress(Exception):
        u = urlparse(text.strip())
        return u.scheme in {"http", "https"} and bool(u.netloc)
    return False


def is_youtube_url(url: str) -> bool:
    """Check if a URL belongs to YouTube or YouTube Music.

    Args:
        url: URL to check.

    Returns:
        True if the URL is from YouTube.
    """
    try:
        host = (urlparse(url).netloc or "").lower()
    except Exception:
        return False
    return any(
        h in host
        for h in ("youtube.", "youtu.be", "music.youtube.")
    )


def is_audio_platform(url: str) -> bool:
    """Heuristically determine if the URL is from an audio-oriented platform.

    Args:
        url: Resource URL.

    Returns:
        True if the site appears to be an audio platform.
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
