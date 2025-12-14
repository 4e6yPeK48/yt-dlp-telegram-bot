import re
from typing import Any, Optional
from ..config import CAPTION_MAX_LEN


def sanitize_query(text: str, max_len: int = 256) -> str:
    """Очищает поисковый запрос (управляющие символы, пробелы, длину).

    Args:
        text (str): Исходный текст.

    Returns:
        str: Санитизированный запрос.
    """
    t = re.sub(r"[\x00-\x1f\x7f]", "", text)
    t = re.sub(r"[\u200B-\u200F\u202A-\u202E\u2060-\u206F]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > max_len:
        t = t[:max_len]
    return t


def make_caption(text: str, limit: int = CAPTION_MAX_LEN) -> str:
    """Очищает текст и обрезает его для подписи (однострочно).

    Args:
        text (str): Исходный текст.
        limit (int): Максимальная длина.

    Returns:
        str: Подготовленная подпись.
    """
    t = re.sub(r"[\x00-\x1f\x7f]", "", text or "")
    t = re.sub(r"[\u200B-\u200F\u202A-\u202E\u2060-\u206F]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > limit:
        t = t[: limit - 1] + "…"
    return t


def make_multiline_caption(text: str, limit: int = CAPTION_MAX_LEN) -> str:
    """Очищает текст (с сохранением перевода строк) и обрезает до лимита.

    Args:
        text (str): Исходный текст.
        limit (int): Максимальная длина.

    Returns:
        str: Подготовленный многострочный текст.
    """
    t = text or ""
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[\x00-\x09\x0B-\x0C\x0E-\x1F\x7F]", "", t)
    t = re.sub(r"[\u200B-\u200F\u202A-\u202E\u2060-\u206F]", "", t)
    lines = [line.rstrip() for line in t.split("\n")]
    t = "\n".join(lines)
    if len(t) > limit:
        t = t[: limit - 1] + "…"
    return t


def format_duration_hms(dur_any: Optional[Any]) -> str:
    """Форматирует длительность в мм:сс или чч:мм:сс.

    Args:
        dur_any (Optional[Any]): Длительность в секундах.

    Returns:
        str: Форматированная строка или '—'.
    """
    if isinstance(dur_any, (int, float)) and dur_any >= 0:
        sec = int(dur_any)
        h, rem = divmod(sec, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
    return "—"


def parse_main_button_intent(text: str) -> Optional[str]:
    """Определяет намерение пользователя на основе текста кнопки.

    Args:
        text (str): Текст кнопки.

    Returns:
        Optional[str]: Намерение ('menu', 'help', 'settings') или None.
    """
    t = (text or "").strip()
    if not t:
        return None
    low = t.lower()

    if re.search(r"/start\b", low) or re.search(r"/menu\b", low):
        return "menu"
    if re.search(r"/help\b", low):
        return "help"
    if re.search(r"/settings\b", low):
        return "settings"

    cleaned = re.sub(r"[^\w\sА-Яа-яёЁ-]", " ", low)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if re.search(r"\bменю\b", cleaned):
        return "menu"
    if re.search(r"\bпомощ", cleaned):
        return "help"
    if re.search(r"\bнастрой", cleaned):
        return "settings"

    return None
