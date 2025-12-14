import asyncio
import logging
import os
import shutil
import tempfile
from contextlib import suppress
from typing import Any, Callable, Dict, List, Optional, Tuple

from yt_dlp import YoutubeDL  # type: ignore[import-untyped]
from yt_dlp.utils import DownloadError  # type: ignore[import-untyped]


from ..config import (
    MAX_PLAYLIST_ITEMS,
    MAX_RESULTS,
    DURATION_LIMIT_SEC,
    CONCURRENT_DOWNLOADS,
)
from ..bot.dispatcher import logger

from ..utils.validators import is_audio_platform

from .media import (
    find_audio_files,
    find_image_files,
    find_video_files,
    norm_base,
    extract_id_from_base,
    process_thumbnail
)

try:
    from ..bot.dispatcher import download_sem  # type: ignore
except Exception:
    download_sem = asyncio.Semaphore(CONCURRENT_DOWNLOADS)


def make_duration_match_filter(max_seconds: int) -> Callable[[Dict[str, Any]], Optional[str]]:
    """Создаёт фильтр yt-dlp, отвергающий слишком длинные записи.

    Args:
        max_seconds (int): Максимальная длительность.

    Returns:
        Callable[[Dict[str, Any]], Optional[str]]: Фильтр (строка-причина или None).
    """

    def _mf(info: Dict[str, Any]) -> Optional[str]:
        dur = info.get("duration")
        if isinstance(dur, (int, float)) and dur > max_seconds:
            return f"duration>{max_seconds}"
        return None

    return _mf


def decide_effective_mode(user_mode: str, url: str) -> str:
    """Определяет итоговый режим скачивания.

    Args:
        user_mode (str): Режим выбранный пользователем ('auto', 'audio', 'video', 'video_nosound').
        url (str): URL источника.

    Returns:
        str: Итоговый режим ('audio'|'video'|'video_nosound').
    """
    if user_mode == "auto":
        return "audio" if is_audio_platform(url) else "video"
    return user_mode


async def ytdlp_extract(
        url_or_query: str, ydl_opts: Dict[str, Any], download: bool
) -> Dict[str, Any]:
    """Вызывает yt-dlp (извлечение или скачивание) в отдельном потоке.

    Args:
        url_or_query (str): URL или поисковый запрос.
        ydl_opts (Dict[str, Any]): Опции yt-dlp.
        download (bool): True для скачивания, False для извлечения.

    Returns:
        Dict[str, Any]: Результат extract_info.
    """

    def _run() -> Dict[str, Any]:
        with YoutubeDL(ydl_opts) as ydl:
            return ydl.extract_info(url_or_query, download=download)

    return await asyncio.to_thread(_run)


async def search_tracks(query: str, cookies_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Ищет треки на YouTube и фильтрует по длительности.

    Args:
        query (str): Поисковая строка.
        cookies_path (Optional[str]): Путь к cookies.txt.

    Returns:
        List[Dict[str, Any]]: Список словарей (title, url, duration, channel).
    """
    ydl_opts: Dict[str, Any] = {
        "quiet": True,
        "skip_download": True,
        "noplaylist": True,
        "default_search": "ytsearch",
    }
    if cookies_path and os.path.exists(cookies_path):
        ydl_opts["cookiefile"] = cookies_path

    info = await ytdlp_extract(f"ytsearch{MAX_RESULTS}:{query}", ydl_opts, download=False)
    entries = info.get("entries") or []
    results: List[Dict[str, Any]] = []
    for e in entries:
        duration = e.get("duration")
        if isinstance(duration, (int, float)) and duration > DURATION_LIMIT_SEC:
            continue
        url = e.get("webpage_url") or e.get("url")
        if not url and e.get("id"):
            url = f"https://www.youtube.com/watch?v={e['id']}"
        title = e.get("title") or "Без названия"
        channel = e.get("uploader") or e.get("channel") or ""
        results.append(
            {"title": title, "url": url, "duration": duration, "channel": channel}
        )
    return results


async def extract_basic_info(url: str, cookies_path: Optional[str] = None) -> Dict[str, Any]:
    """Извлекает базовую информацию без скачивания.

    Args:
        url (str): URL ресурса.
        cookies_path (Optional[str]): Путь к cookies.txt.

    Returns:
        Dict[str, Any]: title, duration, channel, thumbnail.
    """
    ydl_opts: Dict[str, Any] = {
        "quiet": True,
        "skip_download": True,
        "noplaylist": False,
        "playlist_items": "1",
        "logger": logging.getLogger("yt_dlp"),
    }
    if cookies_path and os.path.exists(cookies_path):
        ydl_opts["cookiefile"] = cookies_path
    info = await ytdlp_extract(url, ydl_opts, download=False)
    item = info
    try:
        entries = info.get("entries") if isinstance(info, dict) else None
        if isinstance(entries, list) and entries:
            item = entries[0]
    except Exception:
        pass

    def _pick_thumb(it: Dict[str, Any]) -> Optional[str]:
        t = it.get("thumbnail")
        if t:
            return t
        ts = it.get("thumbnails")
        if isinstance(ts, list) and ts:
            def key_fn(x: Dict[str, Any]) -> Tuple[int, int, int]:
                pref = int(x.get("preference") or 0)
                w = int(x.get("width") or 0)
                h = int(x.get("height") or 0)
                return pref, w * h, w + h

            try:
                ts_sorted = sorted(ts, key=key_fn, reverse=True)
                return ts_sorted[0].get("url")
            except Exception:
                with suppress(Exception):
                    return ts[-1].get("url")
        return None

    title = (
            (item.get("title") if isinstance(item, dict) else None)
            or (item.get("fulltitle") if isinstance(item, dict) else None)
            or (item.get("id") if isinstance(item, dict) else None)
            or "Без названия"
    )
    duration = (item.get("duration") if isinstance(item, dict) else None)
    channel = ""
    if isinstance(item, dict):
        channel = item.get("uploader") or item.get("channel") or ""
    thumbnail = _pick_thumb(item if isinstance(item, dict) else {})

    return {"title": title, "duration": duration, "channel": channel, "thumbnail": thumbnail}


async def download_media_to_temp(url: str, mode: str, cookies_path: Optional[str] = None) -> List[
    Tuple[str, Optional[str]]]:
    """Скачивает медиа и подготавливает миниатюры во временные директории.

    Args:
        url (str): Ссылка.
        mode (str): Режим ('audio'|'video'|'video_nosound').
        cookies_path (Optional[str]): Путь к cookies.txt.

    Returns:
        List[Tuple[str, Optional[str]]]: Пары (путь к медиа, путь к миниатюре или None).
    """
    tmpdir = tempfile.mkdtemp(prefix="dl_")
    if mode == "audio":
        postprocessors = [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            },
            {"key": "FFmpegThumbnailsConvertor", "format": "jpg"},
            {"key": "EmbedThumbnail"},
            {"key": "FFmpegMetadata"},
        ]
        ydl_format = "bestaudio/best"
        extra: Dict[str, Any] = {}
    elif mode == "video":
        postprocessors = [
            {"key": "FFmpegThumbnailsConvertor", "format": "jpg"},
            {"key": "FFmpegMetadata"},
        ]
        ydl_format = "bv*+ba/b"
        extra = {"merge_output_format": "mp4", "recode_video": "mp4"}
    else:
        postprocessors = [
            {"key": "FFmpegThumbnailsConvertor", "format": "jpg"},
            {"key": "FFmpegMetadata"},
        ]
        ydl_format = "bestvideo/best"
        extra = {"recode_video": "mp4"}

    ydl_opts: Dict[str, Any] = {
        "quiet": True,
        "format": ydl_format,
        "outtmpl": os.path.join(tmpdir, "%(title)s [%(id)s].%(ext)s"),
        "noplaylist": False,
        "postprocessors": postprocessors,
        "writethumbnail": True,
        "write_all_thumbnails": True,
        "convert_thumbnails": "jpg",
        "prefer_ffmpeg": True,
        "nocheckcertificate": True,
        "logger": logging.getLogger("yt_dlp"),
        "playlist_items": f"1-{MAX_PLAYLIST_ITEMS}",
        "max_downloads": MAX_PLAYLIST_ITEMS,
        "match_filter": make_duration_match_filter(DURATION_LIMIT_SEC),
        **extra,
    }
    if cookies_path and os.path.exists(cookies_path):
        ydl_opts["cookiefile"] = cookies_path

    async with download_sem:
        try:
            logger.info("Начало загрузки (%s): %s", mode, url)
            await ytdlp_extract(url, ydl_opts, download=True)
        except DownloadError as e:
            raise e
        except Exception as e:
            raise e

    if mode == "audio":
        media_files = find_audio_files(tmpdir)
    else:
        media_files = find_video_files(tmpdir)
    image_files = find_image_files(tmpdir)
    logger.info(
        "Файлов найдено (media=%d, images=%d)", len(media_files), len(image_files)
    )
    if not media_files:
        shutil.rmtree(tmpdir, ignore_errors=True)
        return []

    stable_dir = tempfile.mkdtemp(prefix="out_")

    images_by_base: Dict[str, List[str]] = {}
    for img in image_files:
        clean_base = norm_base(img)
        images_by_base.setdefault(clean_base, []).append(img)

    items: List[Tuple[str, Optional[str]]] = []
    for m in media_files:
        m_base = norm_base(m)
        m_dst = os.path.join(stable_dir, os.path.basename(m))
        with suppress(Exception):
            shutil.move(m, m_dst)

        possible_imgs = list(images_by_base.get(m_base, []))
        if not possible_imgs:
            vid = extract_id_from_base(m_base)
            if vid:
                needle = f"[{vid}]"
                for img in image_files:
                    name_wo_hash = os.path.basename(img).split("#", 1)[0]
                    if needle in name_wo_hash:
                        possible_imgs.append(img)

        t_src: Optional[str] = None
        if possible_imgs:
            with suppress(Exception):
                possible_imgs.sort(key=lambda p: os.path.getsize(p), reverse=True)
            t_src = possible_imgs[0]

        t_dst: Optional[str] = None
        if t_src and os.path.exists(t_src):
            moved = os.path.join(stable_dir, os.path.basename(t_src))
            with suppress(Exception):
                shutil.move(t_src, moved)
            logger.info("Обрабатываю обложку: %s", moved)
            processed = process_thumbnail(moved, stable_dir)
            if os.path.exists(moved) and (not processed or processed != moved):
                with suppress(Exception):
                    os.remove(moved)
            if processed and os.path.exists(processed):
                t_dst = processed

        items.append((m_dst, t_dst))

    shutil.rmtree(tmpdir, ignore_errors=True)
    return items
