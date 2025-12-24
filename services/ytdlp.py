import asyncio
import logging
import os
import shutil
import tempfile
from contextlib import suppress
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

from config import (
    MAX_PLAYLIST_ITEMS,
    MAX_RESULTS,
    DURATION_LIMIT_SEC,
    CONCURRENT_DOWNLOADS,
    YTDLP_THREAD_TIMEOUT,
    MAX_FILE_BYTES,
    SERVER_COOKIES_MAP,
    SERVER_COOKIES_DIR,
)
from bot.dispatcher import logger
from utils.log_helpers import log_debug, log_info, log_error, log_exception

from utils.validators import is_audio_platform

from services.media import (
    find_audio_files,
    find_image_files,
    find_video_files,
    norm_base,
    extract_id_from_base,
    process_thumbnail_sync,
)

try:
    from bot.dispatcher import download_sem
except Exception:
    download_sem = asyncio.Semaphore(CONCURRENT_DOWNLOADS)


class FileTooLargeError(Exception):
    """Исключение, поднимаемое при скачивании файла превышающем MAX_FILE_BYTES."""

    pass


def make_duration_match_filter(
    max_seconds: int,
) -> Callable[[Dict[str, Any]], Optional[str]]:
    """Создаёт фильтр для yt-dlp, отвергающий записи длиннее max_seconds.

    Args:
        max_seconds (int): Максимальная допустимая длительность в секундах.

    Returns:
        Callable[[Dict[str, Any]], Optional[str]]: Функция-фильтр, возвращающая строку причины или None.
    """

    def _mf(info: Dict[str, Any]) -> Optional[str]:
        dur = info.get("duration")
        if isinstance(dur, (int, float)) and dur > max_seconds:
            return f"duration>{max_seconds}"
        return None

    return _mf


def decide_effective_mode(user_mode: str, url: str) -> str:
    """Определяет фактический режим скачивания на основе выбора пользователя и URL.

    Args:
        user_mode (str): Выбранный пользователем режим ('auto', 'audio', 'video', 'video_nosound').
        url (str): URL ресурса.

    Returns:
        str: Эффективный режим ('audio'|'video'|'video_nosound').
    """
    if user_mode == "auto":
        return "audio" if is_audio_platform(url) else "video"
    return user_mode


class YTDLPExtractor:
    """
    Извлекатель информации с помощью yt-dlp.
    """

    async def ytdlp_extract(self, url_or_query: str, ydl_opts: Dict[str, Any], download: bool) -> Dict[str, Any]:
        """Запускает yt-dlp.extract_info в отдельном потоке с таймаутом.

        Args:
            url_or_query (str): URL или поисковый запрос.
            ydl_opts (Dict[str, Any]): Опции для YoutubeDL.
            download (bool): Флаг: True — выполнить скачивание, False — только извлечь информацию.

        Returns:
            Dict[str, Any]: Результат ydl.extract_info.

        Raises:
            asyncio.TimeoutError: При превышении YTDLP_THREAD_TIMEOUT.
            Exception: При других ошибках выполнения yt-dlp.
        """
        log_debug(logger, "ytdlp_extract: запуск", url=url_or_query, extra={"download": download})

        def _run() -> Dict[str, Any]:
            with YoutubeDL(ydl_opts) as ydl:
                return ydl.extract_info(url_or_query, download=download)

        try:
            result = await asyncio.wait_for(asyncio.to_thread(_run), timeout=YTDLP_THREAD_TIMEOUT)
            log_debug(logger, "ytdlp_extract: завершено", url=url_or_query)
            return result
        except asyncio.TimeoutError:
            log_error(logger, "ytdlp_extract: таймаут", url=url_or_query, extra={"timeout_sec": YTDLP_THREAD_TIMEOUT})
            raise
        except Exception as e:
            log_exception(logger, "ytdlp_extract: исключение", url=url_or_query, extra={"err": str(e)})
            raise

    async def search_tracks(self, query: str, cookies_path: Optional[str] = None) -> List[Dict[str, Any]]:
        ydl_opts: Dict[str, Any] = {
            "quiet": True,
            "skip_download": True,
            "noplaylist": True,
            "default_search": "ytsearch",
        }
        if cookies_path and await asyncio.to_thread(os.path.exists, cookies_path):  # mypy: ignore
            ydl_opts["cookiefile"] = cookies_path

        info = await self.ytdlp_extract(f"ytsearch{MAX_RESULTS}:{query}", ydl_opts, download=False)
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
            results.append({"title": title, "url": url, "duration": duration, "channel": channel})
        return results

    async def extract_basic_info(self, url: str, cookies_path: Optional[str] = None) -> Dict[str, Any]:
        ydl_opts: Dict[str, Any] = {
            "quiet": True,
            "skip_download": True,
            "noplaylist": False,
            "playlist_items": "1",
            "logger": logging.getLogger("yt_dlp"),
        }
        if cookies_path and await asyncio.to_thread(os.path.exists, cookies_path):  # mypy: ignore
            ydl_opts["cookiefile"] = cookies_path

        info = await self.ytdlp_extract(url, ydl_opts, download=False)
        item = info
        try:
            entries = info.get("entries") if isinstance(info, dict) else None
            if isinstance(entries, list) and entries:
                item = entries[0]
        except Exception as e:
            log_debug(logger, "extract_basic_info: не удалось выбрать первую запись", url=url, extra={"err": str(e)})

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
                    with __import__("contextlib").suppress(Exception):
                        return ts[-1].get("url")
            return None

        title = (
            (item.get("title") if isinstance(item, dict) else None)
            or (item.get("fulltitle") if isinstance(item, dict) else None)
            or (item.get("id") if isinstance(item, dict) else None)
            or "Без названия"
        )
        duration = item.get("duration") if isinstance(item, dict) else None
        channel = ""
        if isinstance(item, dict):
            channel = item.get("uploader") or item.get("channel") or ""
        thumbnail = _pick_thumb(item if isinstance(item, dict) else {})

        return {"title": title, "duration": duration, "channel": channel, "thumbnail": thumbnail}

    async def find_server_cookies_for_url(self, url: str) -> Optional[str]:
        try:
            host = (urlparse(url).netloc or "").lower()
        except Exception:
            return None
        for key, fname in SERVER_COOKIES_MAP.items():
            if key in host:
                path = os.path.join(SERVER_COOKIES_DIR, fname)
                if await asyncio.to_thread(os.path.exists, path):
                    try:
                        size = await asyncio.to_thread(os.path.getsize, path)
                        if size > 0:
                            return path
                    except Exception:
                        return None
        return None


class YTDLPPostprocessor:
    """
    Постпроцессор для yt-dlp: перемещает файлы в стабильную директорию, готовит миниатюры.
    """

    def __init__(self, max_file_bytes: Optional[int] = None):
        self.max_file_bytes = max_file_bytes

    def _postprocess_sync(self, tmpdir_: str, mode_: str, url: str) -> List[Tuple[str, Optional[str]]]:
        """
        Синхронно обрабатывает скачанные файлы: перемещает в стабильную директорию, готовит миниатюры.

        Args:
            tmpdir_ (str): Временная директория с загруженными файлами.
            mode_ (str): Режим ('audio'|'video'|'video_nosound').
            url (str): Исходный URL.

        Returns:
            List[Tuple[str, Optional[str]]]: Список пар (путь_к_медиа, путь_к_миниатюре_or_None).
        """
        stable_dir = None
        try:
            if mode_ == "audio":
                media_files = find_audio_files(tmpdir_)
            else:
                media_files = find_video_files(tmpdir_)
            image_files = find_image_files(tmpdir_)
            log_info(
                logger,
                "Файлов найдено",
                mode=mode_,
                url=url,
                extra={"media_count": len(media_files), "image_count": len(image_files)},
            )
            if not media_files:
                shutil.rmtree(tmpdir_, ignore_errors=True)
                return []

            stable_dir = tempfile.mkdtemp(prefix="out_")

            images_by_base: Dict[str, List[str]] = {}
            for img in image_files:
                clean_base = norm_base(img)
                images_by_base.setdefault(clean_base, []).append(img)

            items_: List[Tuple[str, Optional[str]]] = []
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
                    log_info(logger, "Обрабатываю обложку", mode=mode_, extra={"thumb": moved})
                    try:
                        processed = process_thumbnail_sync(moved, stable_dir)
                    except Exception:
                        processed = None

                    if os.path.exists(moved) and (not processed or processed != moved):
                        with suppress(Exception):
                            os.remove(moved)
                    if processed and os.path.exists(processed):
                        t_dst = processed

                try:
                    size = os.path.getsize(m_dst)
                except Exception:
                    size = 0
                if self.max_file_bytes and size and size > self.max_file_bytes:
                    shutil.rmtree(stable_dir, ignore_errors=True)
                    shutil.rmtree(tmpdir_, ignore_errors=True)
                    log_error(
                        logger,
                        "Скачанный файл превышает максимально допустимый размер",
                        mode=mode_,
                        extra={"size": size, "path": m_dst, "limit": self.max_file_bytes},
                    )
                    raise FileTooLargeError(f"File too large: {size} bytes (limit {self.max_file_bytes})")

                items_.append((m_dst, t_dst))

            shutil.rmtree(tmpdir_, ignore_errors=True)
            return items_
        except Exception:
            if stable_dir:
                shutil.rmtree(stable_dir, ignore_errors=True)
            shutil.rmtree(tmpdir_, ignore_errors=True)
            raise


class YTDLPDownloader:
    """
    Загрузчик медиа с помощью yt-dlp с постобработкой.
    """

    def __init__(self, extractor: YTDLPExtractor, postprocessor: YTDLPPostprocessor):
        self.extractor = extractor
        self.postprocessor = postprocessor

    async def download_media_to_temp(
        self, url: str, mode: str, cookies_path: Optional[str] = None
    ) -> List[Tuple[str, Optional[str]]]:
        """
        Скачивает медиа с помощью yt-dlp и подготавливает миниатюры во временную директорию.

        Args:
            url (str): Ссылка на ресурс.
            mode (str): Режим ('audio'|'video'|'video_nosound').
            cookies_path (Optional[str]): Путь к cookies.txt.

        Returns:
            List[Tuple[str, Optional[str]]]: Список пар (путь_к_медиа, путь_к_миниатюре_or_None).
        """
        tmpdir = tempfile.mkdtemp(prefix="dl_")
        try:
            if mode == "audio":
                postprocessors = [
                    {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"},
                    {"key": "EmbedThumbnail"},
                    {"key": "FFmpegMetadata"},
                ]
                ydl_format = "bestaudio/best"
                extra: Dict[str, Any] = {}
            elif mode == "video":
                postprocessors = [{"key": "FFmpegMetadata"}]
                ydl_format = "bv*+ba/b"
                extra = {"merge_output_format": "mp4", "recode_video": "mp4"}
            else:
                postprocessors = [{"key": "FFmpegMetadata"}]
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
                "prefer_ffmpeg": True,
                "nocheckcertificate": True,
                "logger": logging.getLogger("yt_dlp"),
                "playlist_items": f"1-{MAX_PLAYLIST_ITEMS}",
                "max_downloads": MAX_PLAYLIST_ITEMS,
                "match_filter": make_duration_match_filter(DURATION_LIMIT_SEC),
                **extra,
            }
            if cookies_path and await asyncio.to_thread(os.path.exists, cookies_path):  # mypy: ignore
                ydl_opts["cookiefile"] = cookies_path

            async with download_sem:
                log_info(logger, "Начало загрузки", mode=mode, url=url)
                await self.extractor.ytdlp_extract(url, ydl_opts, download=True)

            items = await asyncio.to_thread(self.postprocessor._postprocess_sync, tmpdir, mode, url)
            return items
        except DownloadError:
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise
        except FileTooLargeError:
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise
        except Exception as e:
            log_exception(logger, "Неожиданная ошибка в download_media_to_temp", url=url, extra={"err": str(e)})
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise


_extractor = YTDLPExtractor()
_postprocessor = YTDLPPostprocessor(max_file_bytes=MAX_FILE_BYTES)
_downloader = YTDLPDownloader(_extractor, _postprocessor)


async def search_tracks(query: str, cookies_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Ищет треки по запросу (YouTube search) и фильтрует по длине.

    Args:
        query (str): Поисковая строка.
        cookies_path (Optional[str]): Путь к cookies.txt, если нужен.

    Returns:
        List[Dict[str, Any]]: Список словарей с ключами title, url, duration, channel.
    """
    return await _extractor.search_tracks(query, cookies_path=cookies_path)


async def extract_basic_info(url: str, cookies_path: Optional[str] = None) -> Dict[str, Any]:
    """Извлекает базовую информацию о ресурсе без скачивания.

    Args:
        url (str): Ссылка на ресурс.
        cookies_path (Optional[str]): Путь к cookies.txt при необходимости.

    Returns:
        Dict[str, Any]: Словарь с ключами 'title', 'duration', 'channel', 'thumbnail'.
    """
    return await _extractor.extract_basic_info(url, cookies_path=cookies_path)


async def download_media_to_temp(
    url: str, mode: str, cookies_path: Optional[str] = None
) -> List[Tuple[str, Optional[str]]]:
    """Скачивает медиа и подготавливает миниатюры во временные директории.

    Args:
        url (str): Ссылка на ресурс.
        mode (str): Режим ('audio'|'video'|'video_nosound').
        cookies_path (Optional[str]): Путь к cookies.txt.

    Returns:
        List[Tuple[str, Optional[str]]]: Список пар (путь_к_медиа, путь_к_миниатюре_or_None).

    Raises:
        FileTooLargeError: Если скачанный файл превышает MAX_FILE_BYTES.
    """
    return await _downloader.download_media_to_temp(url, mode=mode, cookies_path=cookies_path)


async def find_server_cookies_for_url(url: str) -> Optional[str]:
    """
    Ищет подходящий файл cookies для данного URL в SERVER_COOKIES_DIR.

    Args:
        url (str): URL ресурса.

    Returns:
        Optional[str]: Путь к файлу cookies или None.
    """
    return await _extractor.find_server_cookies_for_url(url)
