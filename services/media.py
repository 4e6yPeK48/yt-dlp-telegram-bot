import io
import os
from asyncio import to_thread
from typing import List, Optional, Set

from PIL import Image, ImageOps
from PIL.Image import Resampling

from bot.dispatcher import logger
from utils.log_helpers import log_info, log_warning, log_exception

from config import AUDIO_EXTS, VIDEO_EXTS, IMAGE_EXTS, THUMB_SIZE, THUMB_MAX_BYTES


def find_files_by_exts(root: str, exts: Set[str]) -> List[str]:
    """Находит файлы с указанными расширениями.

    Args:
        root (str): Корневая директория.
        exts (Set[str]): Набор расширений (с точкой).

    Returns:
        List[str]: Пути найденных файлов.
    """
    out: List[str] = []
    for base, _, files in os.walk(root):
        for name in files:
            if os.path.splitext(name)[1].lower() in exts:
                out.append(os.path.join(base, name))
    return out


def find_audio_files(root: str) -> List[str]:
    """Находит аудиофайлы.

    Args:
        root (str): Корень поиска.

    Returns:
        List[str]: Пути аудио.
    """
    return find_files_by_exts(root, AUDIO_EXTS)


def find_video_files(root: str) -> List[str]:
    """Находит видеофайлы.

    Args:
        root (str): Корень поиска.

    Returns:
        List[str]: Пути видео.
    """
    return find_files_by_exts(root, VIDEO_EXTS)


def find_image_files(root: str) -> List[str]:
    """Находит изображения.

    Args:
        root (str): Корень поиска.

    Returns:
        List[str]: Пути изображений.
    """
    return find_files_by_exts(root, IMAGE_EXTS)


def norm_base(path: str) -> str:
    """Возвращает имя файла без расширения и хвоста после '#'.

    Args:
        path (str): Путь к файлу.

    Returns:
        str: Базовое имя.
    """
    name = os.path.basename(path)
    name = name.split("#", 1)[0]
    base, _ = os.path.splitext(name)
    return base


def extract_id_from_base(base: str) -> Optional[str]:
    """Извлекает ID из квадратных скобок.

    Args:
        base (str): Базовое имя.

    Returns:
        Optional[str]: Извлечённый ID или None.
    """
    import re

    m = re.search(r"\[([0-9A-Za-z_-]{6,})]", base)
    return m.group(1) if m else None


def process_thumbnail_sync(src_path: str, out_dir: str) -> Optional[str]:
    """Готовит миниатюру: 320x320 JPEG ≤ заданного лимита.

    Args:
        src_path (str): Исходный файл.
        out_dir (str): Директория назначения.

    Returns:
        Optional[str]: Путь к миниатюре или None.
    """
    try:
        with Image.open(src_path) as im:
            im = im.convert("RGB")
            im = ImageOps.fit(im, THUMB_SIZE, method=Resampling.LANCZOS)
            quality = 90
            min_q = 40
            step = 5
            out_path = os.path.join(
                out_dir,
                f"{os.path.splitext(os.path.basename(src_path))[0]}_320.jpg",
            )
            last_size: Optional[int] = None
            while quality >= min_q:
                buf = io.BytesIO()
                im.save(
                    buf,
                    format="JPEG",
                    quality=quality,
                    optimize=True,
                    progressive=True,
                    subsampling="4:2:0",
                )
                size = buf.tell()
                if size <= THUMB_MAX_BYTES:
                    with open(out_path, "wb") as f:
                        f.write(buf.getvalue())
                    log_info(
                        logger,
                        f"Подготовлена обложка",
                        extra={
                            "path": out_path,
                            "width": THUMB_SIZE[0],
                            "height": THUMB_SIZE[1],
                            "size": size,
                            "quality": quality,
                        },
                    )
                    return out_path
                last_size = size
                quality -= step
            log_warning(
                logger,
                f"Не удалось сжать обложку, пропускаю",
                extra={"limit": THUMB_MAX_BYTES, "min_q": min_q, "last_size": last_size or -1},
            )
            return None
    except Exception as e:
        log_exception(
            logger,
            f"Не удалось обработать обложку",
            extra={"path": src_path, "err": str(e)},
        )
        return None


async def process_thumbnail(src_path: str, out_dir: str) -> Optional[str]:
    """
    Асинхронно готовит миниатюру: 320x320 JPEG ≤ заданного лимита.

    Args:
        src_path (str): Исходный файл.
        out_dir (str): Директория назначения.

    Returns:
        Optional[str]: Путь к миниатюре или None.
    """
    try:
        return await to_thread(process_thumbnail_sync, src_path, out_dir)
    except Exception as e:
        log_exception(
            logger,
            f"Ошибка при создании миниатюры",
            extra={"path": src_path, "err": str(e)},
        )
        return None
