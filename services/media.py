import io
import logging
import os
from typing import List, Optional, Set

from PIL import Image, ImageOps
from PIL.Image import Resampling

from config import (
    AUDIO_EXTS,
    VIDEO_EXTS,
    IMAGE_EXTS,
    THUMB_SIZE,
    THUMB_MAX_BYTES
)


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


def process_thumbnail(src_path: str, out_dir: str) -> Optional[str]:
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
            im = ImageOps.fit(
                im, THUMB_SIZE, method=Resampling.LANCZOS
            )
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
                    logging.getLogger("bot").info(
                        "Подготовлена обложка %s (%dx%d, %d байт, quality=%d)",
                        out_path,
                        THUMB_SIZE[0],
                        THUMB_SIZE[1],
                        size,
                        quality,
                    )
                    return out_path
                last_size = size
                quality -= step
            logging.getLogger("bot").warning(
                "Не удалось сжать обложку до %d байт, пропускаю (минимальное качество %d, размер %d байт)",
                THUMB_MAX_BYTES,
                min_q,
                last_size or -1,
            )
            return None
    except Exception as e:
        logging.getLogger("bot").warning(
            "Не удалось обработать обложку %s: %s", src_path, e
        )
        return None
