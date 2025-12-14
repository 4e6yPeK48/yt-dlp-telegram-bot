import logging
import os
from logging.handlers import TimedRotatingFileHandler


class OnlyLoggerFilter(logging.Filter):
    """Пропускает только записи выбранного логгера (по префиксу имени)."""

    def __init__(self, prefix: str) -> None:
        super().__init__()
        self.prefix = prefix

    def filter(self, record: logging.LogRecord) -> bool:
        return record.name.startswith(self.prefix)


def make_rotating(
    log_dir: str,
    path: str,
    level: int,
    fmt: logging.Formatter,
) -> TimedRotatingFileHandler:
    handler = TimedRotatingFileHandler(
        filename=os.path.join(log_dir, path),
        when="midnight",
        backupCount=7,
        encoding="utf-8",
    )
    handler.setLevel(level)
    handler.setFormatter(fmt)
    return handler


def setup_logging(log_dir: str = "logs") -> None:
    """Настраивает логирование: консольный вывод и ротацию по уровням.

    Args:
        log_dir (str): Директория для файлов логов.
    """
    os.makedirs(log_dir, exist_ok=True)
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    console.addFilter(OnlyLoggerFilter("bot"))
    root.addHandler(console)

    root.addHandler(make_rotating(log_dir, "app.debug.log", logging.DEBUG, fmt))

    info_h = make_rotating(log_dir, "app.info.log", logging.INFO, fmt)
    info_h.addFilter(OnlyLoggerFilter("bot"))
    root.addHandler(info_h)

    root.addHandler(make_rotating(log_dir, "app.warn.log", logging.WARNING, fmt))
    root.addHandler(make_rotating(log_dir, "app.error.log", logging.ERROR, fmt))

    third_party = [
        ("aiogram", "aiogram.error.log"),
        ("aiohttp", "aiohttp.error.log"),
        ("yt_dlp", "yt-dlp.error.log"),
    ]
    for name, fname in third_party:
        lg = logging.getLogger(name)
        lg.setLevel(logging.DEBUG)
        errh = TimedRotatingFileHandler(
            filename=os.path.join(log_dir, fname),
            when="midnight",
            backupCount=7,
            encoding="utf-8",
        )
        errh.setLevel(logging.ERROR)
        errh.setFormatter(fmt)
        lg.addHandler(errh)

    logging.getLogger("aiogram").setLevel(logging.INFO)
    logging.getLogger("aiohttp").setLevel(logging.INFO)
    logging.getLogger("yt_dlp").setLevel(logging.INFO)
