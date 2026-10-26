import logging
import sys

from .config import config


def _get_log_level_from_config() -> int:
    level_name = getattr(config, "LOG_LEVEL", "DEBUG").upper()
    level = logging.getLevelNamesMapping().get(level_name)
    if level is None:
        return logging.DEBUG
    return level


def get_logger(
    name: str = "DEFAULT_NAME",
    level: int | None = None,
    fmt: str = "%(levelname)-9s %(asctime)s | Func: %(funcName)s | Mod: %(module)s | %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> logging.Logger:

    if level is None:
        level = _get_log_level_from_config()

    logger = logging.getLogger(name=name)
    logger.setLevel(level=level)

    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level=level)
        formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
        console_handler.setFormatter(fmt=formatter)
        logger.addHandler(console_handler)

    logger.propagate = False

    return logger
