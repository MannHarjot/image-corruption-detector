"""Centralized logging configuration for image-corruption-detector."""

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    fmt: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> logging.Logger:
    """Create and configure a named logger.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.
        level: Logging level (e.g. ``logging.DEBUG``, ``logging.INFO``).
        log_file: Optional path to write log output to a file in addition
            to stdout. The parent directory is created if it does not exist.
        fmt: Log record format string.
        datefmt: Date/time format string used in ``fmt``.

    Returns:
        Configured :class:`logging.Logger` instance.

    Example:
        >>> logger = get_logger(__name__, log_file=Path("outputs/logs/train.log"))
        >>> logger.info("Training started")
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers when the logger is retrieved multiple times.
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional file handler
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
