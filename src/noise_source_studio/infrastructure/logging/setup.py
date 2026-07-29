"""Console and rotating-file logging configuration."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOGGER_NAME = "noise_source_studio"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def configure_logging(log_directory: Path, level: str = "INFO") -> Path:
    """Configure the application logger and return the active log file."""
    log_directory.mkdir(parents=True, exist_ok=True)
    log_file = log_directory / "noise-source-studio.log"
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(_resolve_level(level))
    logger.propagate = False

    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter(LOG_FORMAT)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return log_file


def _resolve_level(level: str) -> int:
    resolved = getattr(logging, level.upper(), logging.INFO)
    return resolved if isinstance(resolved, int) else logging.INFO
