from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def configure_logging(level: str = "INFO", log_path: str | None = None) -> None:
    logger.remove()
    logger.add(sys.stderr, level=level, colorize=False, enqueue=False)
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        logger.add(log_path, level=level, rotation="10 MB", enqueue=False)


__all__ = ["configure_logging", "logger"]
