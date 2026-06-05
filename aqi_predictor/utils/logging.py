from __future__ import annotations

import logging
from logging import Logger


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def get_logger(name: str) -> Logger:
    logger = logging.getLogger(name)
    if not logging.getLogger().handlers:
        configure_logging()
    return logger
