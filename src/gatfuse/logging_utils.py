"""Logging configuration."""

import logging
import sys

LOG_FORMAT = "[%(asctime)s] %(levelname)-7s %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

LEVELS = ("debug", "info", "warning", "error")


def configure_logging(level="info"):
    """Install a stderr handler on the ``gatfuse`` logger hierarchy."""
    logger = logging.getLogger("gatfuse")
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper()))
    logger.propagate = False
    return logger


def get_logger(name):
    """Return a child logger of the ``gatfuse`` hierarchy."""
    return logging.getLogger(f"gatfuse.{name}")
