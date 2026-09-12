"""Centralized logging module for Sentinel AI application."""

import logging
import logging.handlers
import time
from functools import wraps
from typing import Any, Callable, ParamSpec, TypeVar

from app.core.config import get_settings

# Constants
LOG_FORMAT = "%(asctime)s | %(levelname)s | [%(process)d] | [%(threadName)s] | %(name)s | %(message)s"
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
BACKUP_COUNT = 5

# Setup UTC timestamping globally
logging.Formatter.converter = time.gmtime

P = ParamSpec("P")
R = TypeVar("R")

settings = get_settings()

# Ensure the log directory exists
if settings.ENABLE_FILE_LOGGING:
    settings.LOG_DIR.mkdir(parents=True, exist_ok=True)


class ColoredFormatter(logging.Formatter):
    """Formatter that adds colors to console logs based on level."""

    COLORS = {
        "DEBUG": "\033[94m",
        "INFO": "\033[92m",
        "WARNING": "\033[93m",
        "ERROR": "\033[91m",
        "CRITICAL": "\033[95m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with colors."""
        log_message = super().format(record)
        level_name = record.levelname
        color = self.COLORS.get(level_name, "")
        return f"{color}{log_message}{self.RESET}"


def get_logger(name: str) -> logging.Logger:
    """Get or create a configured logger instance.

    Args:
        name: The name of the logger (usually __name__).

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(settings.LOG_LEVEL)
    logger.propagate = False

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(settings.LOG_LEVEL)
    if settings.DEBUG:
        console_handler.setFormatter(ColoredFormatter(LOG_FORMAT))
    else:
        console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(console_handler)

    # File Handler
    if settings.ENABLE_FILE_LOGGING:
        file_handler = logging.handlers.RotatingFileHandler(
            settings.log_file_path,
            maxBytes=MAX_FILE_SIZE_BYTES,
            backupCount=BACKUP_COUNT,
            encoding="utf-8",
            delay=True,
        )
        file_handler.setLevel(settings.LOG_LEVEL)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(file_handler)

    return logger


def log_exception(logger: logging.Logger, message: str, exc: Exception) -> None:
    """Log an exception with context and full traceback.

    Args:
        logger: The logger instance to use.
        message: A contextual error message.
        exc: The exception instance that occurred.
    """
    logger.exception(f"{message}: {str(exc)}", exc_info=exc)


def log_execution_time(func: Callable[P, R]) -> Callable[P, R]:
    """Decorator to log function execution time at DEBUG level.

    Args:
        func: The function to be decorated.

    Returns:
        Callable: The wrapped function.
    """
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        logger = get_logger(func.__module__)
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        duration = time.perf_counter() - start_time
        logger.debug("Function %s.%s executed in %.4f s", func.__module__, func.__name__, duration)
        return result

    return wrapper
