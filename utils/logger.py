"""
utils/logger.py
Sistema de logs profesional con tres archivos separados:
  - logs/runtime.log  → actividad general del sistema
  - logs/signals.log  → señales emitidas
  - logs/error.log    → errores y excepciones
"""

import logging
import os
from logging.handlers import RotatingFileHandler

from core.config import LOG_DIR

os.makedirs(LOG_DIR, exist_ok=True)

_FMT = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def _make_handler(filename: str, level: int = logging.DEBUG) -> RotatingFileHandler:
    path = os.path.join(LOG_DIR, filename)
    handler = RotatingFileHandler(path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_FMT, _DATE_FMT))
    return handler


def _make_console_handler(level: int = logging.INFO) -> logging.StreamHandler:
    h = logging.StreamHandler()
    h.setLevel(level)
    h.setFormatter(logging.Formatter(_FMT, _DATE_FMT))
    return h


def get_logger(name: str) -> logging.Logger:
    """Devuelve un logger que escribe en runtime.log + consola."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        logger.addHandler(_make_handler("runtime.log"))
        logger.addHandler(_make_console_handler())
    return logger


# Logger específico para señales
_signal_logger = logging.getLogger("signals")
if not _signal_logger.handlers:
    _signal_logger.setLevel(logging.INFO)
    _signal_logger.addHandler(_make_handler("signals.log"))
    _signal_logger.propagate = False


# Logger específico para errores
_error_logger = logging.getLogger("errors")
if not _error_logger.handlers:
    _error_logger.setLevel(logging.WARNING)
    _error_logger.addHandler(_make_handler("error.log", logging.WARNING))
    _error_logger.addHandler(_make_console_handler(logging.WARNING))
    _error_logger.propagate = False


def log_signal(message: str) -> None:
    _signal_logger.info(message)


def log_error(message: str, exc: Exception | None = None) -> None:
    if exc:
        _error_logger.error("%s | %s: %s", message, type(exc).__name__, exc, exc_info=True)
    else:
        _error_logger.error(message)