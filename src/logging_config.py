# logging_config.py - Centralized logging configuration
"""
Centralized logging configuration for the AI Stock Prediction application.

Usage:
    from logging_config import setup_logging
    setup_logging()  # Call once at application start
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_to_console: bool = True
) -> None:
    """
    Configure logging for the application.

    Args:
        level: Logging level (default: INFO)
        log_file: Optional path to log file. If None, logs only to console.
        log_to_console: Whether to log to console (default: True)
    """
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        try:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        except Exception as e:
            root_logger.warning(f"Could not create log file: {e}")

    # Suppress noisy third-party loggers
    logging.getLogger('yfinance').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('tensorflow').setLevel(logging.WARNING)

    root_logger.info("Logging configured successfully")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured Logger instance
    """
    return logging.getLogger(name)
