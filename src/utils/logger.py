"""Logging setup for the project."""
import logging
import sys
from typing import Optional


def setup_logger(name: str = "hotel_cancellation", level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a standardized logger.
    
    Args:
        name: Name of the logger instance.
        level: Logging level (e.g., logging.INFO, logging.DEBUG).
        
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger
