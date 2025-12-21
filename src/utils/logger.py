"""
Logging configuration for the Agentic Data Product Builder.

Uses loguru for structured, colorful logging with file rotation.
"""

import sys
from pathlib import Path

from loguru import logger

from src.utils.config import settings, PROJECT_ROOT


def setup_logging(
    log_level: str = None,
    log_to_file: bool = True,
    log_dir: Path = None,
) -> None:
    """
    Set up application logging.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_to_file: Whether to write logs to file
        log_dir: Directory for log files
    """
    # Remove default handler
    logger.remove()
    
    # Use settings if not provided
    level = log_level or settings.app.log_level
    
    # Add console handler with colors
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        level=level,
        colorize=True,
    )
    
    # Add file handler if requested
    if log_to_file:
        log_path = log_dir or (PROJECT_ROOT / "logs")
        log_path.mkdir(exist_ok=True)
        
        logger.add(
            log_path / "app_{time:YYYY-MM-DD}.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            level=level,
            rotation="1 day",
            retention="7 days",
            compression="zip",
        )


def get_logger(name: str = None):
    """
    Get a logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    if name:
        return logger.bind(name=name)
    return logger


# Set up logging on module import
setup_logging()

