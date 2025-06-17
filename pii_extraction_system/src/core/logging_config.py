"""Logging configuration for the PII extraction system."""

import sys
from pathlib import Path
from typing import Dict, Any

from loguru import logger

from .config import settings


def setup_logging() -> None:
    """Configure logging for the application."""
    
    # Remove default handler
    logger.remove()
    
    # Console handler with colored output
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        level=settings.log_level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )
    
    # File handler for persistent logging
    log_file = Path(settings.log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        level=settings.log_level,
        rotation="10 MB",
        retention="30 days",
        compression="gz",
        backtrace=True,
        diagnose=True,
    )
    
    # Audit log for compliance
    if settings.privacy.audit_logging:
        audit_file = log_file.parent / "audit.log"
        logger.add(
            audit_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | AUDIT | {message}",
            level="INFO",
            filter=lambda record: record.get("audit", False),
            rotation="50 MB",
            retention="1 year",
            compression="gz",
        )
    
    # Error-only log file
    error_file = log_file.parent / "errors.log"
    logger.add(
        error_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        level="ERROR",
        rotation="5 MB",
        retention="90 days",
        compression="gz",
        backtrace=True,
        diagnose=True,
    )
    
    logger.info("Logging configuration initialized")


def get_logger(name: str) -> Any:
    """Get a logger instance with the given name."""
    return logger.bind(name=name)


def audit_log(message: str, **kwargs: Dict[str, Any]) -> None:
    """Log an audit message."""
    logger.bind(audit=True).info(message, **kwargs)


# Initialize logging on import
setup_logging()