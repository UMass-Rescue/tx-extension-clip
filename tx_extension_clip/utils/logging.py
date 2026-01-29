"""Logging utilities for flexible Flask/standalone logging."""

import logging


def log_info(message: str, logger_name: str = None):
    """Log message using Flask's logger if available, otherwise use module logger.
    
    Args:
        message: The message to log
        logger_name: Optional logger name to use for fallback logging
    """
    try:
        from flask import current_app
        current_app.logger.info(message)
    except (ImportError, RuntimeError):
        logger = logging.getLogger(logger_name) if logger_name else logging.getLogger()
        logger.info(message)
