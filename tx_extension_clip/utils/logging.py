"""Logging utilities for flexible Flask/standalone logging."""

import logging

# Configure basic logging for the tx_extension_clip package
# Logs will propagate to parent application (e.g., HMA/OpenMediaMatch/Flask)
_logger = logging.getLogger("tx_extension_clip")
_logger.setLevel(logging.INFO)
# propagate = True by default, so logs reach HMA's logging system


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
        # Use the configured tx_extension_clip logger if no specific logger requested
        if logger_name:
            logger = logging.getLogger(logger_name)
        else:
            logger = logging.getLogger("tx_extension_clip")
        logger.info(message)
