import logging

from sarm.config.sarm_config import SarmConfig


def setup_logger(config: SarmConfig, logger: logging.Logger):
    """Setup logger with custom format and level from config.

    Args:
        config: SarmConfig containing logging configuration
        logger: Logger to setup
    """
    # Get log level from config
    log_level = getattr(logging, config.logging_config.level.upper(), logging.INFO)

    # Create formatter with custom format
    formatter = logging.Formatter(fmt=config.logging_config.format, datefmt=config.logging_config.date_format)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add console handler with custom format
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    logger.info(f"Logger initialized with level: {config.logging_config.level}")
    logger.debug("Debug logging enabled")
