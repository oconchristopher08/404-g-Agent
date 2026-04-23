"""
Configures the global Loguru logger with file rotation and log level from settings.
"""

import sys
from loguru import logger


def setup_logger(log_level: str = "INFO"):
    logger.remove()
    logger.add(sys.stdout, level=log_level, colorize=True,
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | {message}")
    logger.add("logs/agent.log", rotation="10 MB", retention="7 days",
               level=log_level, encoding="utf-8")
    logger.info("Logger initialized at level: {}", log_level)
