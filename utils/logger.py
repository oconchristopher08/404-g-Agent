"""
Configures the global Loguru logger with file rotation and log level from settings.
"""

import sys
from pathlib import Path
from loguru import logger


def setup_logger(log_level: str = "INFO"):
    # Ensure the logs directory exists before Loguru tries to open the file.
    # Without this, setup_logger raises FileNotFoundError in fresh environments.
    Path("logs").mkdir(exist_ok=True)

    logger.remove()
    logger.add(sys.stdout, level=log_level, colorize=True,
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | {message}")
    logger.add("logs/agent.log", rotation="10 MB", retention="7 days",
               level=log_level, encoding="utf-8")
    logger.info("Logger initialized at level: {}", log_level)
