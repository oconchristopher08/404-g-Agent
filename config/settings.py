"""
Central settings loaded from environment variables via .env
"""

import os
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


def _get_int(name: str, default: int) -> int:
    """Read an integer env var, falling back to *default* on invalid input."""
    raw = os.getenv(name, str(default))
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid value for {}: '{}' — using default {}", name, raw, default)
        return default


def _get_float(name: str, default: float) -> float:
    """Read a float env var, falling back to *default* on invalid input."""
    raw = os.getenv(name, str(default))
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid value for {}: '{}' — using default {}", name, raw, default)
        return default


class Settings:
    # Twitter / Social
    TWITTER_API_KEY: str = os.getenv("TWITTER_API_KEY", "")
    TWITTER_API_SECRET: str = os.getenv("TWITTER_API_SECRET", "")
    TWITTER_ACCESS_TOKEN: str = os.getenv("TWITTER_ACCESS_TOKEN", "")
    TWITTER_ACCESS_SECRET: str = os.getenv("TWITTER_ACCESS_SECRET", "")

    # Telegram
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")

    # Blockchain
    WEB3_PROVIDER_URL: str = os.getenv("WEB3_PROVIDER_URL", "")
    ETHERSCAN_API_KEY: str = os.getenv("ETHERSCAN_API_KEY", "")
    BSC_RPC_URL: str = os.getenv("BSC_RPC_URL", "")

    # DEX / Token Data
    DEXSCREENER_API_URL: str = os.getenv(
        "DEXSCREENER_API_URL", "https://api.dexscreener.com/latest"
    )
    COINGECKO_API_KEY: str = os.getenv("COINGECKO_API_KEY", "")

    # Redis
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # Agent tuning — use safe helpers so a malformed env var doesn't crash at
    # import time before any error handling is active.
    SCAN_INTERVAL_SECONDS: int = _get_int("SCAN_INTERVAL_SECONDS", 60)
    SENTIMENT_THRESHOLD: float = _get_float("SENTIMENT_THRESHOLD", 0.6)
    WHALE_WALLET_MIN_USD: float = _get_float("WHALE_WALLET_MIN_USD", 100000.0)
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
