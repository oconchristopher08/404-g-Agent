"""
Central settings loaded from environment variables via .env
"""

import os
from dotenv import load_dotenv

load_dotenv()


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

    # Agent tuning
    SCAN_INTERVAL_SECONDS: int = int(os.getenv("SCAN_INTERVAL_SECONDS", "60"))
    SENTIMENT_THRESHOLD: float = float(os.getenv("SENTIMENT_THRESHOLD", "0.6"))
    WHALE_WALLET_MIN_USD: float = float(os.getenv("WHALE_WALLET_MIN_USD", "100000"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
