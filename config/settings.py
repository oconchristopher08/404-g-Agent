"""
Central settings loaded from environment variables via .env.
All attributes are instance-level so env vars can be patched in tests.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    def __init__(self):
        # ── Twitter / Social ──────────────────────────────────────────
        # v2 API — only bearer token is needed for read-only search
        self.TWITTER_BEARER_TOKEN: str = os.getenv("TWITTER_BEARER_TOKEN", "")
        # Legacy v1.1 keys kept for reference but no longer used
        self.TWITTER_API_KEY: str = os.getenv("TWITTER_API_KEY", "")
        self.TWITTER_API_SECRET: str = os.getenv("TWITTER_API_SECRET", "")
        self.TWITTER_ACCESS_TOKEN: str = os.getenv("TWITTER_ACCESS_TOKEN", "")
        self.TWITTER_ACCESS_SECRET: str = os.getenv("TWITTER_ACCESS_SECRET", "")

        # ── Telegram ──────────────────────────────────────────────────
        self.TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

        # ── Blockchain ────────────────────────────────────────────────
        self.WEB3_PROVIDER_URL: str = os.getenv("WEB3_PROVIDER_URL", "")
        self.ETHERSCAN_API_KEY: str = os.getenv("ETHERSCAN_API_KEY", "")
        self.BSC_RPC_URL: str = os.getenv("BSC_RPC_URL", "")

        # ── DEX / Token Data ──────────────────────────────────────────
        self.DEXSCREENER_API_URL: str = os.getenv(
            "DEXSCREENER_API_URL", "https://api.dexscreener.com/latest"
        )
        self.COINGECKO_API_KEY: str = os.getenv("COINGECKO_API_KEY", "")

        # ── Redis ─────────────────────────────────────────────────────
        self.REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

        # ── Whale watchlist ───────────────────────────────────────────
        self.WHALE_WATCHLIST_PATH: str = os.getenv(
            "WHALE_WATCHLIST_PATH", "data/whales.json"
        )

        # ── Price oracle ──────────────────────────────────────────────
        # How long (seconds) to cache fetched token prices before refreshing
        self.ETH_PRICE_TTL_SECONDS: int = int(os.getenv("ETH_PRICE_TTL_SECONDS", "300"))

        # ── Agent tuning ──────────────────────────────────────────────
        self.SCAN_INTERVAL_SECONDS: int = int(os.getenv("SCAN_INTERVAL_SECONDS", "60"))
        self.SENTIMENT_THRESHOLD: float = float(os.getenv("SENTIMENT_THRESHOLD", "0.6"))
        self.WHALE_WALLET_MIN_USD: float = float(os.getenv("WHALE_WALLET_MIN_USD", "100000"))
        self.LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

        # ── Prometheus ────────────────────────────────────────────────
        self.METRICS_PORT: int = int(os.getenv("METRICS_PORT", "8000"))
