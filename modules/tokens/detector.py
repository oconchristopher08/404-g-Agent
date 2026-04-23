"""
TokenDetector — monitors DexScreener for newly listed or trending tokens
with unusual volume/price activity suggesting early opportunity.
"""

import aiohttp
from loguru import logger
from config.settings import Settings


class TokenDetector:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.base_url = settings.DEXSCREENER_API_URL

    async def scan(self) -> list:
        """Fetch trending token pairs and return high-signal tokens."""
        signals = []
        try:
            pairs = await self._fetch_trending_pairs()
            for pair in pairs:
                confidence = self._score_pair(pair)
                if confidence >= self.settings.SENTIMENT_THRESHOLD:
                    signals.append({
                        "source": "dexscreener",
                        "token": pair.get("baseToken", {}).get("symbol", "UNKNOWN"),
                        "pair_address": pair.get("pairAddress"),
                        "price_usd": pair.get("priceUsd"),
                        "volume_24h": pair.get("volume", {}).get("h24"),
                        "price_change_24h": pair.get("priceChange", {}).get("h24"),
                        "confidence": confidence,
                    })
        except Exception as e:
            logger.error("Token detector error: {}", e)

        logger.info("Tokens: {} signals found.", len(signals))
        return signals

    async def _fetch_trending_pairs(self) -> list:
        url = f"{self.base_url}/dex/tokens/trending"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                data = await resp.json()
                return data.get("pairs", [])

    def _score_pair(self, pair: dict) -> float:
        """Score a token pair based on volume change and price momentum."""
        try:
            price_change = float(pair.get("priceChange", {}).get("h24", 0))
            volume = float(pair.get("volume", {}).get("h24", 0))
            score = 0.0
            if price_change > 50:
                score += 0.5
            if volume > 500_000:
                score += 0.3
            if price_change > 100:
                score += 0.2
            return min(score, 1.0)
        except Exception:
            return 0.0
