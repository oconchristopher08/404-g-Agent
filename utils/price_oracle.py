"""
PriceOracle — fetches live token prices from CoinGecko with a TTL cache.
Used by WalletMonitor to convert ETH/BNB values to USD without hardcoding.
"""

import time
import aiohttp
from loguru import logger

_COINGECKO_URL = "https://api.coingecko.com/api/v3/simple/price"

# Map internal token IDs to CoinGecko coin IDs
_COIN_IDS = {
    "ETH": "ethereum",
    "BNB": "binancecoin",
}


class PriceOracle:
    def __init__(self, ttl_seconds: int = 300, api_key: str = ""):
        self._ttl = ttl_seconds
        self._api_key = api_key
        self._cache: dict[str, float] = {}
        self._fetched_at: float = 0.0

    def _is_stale(self) -> bool:
        return (time.monotonic() - self._fetched_at) > self._ttl

    async def get_prices(self) -> dict[str, float]:
        """Return cached prices, refreshing from CoinGecko if the TTL has expired."""
        if self._cache and not self._is_stale():
            return self._cache

        try:
            params: dict = {
                "ids": ",".join(_COIN_IDS.values()),
                "vs_currencies": "usd",
            }
            headers = {}
            if self._api_key:
                headers["x-cg-demo-api-key"] = self._api_key

            async with aiohttp.ClientSession() as session:
                async with session.get(_COINGECKO_URL, params=params, headers=headers) as resp:
                    resp.raise_for_status()
                    data = await resp.json()

            prices: dict[str, float] = {}
            for symbol, coin_id in _COIN_IDS.items():
                price = data.get(coin_id, {}).get("usd")
                if price:
                    prices[symbol] = float(price)

            if prices:
                self._cache = prices
                self._fetched_at = time.monotonic()
                logger.debug("PriceOracle refreshed: {}", prices)
            else:
                logger.warning("PriceOracle: empty response from CoinGecko, using stale cache.")

        except Exception as e:
            logger.warning("PriceOracle fetch failed ({}), using stale/fallback prices.", e)
            # Fall back to last known values; if cache is empty use conservative defaults
            if not self._cache:
                self._cache = {"ETH": 3000.0, "BNB": 400.0}
                logger.warning("PriceOracle: no cache available, using hardcoded fallback prices.")

        return self._cache

    async def get(self, symbol: str) -> float:
        """Return the USD price for a single symbol (e.g. 'ETH')."""
        prices = await self.get_prices()
        return prices.get(symbol.upper(), 0.0)
