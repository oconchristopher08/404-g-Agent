"""
WalletMonitor — tracks large wallet movements on Ethereum and BSC
to detect whale activity before it hits the market.
"""

import aiohttp
from loguru import logger
from config.settings import Settings

ETHERSCAN_TX_URL = "https://api.etherscan.io/api"


class WalletMonitor:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.known_whales: list[str] = []  # Populate from a watchlist file

    async def scan(self) -> list:
        """Scan recent large transactions and return wallet signals."""
        signals = []
        if not self.settings.ETHERSCAN_API_KEY:
            logger.warning("ETHERSCAN_API_KEY not set — skipping wallet scan.")
            return signals

        for wallet in self.known_whales:
            try:
                txs = await self._fetch_recent_txs(wallet)
                for tx in txs:
                    value_eth = int(tx.get("value", 0)) / 1e18
                    value_usd = value_eth * 3000  # rough ETH price estimate
                    if value_usd >= self.settings.WHALE_WALLET_MIN_USD:
                        signals.append({
                            "source": "onchain",
                            "wallet": wallet,
                            "value_usd": round(value_usd, 2),
                            "tx_hash": tx.get("hash"),
                            "confidence": min(value_usd / 1_000_000, 1.0),
                        })
            except Exception as e:
                logger.error("Wallet monitor error for {}: {}", wallet, e)

        logger.info("OnChain: {} whale signals found.", len(signals))
        return signals

    async def _fetch_recent_txs(self, address: str) -> list:
        params = {
            "module": "account",
            "action": "txlist",
            "address": address,
            "sort": "desc",
            "apikey": self.settings.ETHERSCAN_API_KEY,
            "offset": 10,
            "page": 1,
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(ETHERSCAN_TX_URL, params=params) as resp:
                data = await resp.json()
                return data.get("result", [])
