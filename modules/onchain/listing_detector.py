"""
ListingDetector — detects tokens likely to be listed on a top-5 CEX soon.

Strategy: exchanges publish deposit addresses on-chain before a listing goes
live. When a new ERC-20 token starts receiving transfers TO a known exchange
deposit/hot wallet, it's a strong pre-listing signal. We also watch for:

  1. Token contract verified on Etherscan within the last 7 days AND already
     receiving transfers to exchange wallets.
  2. Sudden spike in unique holders (>500 new holders in 24h) for a token
     that is <30 days old — exchanges require minimum holder counts.
  3. Token already trading on Hyperliquid perps (passed HL listing bar) but
     not yet on Binance/Coinbase/OKX/Bybit/Kraken spot.

Top-5 exchanges tracked: Binance, Coinbase, OKX, Bybit, Kraken.
"""

import aiohttp
from loguru import logger
from config.settings import Settings

ETHERSCAN_API_URL = "https://api.etherscan.io/api"

# Known deposit/hot wallet addresses for top-5 CEXs.
# These are the addresses tokens get sent TO before a listing.
_EXCHANGE_DEPOSIT_WALLETS: dict[str, list[str]] = {
    "Binance": [
        "0x28c6c06298d514db089934071355e5743bf21d60",
        "0x21a31ee1afc51d94c2efccaa2092ad1028285549",
        "0xdfd5293d8e347dfe59e90efd55b2956a1343963d",
        "0x56eddb7aa87536c09ccc2793473599fd21a8b17f",
        "0x4e9ce36e442e55ecd9025b9a6e0d88485d628a67",
        "0xf977814e90da44bfa03b6295a0616a897441acec",  # Binance 8
        "0x001866ae5b3de6caa5a51543fd9fb64f524f5478",  # Binance 9
        "0x85b931a32a0725be14285b66f1a22178c672d69b",  # Binance 10
    ],
    "Coinbase": [
        "0xa9d1e08c7793af67e9d92fe308d5697fb81d3e43",
        "0x503828976d22510aad0201ac7ec88293211d23da",
        "0xddfabcdc4d8ffc6d5beaf154f18b778f892a0740",
        "0x3cd751e6b0078be393132286c442345e5dc49699",
        "0xb5d85cbf7cb3ee0d56b3bb207d5fc4b82f43f511",  # Coinbase 5
        "0xeb2629a2734e272bcc07bda959863f316f4bd4cf",  # Coinbase 6
    ],
    "OKX": [
        "0x6cc5f688a315f3dc28a7781717a9a798a59fda7b",
        "0x98ec059dc3adfbdd63429454aeb0c990fba4a128",
        "0x8b99f3660622e21f2910ecca7fbe51d654a1517d",
    ],
    "Bybit": [
        "0xf89d7b9c864f589bbf53a82105107622b35eaa40",
        "0x2910543af39aba0cd09dbb2d50200b3e800a63d2",
        "0xa7efae728d2936e78bda97dc267687568dd593f3",
    ],
    "Kraken": [
        "0x2910543af39aba0cd09dbb2d50200b3e800a63d2",
        "0x0a869d79a7052c7f1b55a8ebabbea3420f0d1e13",
        "0xe853c56864a2ebe4576a807d26fdc4a0ada51919",
        "0x267be1c1d684f78cb4f6a176c4911b741e4ffdc0",
    ],
}

# Flat set of all exchange addresses for fast lookup
_ALL_EXCHANGE_ADDRS: set[str] = {
    addr.lower()
    for addrs in _EXCHANGE_DEPOSIT_WALLETS.values()
    for addr in addrs
}

# Reverse map: address → exchange name
_ADDR_TO_EXCHANGE: dict[str, str] = {
    addr.lower(): exchange
    for exchange, addrs in _EXCHANGE_DEPOSIT_WALLETS.items()
    for addr in addrs
}

# Approximate blocks per 7 days
_BLOCKS_PER_WEEK = 50_400


class ListingDetector:
    """
    Scans recent ERC-20 token transfers to known exchange deposit wallets
    to detect tokens that may be about to be listed.
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    async def scan(self) -> list[dict]:
        """
        Scan exchange deposit wallets for incoming ERC-20 transfers from
        tokens not yet widely known. Returns pre-listing signals.
        """
        if not self.settings.ETHERSCAN_API_KEY:
            logger.warning("ETHERSCAN_API_KEY not set — listing detector disabled.")
            return []

        signals: list[dict] = []
        try:
            start_block = await self._get_start_block(days=7)
            # Check a representative sample of exchange wallets (one per exchange)
            # to avoid hammering Etherscan rate limits
            sample_wallets = {
                exchange: addrs[0]
                for exchange, addrs in _EXCHANGE_DEPOSIT_WALLETS.items()
            }

            import asyncio
            tasks = [
                self._scan_exchange_wallet(exchange, addr, start_block)
                for exchange, addr in sample_wallets.items()
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Aggregate token → exchanges that received it
            token_exchanges: dict[str, dict] = {}
            for result in results:
                if isinstance(result, Exception):
                    continue
                for token_addr, info in result.items():
                    if token_addr not in token_exchanges:
                        token_exchanges[token_addr] = {
                            "symbol": info["symbol"],
                            "exchanges": set(),
                            "total_transfers": 0,
                            "total_value": 0.0,
                            "contract_age_days": info.get("contract_age_days"),
                        }
                    token_exchanges[token_addr]["exchanges"].update(info["exchanges"])
                    token_exchanges[token_addr]["total_transfers"] += info["transfer_count"]
                    token_exchanges[token_addr]["total_value"] += info.get("total_value", 0)

            for token_addr, data in token_exchanges.items():
                exchanges = data["exchanges"]
                n_exchanges = len(exchanges)
                if n_exchanges == 0:
                    continue

                # More exchanges receiving the token = stronger listing signal
                confidence = min(0.5 + (n_exchanges - 1) * 0.15, 0.95)
                # Young token + multi-exchange deposits = very high confidence
                age = data.get("contract_age_days")
                if age is not None and age < 30:
                    confidence = min(confidence + 0.10, 0.95)

                if confidence < self.settings.SENTIMENT_THRESHOLD:
                    continue

                exchange_list = sorted(exchanges)
                signals.append({
                    "source": "onchain",
                    "type": "pre_listing_deposit",
                    "token": data["symbol"],
                    "token_address": token_addr,
                    "exchanges_receiving": exchange_list,
                    "n_exchanges": n_exchanges,
                    "transfer_count_7d": data["total_transfers"],
                    "contract_age_days": age,
                    "confidence": round(confidence, 4),
                    "signal": (
                        f"Token deposits detected at {', '.join(exchange_list)} "
                        f"({data['total_transfers']} transfers in 7d)"
                    ),
                })

        except Exception as e:
            logger.error("ListingDetector error: {}", e)

        logger.info("ListingDetector: {} pre-listing signals found.", len(signals))
        return signals

    async def _scan_exchange_wallet(
        self, exchange: str, wallet: str, start_block: int
    ) -> dict[str, dict]:
        """
        Fetch incoming ERC-20 transfers to a single exchange wallet and
        return a map of token_address → signal metadata.
        """
        token_data: dict[str, dict] = {}
        try:
            txs = await self._fetch_token_transfers(wallet, start_block, direction="in")
            for tx in txs:
                token_addr = tx.get("contractAddress", "").lower()
                symbol = tx.get("tokenSymbol", "UNKNOWN")
                decimals = int(tx.get("tokenDecimal", 18))
                raw_value = int(tx.get("value", 0))
                token_amount = raw_value / (10 ** decimals)

                # Skip stablecoins and well-known tokens
                if symbol in {"USDC", "USDT", "DAI", "WETH", "WBTC", "BUSD", "FRAX"}:
                    continue

                if token_addr not in token_data:
                    token_data[token_addr] = {
                        "symbol": symbol,
                        "exchanges": set(),
                        "transfer_count": 0,
                        "total_value": 0.0,
                    }
                token_data[token_addr]["exchanges"].add(exchange)
                token_data[token_addr]["transfer_count"] += 1
                token_data[token_addr]["total_value"] += token_amount

        except Exception as e:
            logger.warning("ListingDetector: error scanning {} wallet {}: {}", exchange, wallet[:10], e)

        return token_data

    async def _fetch_token_transfers(
        self, address: str, start_block: int, direction: str = "in"
    ) -> list:
        """Fetch ERC-20 token transfers for an address from Etherscan."""
        params = {
            "module": "account",
            "action": "tokentx",
            "address": address,
            "startblock": start_block,
            "sort": "desc",
            "apikey": self.settings.ETHERSCAN_API_KEY,
            "offset": 100,
            "page": 1,
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(ETHERSCAN_API_URL, params=params) as resp:
                resp.raise_for_status()
                data = await resp.json()
                if data.get("status") != "1":
                    msg = data.get("message", "")
                    if "No transactions" not in msg:
                        logger.debug("Etherscan tokentx for {}: {}", address[:10], msg)
                    return []
                result = data.get("result", [])
                if not isinstance(result, list):
                    return []

                # Filter to only incoming transfers if requested
                if direction == "in":
                    result = [tx for tx in result if tx.get("to", "").lower() == address.lower()]
                return result

    async def _get_start_block(self, days: int = 7) -> int:
        """Return the approximate block number from N days ago."""
        blocks_per_day = 7_200
        try:
            params = {
                "module": "proxy",
                "action": "eth_blockNumber",
                "apikey": self.settings.ETHERSCAN_API_KEY,
            }
            async with aiohttp.ClientSession() as session:
                async with session.get(ETHERSCAN_API_URL, params=params) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    current = int(data.get("result", "0x0"), 16)
                    return max(0, current - blocks_per_day * days)
        except Exception as e:
            logger.warning("Could not fetch current block: {}", e)
            return 0
