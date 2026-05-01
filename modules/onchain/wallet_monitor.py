"""
WalletMonitor — tracks large ETH transfers and ERC-20 token movements from
a curated whale watchlist. Uses live ETH price from PriceOracle and filters
to the last ~24 hours via Etherscan's startblock parameter.
"""

import json
import time
import aiohttp
from loguru import logger
from config.settings import Settings
from utils.price_oracle import PriceOracle

ETHERSCAN_API_URL = "https://api.etherscan.io/api"

# Approximate blocks per 24 hours on Ethereum (~12s block time)
_BLOCKS_PER_DAY = 7_200

# Known exchange/custodian hot wallets — movements from these are not alpha
_EXCHANGE_ADDRESSES = {
    "0x28c6c06298d514db089934071355e5743bf21d60",  # Binance hot wallet
    "0x21a31ee1afc51d94c2efccaa2092ad1028285549",  # Binance cold wallet
    "0xdfd5293d8e347dfe59e90efd55b2956a1343963d",  # Binance 14
    "0x56eddb7aa87536c09ccc2793473599fd21a8b17f",  # Binance 15
    "0x4e9ce36e442e55ecd9025b9a6e0d88485d628a67",  # Binance 16
    "0xa9d1e08c7793af67e9d92fe308d5697fb81d3e43",  # Coinbase
    "0x503828976d22510aad0201ac7ec88293211d23da",  # Coinbase 2
    "0xddfabcdc4d8ffc6d5beaf154f18b778f892a0740",  # Coinbase 3
    "0x3cd751e6b0078be393132286c442345e5dc49699",  # Coinbase 4
    "0x2b5634c42055806a59e9107ed44d43c426e58258",  # KuCoin
    "0xd6216fc19db775df9774a6e33526131da7d19a2c",  # KuCoin 2
}


class WalletMonitor:
    def __init__(self, settings: Settings, price_oracle: PriceOracle | None = None):
        self.settings = settings
        self.price_oracle = price_oracle or PriceOracle(
            ttl_seconds=settings.ETH_PRICE_TTL_SECONDS,
            api_key=settings.COINGECKO_API_KEY,
        )
        self.known_whales: list[str] = self._load_watchlist()

    def _load_watchlist(self) -> list[str]:
        path = self.settings.WHALE_WATCHLIST_PATH
        try:
            with open(path) as f:
                data = json.load(f)
            wallets = [w.lower() for w in data.get("wallets", [])]
            logger.info("WalletMonitor: loaded {} whale addresses from {}", len(wallets), path)
            return wallets
        except FileNotFoundError:
            logger.warning("Whale watchlist not found at '{}' — on-chain module disabled.", path)
            return []
        except Exception as e:
            logger.error("Failed to load whale watchlist: {}", e)
            return []

    async def scan(self) -> list:
        """Scan recent ETH transfers and ERC-20 movements for all watched wallets."""
        signals = []
        if not self.settings.ETHERSCAN_API_KEY:
            logger.warning("ETHERSCAN_API_KEY not set — skipping wallet scan.")
            return signals
        if not self.known_whales:
            logger.warning("Whale watchlist is empty — skipping wallet scan.")
            return signals

        eth_price = await self.price_oracle.get("ETH")
        # Approximate the block number from 24h ago to limit result set
        start_block = await self._get_start_block()

        for wallet in self.known_whales:
            if wallet in _EXCHANGE_ADDRESSES:
                continue
            try:
                eth_txs = await self._fetch_txs(wallet, action="txlist", start_block=start_block)
                erc20_txs = await self._fetch_txs(wallet, action="tokentx", start_block=start_block)

                for tx in eth_txs:
                    signal = self._evaluate_eth_tx(tx, wallet, eth_price)
                    if signal:
                        signals.append(signal)

                for tx in erc20_txs:
                    signal = self._evaluate_erc20_tx(tx, wallet, eth_price)
                    if signal:
                        signals.append(signal)

            except Exception as e:
                logger.error("Wallet monitor error for {}: {}", wallet, e)

        logger.info("OnChain: {} whale signals found.", len(signals))
        return signals

    def _evaluate_eth_tx(self, tx: dict, wallet: str, eth_price: float) -> dict | None:
        try:
            value_eth = int(tx.get("value", 0)) / 1e18
            value_usd = value_eth * eth_price
            if value_usd < self.settings.WHALE_WALLET_MIN_USD:
                return None
            # Ignore if counterparty is a known exchange
            to_addr = tx.get("to", "").lower()
            if to_addr in _EXCHANGE_ADDRESSES:
                return None
            return {
                "source": "onchain",
                "type": "eth_transfer",
                "token": "ETH",
                "wallet": wallet,
                "to": to_addr,
                "value_usd": round(value_usd, 2),
                "value_eth": round(value_eth, 4),
                "tx_hash": tx.get("hash"),
                "confidence": min(value_usd / 1_000_000, 1.0),
            }
        except Exception:
            return None

    def _evaluate_erc20_tx(self, tx: dict, wallet: str, eth_price: float) -> dict | None:
        try:
            decimals = int(tx.get("tokenDecimal", 18))
            raw_value = int(tx.get("value", 0))
            token_amount = raw_value / (10 ** decimals)
            token_symbol = tx.get("tokenSymbol", "UNKNOWN")

            # For stablecoins, value_usd ≈ token_amount
            stablecoins = {"USDC", "USDT", "DAI", "BUSD", "FRAX"}
            if token_symbol in stablecoins:
                value_usd = token_amount
            else:
                # Non-stablecoin ERC-20: flag large movements by token count
                # (price lookup would require per-token API calls; use count heuristic)
                if token_amount < 1_000:
                    return None
                value_usd = 0  # unknown USD value, flag by volume

            if value_usd > 0 and value_usd < self.settings.WHALE_WALLET_MIN_USD:
                return None

            to_addr = tx.get("to", "").lower()
            if to_addr in _EXCHANGE_ADDRESSES:
                return None

            return {
                "source": "onchain",
                "type": "erc20_transfer",
                "token": token_symbol,
                "wallet": wallet,
                "to": to_addr,
                "token_amount": round(token_amount, 2),
                "value_usd": round(value_usd, 2) if value_usd else None,
                "tx_hash": tx.get("hash"),
                "contract": tx.get("contractAddress"),
                "confidence": min(value_usd / 1_000_000, 1.0) if value_usd else 0.65,
            }
        except Exception:
            return None

    async def _get_start_block(self) -> int:
        """Fetch the current block number and subtract ~24h worth of blocks."""
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
                    current_block = int(data.get("result", "0x0"), 16)
                    return max(0, current_block - _BLOCKS_PER_DAY)
        except Exception as e:
            logger.warning("Could not fetch current block number: {}", e)
            return 0

    async def _fetch_txs(self, address: str, action: str, start_block: int = 0) -> list:
        params = {
            "module": "account",
            "action": action,
            "address": address,
            "startblock": start_block,
            "sort": "desc",
            "apikey": self.settings.ETHERSCAN_API_KEY,
            "offset": 20,
            "page": 1,
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(ETHERSCAN_API_URL, params=params) as resp:
                resp.raise_for_status()
                data = await resp.json()
                if data.get("status") != "1":
                    msg = data.get("message", "unknown")
                    # "No transactions found" is a normal empty result, not an error
                    if "No transactions" not in msg:
                        logger.warning("Etherscan {} error for {}: {}", action, address, msg)
                    return []
                result = data.get("result", [])
                if not isinstance(result, list):
                    logger.warning("Unexpected Etherscan result type for {}: {}", address, type(result))
                    return []
                return result
