"""
NansenScanner — surfaces alpha from Nansen's smart money data.

Signals produced:
  - Smart money netflow accumulation: tokens where smart traders / funds
    have net-bought significantly in the last 1h and 24h.
  - Smart money netflow distribution: tokens being sold by smart money
    (useful for risk management / short signals).
  - Token screener hits: tokens flagged by Nansen's screener as having
    unusual smart money activity, new launches with fund interest, or
    tokens approaching exchange listing criteria.

Nansen API docs: https://docs.nansen.ai
Requires a paid Nansen API key (NANSEN_API_KEY in .env).
"""

import aiohttp
from loguru import logger
from config.settings import Settings

_NANSEN_BASE = "https://api.nansen.ai/api/v1"

# Chains to query — Nansen supports these in smart money endpoints
_CHAINS = ["ethereum", "base", "arbitrum", "solana", "bnb"]

# Minimum absolute net flow (USD) to be considered a meaningful signal
_MIN_NETFLOW_USD = 500_000

# Minimum number of distinct smart money traders to filter noise
_MIN_TRADER_COUNT = 3


class NansenScanner:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._enabled = bool(settings.NANSEN_API_KEY)
        if not self._enabled:
            logger.warning(
                "NANSEN_API_KEY not set — Nansen smart money module disabled. "
                "Get a key at https://app.nansen.ai"
            )

    async def scan(self) -> list:
        """Fetch smart money netflows and return accumulation/distribution signals."""
        if not self._enabled:
            return []

        signals: list[dict] = []
        try:
            import asyncio
            netflow_task = self._fetch_netflows()
            screener_task = self._fetch_token_screener()
            netflows, screener_hits = await asyncio.gather(
                netflow_task, screener_task, return_exceptions=True
            )

            if isinstance(netflows, Exception):
                logger.warning("Nansen netflows fetch failed: {}", netflows)
                netflows = []
            if isinstance(screener_hits, Exception):
                logger.warning("Nansen screener fetch failed: {}", screener_hits)
                screener_hits = []

            signals += self._score_netflows(netflows)
            signals += self._score_screener(screener_hits)

        except Exception as e:
            logger.error("NansenScanner error: {}", e)

        logger.info("Nansen: {} signals found.", len(signals))
        return signals

    async def _fetch_netflows(self) -> list:
        """
        Fetch tokens with the highest net smart money inflow across all
        configured chains, sorted by 1h flow to catch early moves.
        """
        payload = {
            "chains": _CHAINS,
            "filters": {
                "include_stablecoins": False,
                "include_native_tokens": False,
                "trader_count": {"min": _MIN_TRADER_COUNT},
            },
            "order_by": [{"field": "net_flow_1h_usd", "direction": "DESC"}],
            "pagination": {"page": 1, "per_page": 50},
        }
        data = await self._post("/smart-money/netflow", payload)
        return data.get("data", []) if isinstance(data, dict) else []

    async def _fetch_token_screener(self) -> list:
        """
        Fetch tokens flagged by Nansen's screener — new launches with smart
        money interest, tokens with unusual holder concentration changes, etc.
        """
        payload = {
            "chains": _CHAINS,
            "filters": {
                "include_stablecoins": False,
            },
            "order_by": [{"field": "smart_money_count", "direction": "DESC"}],
            "pagination": {"page": 1, "per_page": 30},
        }
        try:
            data = await self._post("/token-god-mode/token-screener", payload)
            return data.get("data", []) if isinstance(data, dict) else []
        except Exception as e:
            # Token screener may require a higher tier — degrade gracefully
            logger.debug("Nansen token screener unavailable: {}", e)
            return []

    def _score_netflows(self, records: list) -> list[dict]:
        """Convert Nansen netflow records into agent signals."""
        signals: list[dict] = []
        threshold = self.settings.SENTIMENT_THRESHOLD

        for rec in records:
            flow_1h = float(rec.get("net_flow_1h_usd", 0) or 0)
            flow_24h = float(rec.get("net_flow_24h_usd", 0) or 0)
            flow_7d = float(rec.get("net_flow_7d_usd", 0) or 0)
            trader_count = int(rec.get("trader_count", 0) or 0)
            symbol = rec.get("token_symbol", "UNKNOWN")
            chain = rec.get("chain", "unknown")
            token_address = rec.get("token_address", "")
            token_age_days = rec.get("token_age_days")
            market_cap = rec.get("market_cap_usd")

            # Skip if flows are too small to be meaningful
            if abs(flow_1h) < _MIN_NETFLOW_USD and abs(flow_24h) < _MIN_NETFLOW_USD:
                continue

            # Confidence: based on flow magnitude and trader count
            # Normalise against $10M as a "strong" signal
            flow_score = min(max(abs(flow_1h), abs(flow_24h)) / 10_000_000, 0.7)
            trader_score = min(trader_count / 20, 0.3)
            confidence = round(flow_score + trader_score, 4)

            if confidence < threshold:
                continue

            direction = "accumulation" if flow_1h > 0 or flow_24h > 0 else "distribution"
            signal_type = "smart_money_accumulation" if direction == "accumulation" else "smart_money_distribution"

            signals.append({
                "source": "nansen",
                "type": signal_type,
                "token": symbol,
                "chain": chain,
                "token_address": token_address,
                "net_flow_1h_usd": round(flow_1h, 0),
                "net_flow_24h_usd": round(flow_24h, 0),
                "net_flow_7d_usd": round(flow_7d, 0),
                "smart_money_traders": trader_count,
                "token_age_days": token_age_days,
                "market_cap_usd": market_cap,
                "confidence": confidence,
                "signal": (
                    f"Smart money {direction}: "
                    f"${abs(flow_1h)/1e3:.0f}k/1h, "
                    f"${abs(flow_24h)/1e3:.0f}k/24h "
                    f"({trader_count} traders)"
                ),
            })

        return signals

    def _score_screener(self, records: list) -> list[dict]:
        """Convert Nansen token screener records into agent signals."""
        signals: list[dict] = []
        threshold = self.settings.SENTIMENT_THRESHOLD

        for rec in records:
            symbol = rec.get("token_symbol") or rec.get("symbol", "UNKNOWN")
            chain = rec.get("chain", "unknown")
            smart_money_count = int(rec.get("smart_money_count", 0) or 0)
            token_age_days = rec.get("token_age_days")
            market_cap = rec.get("market_cap_usd")
            token_address = rec.get("token_address", "")

            if smart_money_count < _MIN_TRADER_COUNT:
                continue

            # New token (<30 days) with smart money interest = early alpha
            freshness_bonus = 0.15 if token_age_days is not None and token_age_days < 30 else 0.0
            confidence = round(min(smart_money_count / 30, 0.85) + freshness_bonus, 4)

            if confidence < threshold:
                continue

            age_str = f"{token_age_days}d old" if token_age_days is not None else "age unknown"
            signals.append({
                "source": "nansen",
                "type": "screener_hit",
                "token": symbol,
                "chain": chain,
                "token_address": token_address,
                "smart_money_traders": smart_money_count,
                "token_age_days": token_age_days,
                "market_cap_usd": market_cap,
                "confidence": confidence,
                "signal": f"Nansen screener: {smart_money_count} smart money wallets ({age_str})",
            })

        return signals

    async def _post(self, path: str, payload: dict) -> dict:
        url = f"{_NANSEN_BASE}{path}"
        headers = {
            "Content-Type": "application/json",
            "apikey": self.settings.NANSEN_API_KEY,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status == 402:
                    logger.warning("Nansen: credit limit reached or subscription required for {}", path)
                    return {}
                if resp.status == 403:
                    logger.warning("Nansen: API key invalid or insufficient tier for {}", path)
                    return {}
                resp.raise_for_status()
                return await resp.json()
