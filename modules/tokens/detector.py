"""
TokenDetector — monitors DexScreener for newly listed and trending tokens
across multiple chains, scoring each pair on a multi-factor heuristic that
accounts for price momentum, volume, liquidity health, token age, and
buy/sell pressure.
"""

import time
import aiohttp
from loguru import logger
from config.settings import Settings

# Chains to scan. DexScreener supports filtering by chainId in search.
_CHAINS = ["ethereum", "solana", "bsc", "arbitrum", "base"]

# DexScreener endpoints
_SEARCH_URL = "https://api.dexscreener.com/latest/dex/search"
_BOOSTS_URL = "https://api.dexscreener.com/token-boosts/top/v1"
_PROFILES_URL = "https://api.dexscreener.com/token-profiles/latest/v1"

# 48 hours in milliseconds — pairs newer than this get a freshness bonus
_NEW_PAIR_THRESHOLD_MS = 48 * 60 * 60 * 1000


class TokenDetector:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def scan(self) -> list:
        """Fetch trending and newly boosted pairs across all chains and score them."""
        signals = []
        try:
            pairs = await self._fetch_all_pairs()
            seen_addresses: set[str] = set()

            for pair in pairs:
                addr = pair.get("pairAddress", "")
                if addr in seen_addresses:
                    continue
                seen_addresses.add(addr)

                score, reasons = self._score_pair(pair)
                if score >= self.settings.SENTIMENT_THRESHOLD:
                    signals.append({
                        "source": "dexscreener",
                        "token": pair.get("baseToken", {}).get("symbol", "UNKNOWN"),
                        "chain": pair.get("chainId", "unknown"),
                        "pair_address": addr,
                        "price_usd": pair.get("priceUsd"),
                        "volume_24h": pair.get("volume", {}).get("h24"),
                        "price_change_24h": pair.get("priceChange", {}).get("h24"),
                        "liquidity_usd": pair.get("liquidity", {}).get("usd"),
                        "confidence": round(score, 4),
                        "reasons": reasons,
                        "dexscreener_url": pair.get("url"),
                    })
        except Exception as e:
            logger.error("Token detector error: {}", e)

        logger.info("Tokens: {} signals found.", len(signals))
        return signals

    async def _fetch_all_pairs(self) -> list:
        """Fetch pairs from boosted tokens list and multi-chain search."""
        results: list[dict] = []
        try:
            boosted = await self._fetch_boosted_pairs()
            results.extend(boosted)
        except Exception as e:
            logger.warning("Could not fetch boosted pairs: {}", e)

        for chain in _CHAINS:
            try:
                pairs = await self._fetch_search_pairs(chain)
                results.extend(pairs)
            except Exception as e:
                logger.warning("Could not fetch pairs for chain {}: {}", chain, e)

        return results

    async def _fetch_boosted_pairs(self) -> list:
        """Fetch the top boosted token pairs from DexScreener."""
        async with aiohttp.ClientSession() as session:
            async with session.get(_BOOSTS_URL) as resp:
                resp.raise_for_status()
                data = await resp.json(content_type=None)
                # Boosts endpoint returns a list of token objects, not pairs directly
                if isinstance(data, list):
                    return data
                return data.get("pairs", [])

    async def _fetch_search_pairs(self, chain: str) -> list:
        """Search DexScreener for active pairs on a given chain."""
        async with aiohttp.ClientSession() as session:
            async with session.get(_SEARCH_URL, params={"q": chain}) as resp:
                resp.raise_for_status()
                data = await resp.json(content_type=None)
                return data.get("pairs", [])

    def _score_pair(self, pair: dict) -> tuple[float, list[str]]:
        """
        Score a token pair on multiple factors. Returns (score, reasons).

        Scoring breakdown (max 1.0):
          - Price momentum  : up to 0.30  (tiered: >100% → 0.30, >50% → 0.20, >20% → 0.10)
          - Volume          : up to 0.20  (>$1M → 0.20, >$500k → 0.10)
          - Liquidity health: up to 0.20  (penalise if vol/liq ratio is suspicious)
          - Buy pressure    : up to 0.15  (buy% of h1 txns > 65%)
          - Token freshness : up to 0.15  (pair created < 48h ago)
        """
        score = 0.0
        reasons: list[str] = []

        try:
            price_change = float(pair.get("priceChange", {}).get("h24", 0) or 0)
            volume_h24 = float(pair.get("volume", {}).get("h24", 0) or 0)
            liquidity_usd = float((pair.get("liquidity") or {}).get("usd", 0) or 0)
            txns_h1 = pair.get("txns", {}).get("h1", {}) or {}
            buys_h1 = int(txns_h1.get("buys", 0) or 0)
            sells_h1 = int(txns_h1.get("sells", 0) or 0)
            pair_created_at = pair.get("pairCreatedAt")  # epoch ms

            # ── Price momentum ────────────────────────────────────────
            if price_change > 100:
                score += 0.30
                reasons.append(f"+{price_change:.0f}% 24h (strong)")
            elif price_change > 50:
                score += 0.20
                reasons.append(f"+{price_change:.0f}% 24h (moderate)")
            elif price_change > 20:
                score += 0.10
                reasons.append(f"+{price_change:.0f}% 24h (mild)")

            # ── Volume ────────────────────────────────────────────────
            if volume_h24 > 1_000_000:
                score += 0.20
                reasons.append(f"${volume_h24/1e6:.1f}M 24h volume")
            elif volume_h24 > 500_000:
                score += 0.10
                reasons.append(f"${volume_h24/1e3:.0f}k 24h volume")

            # ── Liquidity health ──────────────────────────────────────
            if liquidity_usd > 0:
                vol_liq_ratio = volume_h24 / liquidity_usd
                if vol_liq_ratio > 50:
                    # Extremely high vol/liq is a rug-pull warning — penalise
                    score -= 0.20
                    reasons.append(f"⚠ vol/liq={vol_liq_ratio:.0f}x (rug risk)")
                elif liquidity_usd > 500_000:
                    score += 0.20
                    reasons.append(f"${liquidity_usd/1e3:.0f}k liquidity (healthy)")
                elif liquidity_usd > 100_000:
                    score += 0.10
                    reasons.append(f"${liquidity_usd/1e3:.0f}k liquidity")

            # ── Buy pressure ──────────────────────────────────────────
            total_txns = buys_h1 + sells_h1
            if total_txns > 0:
                buy_pct = buys_h1 / total_txns
                if buy_pct > 0.70:
                    score += 0.15
                    reasons.append(f"{buy_pct*100:.0f}% buys h1 (accumulation)")
                elif buy_pct > 0.60:
                    score += 0.08
                    reasons.append(f"{buy_pct*100:.0f}% buys h1")

            # ── Token freshness ───────────────────────────────────────
            if pair_created_at:
                age_ms = time.time() * 1000 - float(pair_created_at)
                if 0 < age_ms < _NEW_PAIR_THRESHOLD_MS:
                    score += 0.15
                    age_h = age_ms / 3_600_000
                    reasons.append(f"new pair ({age_h:.1f}h old)")

        except Exception as e:
            logger.debug("Score error for pair {}: {}", pair.get("pairAddress"), e)
            return 0.0, []

        return round(min(max(score, 0.0), 1.0), 4), reasons
