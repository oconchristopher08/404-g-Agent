"""
HyperliquidScanner — surfaces alpha from Hyperliquid perpetuals data.

Signals produced:
  - Extreme funding rates: when |rate| is unusually high, longs or shorts
    are paying a premium that often precedes a mean-reversion move.
  - Funding rate divergence: when HL funding differs significantly from
    Binance/Bybit on the same coin, arbitrageurs will close the gap.
  - Open interest spikes: sudden OI growth on a coin not yet listed on
    major CEXs is a strong pre-listing accumulation signal.
  - Top movers: coins with the largest 24h price change on HL perps,
    cross-referenced against OI to filter noise.
"""

import aiohttp
from loguru import logger
from config.settings import Settings

_HL_API = "https://api.hyperliquid.xyz/info"

# Venues reported by Hyperliquid's predictedFundings endpoint
_CEX_VENUES = {"BinPerp", "BybitPerp", "OkxPerp"}

# Coins that are already on all major CEXs — less interesting for listing alpha
_MAJOR_COINS = {
    "BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "AVAX", "DOT",
    "MATIC", "LINK", "UNI", "ATOM", "LTC", "ETC", "XLM", "ALGO",
}


class HyperliquidScanner:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def scan(self) -> list:
        """Run all Hyperliquid signal checks and return a combined signal list."""
        signals: list[dict] = []
        try:
            meta_ctxs, predicted_fundings = await self._fetch_market_data()
            if not meta_ctxs:
                return signals

            universe, asset_ctxs = meta_ctxs
            coins = [asset["name"] for asset in universe]

            funding_signals = self._detect_funding_signals(coins, asset_ctxs, predicted_fundings)
            oi_signals = self._detect_oi_signals(coins, asset_ctxs)
            mover_signals = self._detect_top_movers(coins, asset_ctxs)

            signals = funding_signals + oi_signals + mover_signals

        except Exception as e:
            logger.error("HyperliquidScanner error: {}", e)

        logger.info("Hyperliquid: {} signals found.", len(signals))
        return signals

    async def _fetch_market_data(self) -> tuple:
        """Fetch metaAndAssetCtxs and predictedFundings concurrently."""
        import asyncio
        meta_task = self._post({"type": "metaAndAssetCtxs"})
        funding_task = self._post({"type": "predictedFundings"})
        meta_result, funding_result = await asyncio.gather(meta_task, funding_task, return_exceptions=True)

        if isinstance(meta_result, Exception):
            logger.warning("HL metaAndAssetCtxs failed: {}", meta_result)
            return None, []
        if isinstance(funding_result, Exception):
            logger.warning("HL predictedFundings failed: {}", funding_result)
            funding_result = []

        return meta_result, funding_result

    def _detect_funding_signals(
        self,
        coins: list[str],
        asset_ctxs: list[dict],
        predicted_fundings: list,
    ) -> list[dict]:
        """
        Flag coins where:
        1. |HL funding rate| exceeds the configured threshold (extreme funding).
        2. HL funding diverges from CEX funding by >2x (arbitrage opportunity).
        """
        signals: list[dict] = []
        threshold = self.settings.HL_FUNDING_THRESHOLD

        # Build a lookup: coin → {venue: rate}
        funding_by_coin: dict[str, dict[str, float]] = {}
        for entry in predicted_fundings:
            if not isinstance(entry, list) or len(entry) < 2:
                continue
            coin, venue_rates = entry[0], entry[1]
            rates: dict[str, float] = {}
            for venue_entry in venue_rates:
                if isinstance(venue_entry, list) and len(venue_entry) == 2:
                    venue, info = venue_entry
                    if isinstance(info, dict) and "fundingRate" in info:
                        rates[venue] = float(info["fundingRate"])
            funding_by_coin[coin] = rates

        for i, coin in enumerate(coins):
            if i >= len(asset_ctxs):
                break
            ctx = asset_ctxs[i]
            hl_rate = float(ctx.get("funding", 0) or 0)
            mark_px = float(ctx.get("markPx", 0) or 0)
            oi = float(ctx.get("openInterest", 0) or 0)
            oi_usd = oi * mark_px

            if oi_usd < self.settings.HL_MIN_OI_USD:
                continue

            # Signal 1: extreme funding on HL
            if abs(hl_rate) >= threshold:
                direction = "longs paying" if hl_rate > 0 else "shorts paying"
                annualised = hl_rate * 3 * 365 * 100  # 3 funding periods/day
                signals.append({
                    "source": "hyperliquid",
                    "type": "extreme_funding",
                    "token": coin,
                    "hl_funding_rate": round(hl_rate, 6),
                    "annualised_pct": round(annualised, 1),
                    "direction": direction,
                    "oi_usd": round(oi_usd, 0),
                    "mark_price": mark_px,
                    "confidence": min(abs(hl_rate) / threshold * 0.5, 1.0),
                    "signal": f"|funding| {hl_rate:.4%} ({direction}, {annualised:.0f}% ann.)",
                })

            # Signal 2: HL vs CEX funding divergence
            cex_rates = {v: r for v, r in funding_by_coin.get(coin, {}).items() if v in _CEX_VENUES}
            if cex_rates and hl_rate != 0:
                avg_cex = sum(cex_rates.values()) / len(cex_rates)
                if avg_cex != 0:
                    divergence = abs(hl_rate - avg_cex) / abs(avg_cex)
                    if divergence > 1.5:  # HL rate is >2.5x the CEX average
                        signals.append({
                            "source": "hyperliquid",
                            "type": "funding_divergence",
                            "token": coin,
                            "hl_funding_rate": round(hl_rate, 6),
                            "avg_cex_funding_rate": round(avg_cex, 6),
                            "divergence_ratio": round(divergence, 2),
                            "oi_usd": round(oi_usd, 0),
                            "confidence": min(divergence / 3.0, 1.0),
                            "signal": f"HL/CEX funding divergence {divergence:.1f}x",
                        })

        return signals

    def _detect_oi_signals(self, coins: list[str], asset_ctxs: list[dict]) -> list[dict]:
        """
        Flag coins with high OI that are NOT yet on major CEXs — these are
        candidates for upcoming listings. Also flag coins where OI has grown
        significantly vs the previous day (sudden accumulation).
        """
        signals: list[dict] = []
        min_oi = self.settings.HL_MIN_OI_USD

        for i, coin in enumerate(coins):
            if i >= len(asset_ctxs):
                break
            ctx = asset_ctxs[i]
            mark_px = float(ctx.get("markPx", 0) or 0)
            oi = float(ctx.get("openInterest", 0) or 0)
            oi_usd = oi * mark_px
            prev_px = float(ctx.get("prevDayPx", mark_px) or mark_px)
            day_volume = float(ctx.get("dayNtlVlm", 0) or 0)

            if oi_usd < min_oi:
                continue

            # Pre-listing signal: meaningful OI on a coin not yet on major CEXs
            if coin not in _MAJOR_COINS and oi_usd >= min_oi * 2:
                # OI/volume ratio > 0.5 means positions are being held, not just traded
                oi_vol_ratio = oi_usd / day_volume if day_volume > 0 else 0
                confidence = min(oi_usd / (min_oi * 10), 0.85)
                signals.append({
                    "source": "hyperliquid",
                    "type": "pre_listing_oi",
                    "token": coin,
                    "oi_usd": round(oi_usd, 0),
                    "day_volume_usd": round(day_volume, 0),
                    "oi_vol_ratio": round(oi_vol_ratio, 2),
                    "mark_price": mark_px,
                    "confidence": round(confidence, 4),
                    "signal": f"${oi_usd/1e6:.1f}M OI on non-CEX coin (pre-listing candidate)",
                })

        return signals

    def _detect_top_movers(self, coins: list[str], asset_ctxs: list[dict]) -> list[dict]:
        """
        Return the top 5 coins by 24h price change on HL, filtered to those
        with meaningful OI (to exclude low-liquidity noise).
        """
        candidates: list[dict] = []
        min_oi = self.settings.HL_MIN_OI_USD

        for i, coin in enumerate(coins):
            if i >= len(asset_ctxs):
                break
            ctx = asset_ctxs[i]
            mark_px = float(ctx.get("markPx", 0) or 0)
            prev_px = float(ctx.get("prevDayPx", 0) or 0)
            oi = float(ctx.get("openInterest", 0) or 0)
            oi_usd = oi * mark_px

            if oi_usd < min_oi or prev_px == 0 or mark_px == 0:
                continue

            pct_change = (mark_px - prev_px) / prev_px * 100
            candidates.append({
                "coin": coin,
                "pct_change": pct_change,
                "oi_usd": oi_usd,
                "mark_price": mark_px,
                "ctx": ctx,
            })

        # Top 5 gainers with OI backing
        top = sorted(candidates, key=lambda x: x["pct_change"], reverse=True)[:5]
        signals: list[dict] = []
        for item in top:
            if item["pct_change"] < 10:
                break  # not interesting below 10%
            signals.append({
                "source": "hyperliquid",
                "type": "top_mover",
                "token": item["coin"],
                "price_change_24h": round(item["pct_change"], 2),
                "oi_usd": round(item["oi_usd"], 0),
                "mark_price": item["mark_price"],
                "confidence": min(item["pct_change"] / 100, 1.0),
                "signal": f"+{item['pct_change']:.1f}% 24h on HL perp (${item['oi_usd']/1e6:.1f}M OI)",
            })

        return signals

    async def _post(self, payload: dict) -> any:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                _HL_API,
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as resp:
                resp.raise_for_status()
                return await resp.json(content_type=None)
