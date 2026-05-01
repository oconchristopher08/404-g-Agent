"""
AlphaScoutAgent — orchestrates all five intelligence modules, correlates
signals across sources, deduplicates via Redis, fires Telegram alerts,
and exposes Prometheus metrics.

Modules:
  - SentimentAnalyzer  : Twitter v2 sentiment (VADER + crypto lexicon)
  - WalletMonitor      : Whale ETH/ERC-20 on-chain movements
  - TokenDetector      : DexScreener multi-chain token scoring
  - HyperliquidScanner : Perp funding rates, OI spikes, top movers
  - NansenScanner      : Smart money netflows + token screener
  - ListingDetector    : Pre-listing deposit detection at top-5 CEXs
"""

import asyncio
import signal
from collections import defaultdict

from loguru import logger
from prometheus_client import Counter, Histogram, start_http_server

from config.settings import Settings
from modules.sentiment.analyzer import SentimentAnalyzer
from modules.onchain.wallet_monitor import WalletMonitor
from modules.onchain.listing_detector import ListingDetector
from modules.tokens.detector import TokenDetector
from modules.hyperliquid.scanner import HyperliquidScanner
from modules.nansen.smart_money import NansenScanner
from utils.notifier import Notifier
from utils.price_oracle import PriceOracle

# ── Prometheus metrics ────────────────────────────────────────────────────
_CYCLES_TOTAL = Counter("agent_scan_cycles_total", "Total scan cycles completed")
_SIGNALS_TOTAL = Counter("agent_signals_total", "Signals found", ["source"])
_ALERTS_TOTAL = Counter("agent_alerts_total", "Alerts fired", ["source"])
_CYCLE_DURATION = Histogram("agent_cycle_duration_seconds", "Scan cycle wall time")

_DEDUP_TTL_SECONDS = 3600

# Sources that produce listing-related signals — given a confidence boost
# when corroborated because they are independent data pipelines
_LISTING_SOURCES = {"hyperliquid", "nansen", "onchain"}


class AlphaScoutAgent:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.price_oracle = PriceOracle(
            ttl_seconds=settings.ETH_PRICE_TTL_SECONDS,
            api_key=settings.COINGECKO_API_KEY,
        )
        self.sentiment = SentimentAnalyzer(settings)
        self.wallet_monitor = WalletMonitor(settings, price_oracle=self.price_oracle)
        self.listing_detector = ListingDetector(settings)
        self.token_detector = TokenDetector(settings)
        self.hl_scanner = HyperliquidScanner(settings)
        self.nansen_scanner = NansenScanner(settings)
        self.notifier = Notifier(
            bot_token=settings.TELEGRAM_BOT_TOKEN,
            chat_id=settings.TELEGRAM_CHAT_ID,
        )
        self._dedup_cache: dict[str, float] = {}
        self._redis = None
        self._start_metrics_server()

    def _start_metrics_server(self):
        try:
            start_http_server(self.settings.METRICS_PORT)
            logger.info("Prometheus metrics on :{}", self.settings.METRICS_PORT)
        except Exception as e:
            logger.warning("Could not start metrics server: {}", e)

    async def _init_redis(self):
        try:
            import redis.asyncio as aioredis
            self._redis = aioredis.from_url(self.settings.REDIS_URL, decode_responses=True)
            await self._redis.ping()
            logger.info("Redis connected for alert deduplication.")
        except Exception as e:
            logger.warning("Redis unavailable ({}), using in-process dedup cache.", e)
            self._redis = None

    async def run(self):
        await self._init_redis()
        logger.info("Agent scan loop started. Interval: {}s", self.settings.SCAN_INTERVAL_SECONDS)

        loop = asyncio.get_running_loop()
        task = asyncio.current_task()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, task.cancel)
            except NotImplementedError:
                pass

        while True:
            try:
                await self._scan_cycle()
            except asyncio.CancelledError:
                logger.info("Agent shutting down gracefully.")
                raise
            except Exception as e:
                logger.error("Unhandled error in run loop: {}", e)
            await asyncio.sleep(self.settings.SCAN_INTERVAL_SECONDS)

    async def _scan_cycle(self):
        logger.info("--- Starting scan cycle ---")
        with _CYCLE_DURATION.time():
            try:
                (
                    sentiment_signals,
                    wallet_signals,
                    listing_signals,
                    token_signals,
                    hl_signals,
                    nansen_signals,
                ) = await asyncio.gather(
                    self.sentiment.scan(),
                    self.wallet_monitor.scan(),
                    self.listing_detector.scan(),
                    self.token_detector.scan(),
                    self.hl_scanner.scan(),
                    self.nansen_scanner.scan(),
                )

                source_map = {
                    "twitter": sentiment_signals,
                    "onchain": (wallet_signals or []) + (listing_signals or []),
                    "dexscreener": token_signals,
                    "hyperliquid": hl_signals,
                    "nansen": nansen_signals,
                }
                for src, sigs in source_map.items():
                    _SIGNALS_TOTAL.labels(source=src).inc(len(sigs or []))

                alerts = await self._correlate_and_filter(
                    sentiment_signals or [],
                    (wallet_signals or []) + (listing_signals or []),
                    token_signals or [],
                    hl_signals or [],
                    nansen_signals or [],
                )

                for alert in alerts:
                    dedup_key = self._dedup_key(alert)
                    if await self._is_duplicate(dedup_key):
                        continue
                    await self._mark_seen(dedup_key)
                    _ALERTS_TOTAL.labels(source=alert.get("source", "unknown")).inc()
                    await self.notifier.send_alert(alert)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error("Scan cycle error: {}", e)

        _CYCLES_TOTAL.inc()

    async def _correlate_and_filter(
        self,
        sentiment: list,
        onchain: list,
        tokens: list,
        hyperliquid: list,
        nansen: list,
    ) -> list:
        """
        Merge signals from all five sources. Confidence is boosted when the
        same token appears across multiple independent pipelines.

        Bonus schedule:
          +0.15 per additional source (general)
          +0.20 extra when listing-specific sources (HL + Nansen + onchain)
          all agree — these three are the strongest pre-listing indicators.
        """
        all_signals = sentiment + onchain + tokens + hyperliquid + nansen
        threshold = self.settings.SENTIMENT_THRESHOLD

        by_token: dict[str, list[dict]] = defaultdict(list)
        for sig in all_signals:
            token = (sig.get("token") or sig.get("query") or "UNKNOWN").upper().lstrip("$")
            by_token[token].append(sig)

        alerts: list[dict] = []
        for token, sigs in by_token.items():
            sources = {s.get("source") for s in sigs}
            base_confidence = max(s.get("confidence", 0) for s in sigs)

            # Standard multi-source bonus
            multi_source_bonus = (len(sources) - 1) * 0.15

            # Extra bonus when multiple listing-signal sources agree
            listing_sources_present = sources & _LISTING_SOURCES
            listing_bonus = 0.20 if len(listing_sources_present) >= 2 else 0.0

            final_confidence = min(base_confidence + multi_source_bonus + listing_bonus, 1.0)

            if final_confidence < threshold:
                continue

            best = max(sigs, key=lambda s: s.get("confidence", 0))
            alert = {**best, "confidence": round(final_confidence, 4)}
            if len(sources) > 1:
                alert["corroborated_by"] = sorted(sources)
            if listing_bonus > 0:
                alert["listing_signal"] = True
            alerts.append(alert)

        # Sort: listing signals first, then by confidence
        alerts.sort(key=lambda a: (a.get("listing_signal", False), a["confidence"]), reverse=True)
        return alerts

    def _dedup_key(self, signal: dict) -> str:
        if tx := signal.get("tx_hash"):
            return f"tx:{tx}"
        if pair := signal.get("pair_address"):
            return f"pair:{pair}"
        if token_addr := signal.get("token_address"):
            return f"token:{token_addr}:{signal.get('type', '')}"
        token = signal.get("token", "")
        sig_type = signal.get("type", signal.get("sentiment", ""))
        return f"{signal.get('source', '')}:{token}:{sig_type}"

    async def _is_duplicate(self, key: str) -> bool:
        if self._redis:
            try:
                return bool(await self._redis.exists(key))
            except Exception:
                pass
        import time
        return time.monotonic() < self._dedup_cache.get(key, 0)

    async def _mark_seen(self, key: str) -> None:
        if self._redis:
            try:
                await self._redis.setex(key, _DEDUP_TTL_SECONDS, "1")
                return
            except Exception:
                pass
        import time
        now = time.monotonic()
        self._dedup_cache[key] = now + _DEDUP_TTL_SECONDS
        self._dedup_cache = {k: v for k, v in self._dedup_cache.items() if v > now}

    def _merge_signals(self, sentiment, wallets, tokens) -> list:
        """Legacy helper kept for backward compatibility with tests."""
        all_signals = (sentiment or []) + (wallets or []) + (tokens or [])
        return [s for s in all_signals if s.get("confidence", 0) >= self.settings.SENTIMENT_THRESHOLD]
