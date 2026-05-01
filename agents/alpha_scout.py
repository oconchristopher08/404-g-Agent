"""
AlphaScoutAgent — orchestrates sentiment, on-chain, and token modules,
correlates signals across sources, deduplicates via Redis, fires Telegram
alerts, and exposes Prometheus metrics.
"""

import asyncio
import signal
from collections import defaultdict

from loguru import logger
from prometheus_client import Counter, Histogram, start_http_server

from config.settings import Settings
from modules.sentiment.analyzer import SentimentAnalyzer
from modules.onchain.wallet_monitor import WalletMonitor
from modules.tokens.detector import TokenDetector
from utils.notifier import Notifier
from utils.price_oracle import PriceOracle

# ── Prometheus metrics ────────────────────────────────────────────────────
_CYCLES_TOTAL = Counter("agent_scan_cycles_total", "Total scan cycles completed")
_SIGNALS_TOTAL = Counter("agent_signals_total", "Signals found", ["source"])
_ALERTS_TOTAL = Counter("agent_alerts_total", "Alerts fired", ["source"])
_CYCLE_DURATION = Histogram("agent_cycle_duration_seconds", "Scan cycle wall time")

# Alert deduplication TTL — suppress re-alerting the same signal for this long
_DEDUP_TTL_SECONDS = 3600


class AlphaScoutAgent:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.price_oracle = PriceOracle(
            ttl_seconds=settings.ETH_PRICE_TTL_SECONDS,
            api_key=settings.COINGECKO_API_KEY,
        )
        self.sentiment = SentimentAnalyzer(settings)
        self.wallet_monitor = WalletMonitor(settings, price_oracle=self.price_oracle)
        self.token_detector = TokenDetector(settings)
        self.notifier = Notifier(
            bot_token=settings.TELEGRAM_BOT_TOKEN,
            chat_id=settings.TELEGRAM_CHAT_ID,
        )
        # In-process dedup cache: key → expiry timestamp
        # Redis is used when available; falls back to this dict.
        self._dedup_cache: dict[str, float] = {}
        self._redis = None
        self._start_metrics_server()

    def _start_metrics_server(self):
        try:
            start_http_server(self.settings.METRICS_PORT)
            logger.info("Prometheus metrics available on :{}", self.settings.METRICS_PORT)
        except Exception as e:
            logger.warning("Could not start metrics server: {}", e)

    async def _init_redis(self):
        """Attempt to connect to Redis for persistent alert deduplication."""
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

        # Register SIGTERM/SIGINT for graceful shutdown
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, task.cancel)
            except NotImplementedError:
                pass  # Windows doesn't support add_signal_handler

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
                sentiment_signals, wallet_signals, token_signals = await asyncio.gather(
                    self.sentiment.scan(),
                    self.wallet_monitor.scan(),
                    self.token_detector.scan(),
                )

                for src, sigs in [
                    ("twitter", sentiment_signals),
                    ("onchain", wallet_signals),
                    ("dexscreener", token_signals),
                ]:
                    _SIGNALS_TOTAL.labels(source=src).inc(len(sigs or []))

                alerts = await self._correlate_and_filter(
                    sentiment_signals or [],
                    wallet_signals or [],
                    token_signals or [],
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
        wallets: list,
        tokens: list,
    ) -> list:
        """
        Merge signals from all three sources. When the same token appears in
        multiple sources, boost its confidence score — multi-source confirmation
        is the core value of a multi-module agent.
        """
        all_signals = sentiment + wallets + tokens
        threshold = self.settings.SENTIMENT_THRESHOLD

        # Group by normalised token symbol
        by_token: dict[str, list[dict]] = defaultdict(list)
        for sig in all_signals:
            token = (sig.get("token") or sig.get("query") or "UNKNOWN").upper().lstrip("$")
            by_token[token].append(sig)

        alerts: list[dict] = []
        for token, sigs in by_token.items():
            sources = {s.get("source") for s in sigs}
            base_confidence = max(s.get("confidence", 0) for s in sigs)

            # Multi-source confirmation bonus: +0.15 per additional source
            multi_source_bonus = (len(sources) - 1) * 0.15
            final_confidence = min(base_confidence + multi_source_bonus, 1.0)

            if final_confidence < threshold:
                continue

            # Use the highest-confidence signal as the base alert, then annotate
            best = max(sigs, key=lambda s: s.get("confidence", 0))
            alert = {**best, "confidence": round(final_confidence, 4)}
            if len(sources) > 1:
                alert["corroborated_by"] = sorted(sources)
            alerts.append(alert)

        return alerts

    def _dedup_key(self, signal: dict) -> str:
        """Stable key that identifies a unique alert event."""
        source = signal.get("source", "")
        # For on-chain: deduplicate by tx hash
        if tx := signal.get("tx_hash"):
            return f"tx:{tx}"
        # For tokens: deduplicate by pair address
        if pair := signal.get("pair_address"):
            return f"pair:{pair}"
        # For Twitter: deduplicate by token + sentiment direction
        token = signal.get("token", "")
        sentiment = signal.get("sentiment", "")
        return f"{source}:{token}:{sentiment}"

    async def _is_duplicate(self, key: str) -> bool:
        if self._redis:
            try:
                return bool(await self._redis.exists(key))
            except Exception:
                pass
        # Fallback: in-process dict with expiry
        import time
        expiry = self._dedup_cache.get(key, 0)
        return time.monotonic() < expiry

    async def _mark_seen(self, key: str) -> None:
        if self._redis:
            try:
                await self._redis.setex(key, _DEDUP_TTL_SECONDS, "1")
                return
            except Exception:
                pass
        import time
        self._dedup_cache[key] = time.monotonic() + _DEDUP_TTL_SECONDS
        # Prune expired entries to prevent unbounded growth
        now = time.monotonic()
        self._dedup_cache = {k: v for k, v in self._dedup_cache.items() if v > now}

    def _merge_signals(self, sentiment, wallets, tokens) -> list:
        """Legacy helper kept for backward compatibility with tests."""
        all_signals = (sentiment or []) + (wallets or []) + (tokens or [])
        return [s for s in all_signals if s.get("confidence", 0) >= self.settings.SENTIMENT_THRESHOLD]
