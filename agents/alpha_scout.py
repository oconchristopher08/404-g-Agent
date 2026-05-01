"""
AlphaScoutAgent — orchestrates sentiment, on-chain, and token modules
to surface early crypto market opportunities.
"""

import asyncio
from loguru import logger
from config.settings import Settings
from modules.sentiment.analyzer import SentimentAnalyzer
from modules.onchain.wallet_monitor import WalletMonitor
from modules.tokens.detector import TokenDetector


class AlphaScoutAgent:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.sentiment = SentimentAnalyzer(settings)
        self.wallet_monitor = WalletMonitor(settings)
        self.token_detector = TokenDetector(settings)

    async def run(self):
        logger.info("Agent scan loop started. Interval: {}s", self.settings.SCAN_INTERVAL_SECONDS)
        while True:
            try:
                await self._scan_cycle()
            except asyncio.CancelledError:
                logger.info("Agent run loop cancelled.")
                raise
            except Exception as e:
                logger.error("Unhandled error in run loop: {}", e)
            # Sleep is always reached so a scan failure doesn't cause a tight
            # spin loop that hammers external APIs with no delay.
            await asyncio.sleep(self.settings.SCAN_INTERVAL_SECONDS)

    async def _scan_cycle(self):
        logger.info("--- Starting scan cycle ---")
        try:
            # Run all three I/O-bound scans concurrently instead of sequentially
            # so cycle time is bounded by the slowest module, not their sum.
            sentiment_signals, wallet_signals, token_signals = await asyncio.gather(
                self.sentiment.scan(),
                self.wallet_monitor.scan(),
                self.token_detector.scan(),
            )

            alerts = self._merge_signals(sentiment_signals, wallet_signals, token_signals)
            if alerts:
                for alert in alerts:
                    logger.info("🚨 ALERT: {}", alert)
        except Exception as e:
            logger.error("Scan cycle error: {}", e)

    def _merge_signals(self, sentiment, wallets, tokens) -> list:
        """Combine all signals and return high-confidence alerts."""
        alerts = []
        # Guard against None returns from any module so list concatenation
        # doesn't raise TypeError and silence the other modules' results.
        all_signals = (sentiment or []) + (wallets or []) + (tokens or [])
        for signal in all_signals:
            if signal.get("confidence", 0) >= self.settings.SENTIMENT_THRESHOLD:
                alerts.append(signal)
        return alerts
