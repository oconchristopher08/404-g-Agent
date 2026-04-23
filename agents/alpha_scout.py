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
            await self._scan_cycle()
            await asyncio.sleep(self.settings.SCAN_INTERVAL_SECONDS)

    async def _scan_cycle(self):
        logger.info("--- Starting scan cycle ---")
        try:
            sentiment_signals = await self.sentiment.scan()
            wallet_signals = await self.wallet_monitor.scan()
            token_signals = await self.token_detector.scan()

            alerts = self._merge_signals(sentiment_signals, wallet_signals, token_signals)
            if alerts:
                for alert in alerts:
                    logger.info("🚨 ALERT: {}", alert)
        except Exception as e:
            logger.error("Scan cycle error: {}", e)

    def _merge_signals(self, sentiment, wallets, tokens) -> list:
        """Combine all signals and return high-confidence alerts."""
        alerts = []
        all_signals = sentiment + wallets + tokens
        for signal in all_signals:
            if signal.get("confidence", 0) >= self.settings.SENTIMENT_THRESHOLD:
                alerts.append(signal)
        return alerts
