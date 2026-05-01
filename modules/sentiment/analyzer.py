"""
SentimentAnalyzer — pulls social data from Twitter/X and scores
crypto-related mentions using VADER sentiment analysis.
"""

import asyncio
import tweepy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from loguru import logger
from config.settings import Settings


class SentimentAnalyzer:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.vader = SentimentIntensityAnalyzer()
        self._init_twitter()

    def _init_twitter(self):
        try:
            auth = tweepy.OAuth1UserHandler(
                self.settings.TWITTER_API_KEY,
                self.settings.TWITTER_API_SECRET,
                self.settings.TWITTER_ACCESS_TOKEN,
                self.settings.TWITTER_ACCESS_SECRET,
            )
            self.twitter = tweepy.API(auth, wait_on_rate_limit=True)
            logger.info("Twitter client initialized.")
        except Exception as e:
            logger.warning("Twitter init failed: {}", e)
            self.twitter = None

    async def scan(self) -> list:
        """Scan recent crypto tweets and return sentiment signals."""
        signals = []
        if not self.twitter:
            return signals

        queries = ["$BTC", "$ETH", "$SOL", "crypto gem", "100x altcoin"]
        loop = asyncio.get_event_loop()
        for query in queries:
            try:
                # tweepy.API.search_tweets is synchronous. Run it in a thread
                # pool executor so it doesn't block the asyncio event loop and
                # stall the concurrent wallet and token scans.
                tweets = await loop.run_in_executor(
                    None,
                    lambda q=query: self.twitter.search_tweets(q=q, lang="en", count=50),
                )
                for tweet in tweets:
                    score = self.vader.polarity_scores(tweet.text)
                    compound = score["compound"]
                    if abs(compound) >= self.settings.SENTIMENT_THRESHOLD:
                        signals.append({
                            "source": "twitter",
                            "query": query,
                            "text": tweet.text[:120],
                            "confidence": abs(compound),
                            # compound == 0.0 is neutral, not bearish
                            "sentiment": "bullish" if compound > 0 else ("bearish" if compound < 0 else "neutral"),
                        })
            except Exception as e:
                logger.error("Sentiment scan error for {}: {}", query, e)

        logger.info("Sentiment: {} signals found.", len(signals))
        return signals
