"""
SentimentAnalyzer — scans recent crypto tweets via Twitter v2 API and scores
them with VADER. Falls back gracefully when no bearer token is configured.
"""

import asyncio
from functools import partial

import tweepy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from loguru import logger
from config.settings import Settings

# Crypto-specific lexicon additions that VADER misses.
# Positive slang that VADER scores neutral or negative.
_CRYPTO_LEXICON: dict[str, float] = {
    "moon": 2.5,
    "mooning": 2.5,
    "wen moon": 2.0,
    "bullish": 2.5,
    "pump": 1.5,
    "ape": 1.5,
    "aping": 1.5,
    "gm": 1.0,
    "wagmi": 2.5,
    "ngmi": -2.5,
    "rekt": -3.0,
    "rug": -3.5,
    "rugpull": -3.5,
    "dump": -2.0,
    "dumping": -2.0,
    "bearish": -2.5,
    "degen": 1.0,
    "gem": 2.0,
    "100x": 2.5,
    "1000x": 3.0,
    "alpha": 2.0,
    "based": 1.5,
    "ser": 0.5,
}

_QUERIES = [
    "$BTC lang:en -is:retweet",
    "$ETH lang:en -is:retweet",
    "$SOL lang:en -is:retweet",
    "crypto gem 100x lang:en -is:retweet",
    "altcoin breakout lang:en -is:retweet",
]


class SentimentAnalyzer:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.vader = SentimentIntensityAnalyzer()
        # Inject crypto-specific lexicon so VADER understands on-chain slang
        self.vader.lexicon.update(_CRYPTO_LEXICON)
        self._client: tweepy.Client | None = None
        self._init_twitter()

    def _init_twitter(self):
        token = self.settings.TWITTER_BEARER_TOKEN
        if not token:
            logger.warning(
                "TWITTER_BEARER_TOKEN not set — sentiment module disabled. "
                "Set it in .env to enable Twitter v2 scanning."
            )
            return
        try:
            self._client = tweepy.Client(bearer_token=token, wait_on_rate_limit=True)
            logger.info("Twitter v2 client initialized.")
        except Exception as e:
            logger.warning("Twitter v2 init failed: {}", e)

    async def scan(self) -> list:
        """Search recent tweets for each query and return sentiment signals."""
        if not self._client:
            return []

        signals = []
        loop = asyncio.get_event_loop()

        for query in _QUERIES:
            try:
                # search_recent_tweets is synchronous; run it in a thread pool
                # so it doesn't block the event loop while waiting on the network.
                fetch = partial(
                    self._client.search_recent_tweets,
                    query=query,
                    max_results=100,
                    tweet_fields=["text", "created_at", "public_metrics"],
                )
                response = await loop.run_in_executor(None, fetch)

                if not response or not response.data:
                    continue

                for tweet in response.data:
                    text = tweet.text
                    scores = self.vader.polarity_scores(text)
                    compound = scores["compound"]
                    if abs(compound) >= self.settings.SENTIMENT_THRESHOLD:
                        # Extract a token symbol from the query (e.g. "$BTC" → "BTC")
                        token = query.split()[0].lstrip("$") if query.startswith("$") else "CRYPTO"
                        signals.append({
                            "source": "twitter",
                            "token": token,
                            "query": query,
                            "text": text[:140],
                            "confidence": round(abs(compound), 4),
                            "sentiment": "bullish" if compound > 0 else "bearish",
                            "likes": tweet.public_metrics.get("like_count", 0) if tweet.public_metrics else 0,
                            "retweets": tweet.public_metrics.get("retweet_count", 0) if tweet.public_metrics else 0,
                        })
            except tweepy.TooManyRequests:
                logger.warning("Twitter rate limit hit — skipping remaining queries this cycle.")
                break
            except Exception as e:
                logger.error("Sentiment scan error for '{}': {}", query, e)

        logger.info("Sentiment: {} signals found.", len(signals))
        return signals
