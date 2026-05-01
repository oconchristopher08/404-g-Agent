"""
Unit tests for 404-g Agent modules.
"""

import json
import os
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, mock_open

from config.settings import Settings
from modules.tokens.detector import TokenDetector
from modules.onchain.wallet_monitor import WalletMonitor
from modules.sentiment.analyzer import SentimentAnalyzer
from agents.alpha_scout import AlphaScoutAgent
from utils.price_oracle import PriceOracle
from utils.notifier import Notifier


@pytest.fixture
def settings():
    s = Settings()
    s.SENTIMENT_THRESHOLD = 0.6
    s.WHALE_WALLET_MIN_USD = 100_000
    s.ETHERSCAN_API_KEY = "dummy"
    s.COINGECKO_API_KEY = ""
    s.TELEGRAM_BOT_TOKEN = ""
    s.TELEGRAM_CHAT_ID = ""
    s.WHALE_WATCHLIST_PATH = "data/whales.json"
    s.ETH_PRICE_TTL_SECONDS = 300
    s.METRICS_PORT = 9999
    return s


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class TestSettings:
    def test_instance_level_attributes(self):
        """Settings attributes must be instance-level, not class-level."""
        s1 = Settings()
        s2 = Settings()
        s1.SENTIMENT_THRESHOLD = 0.9
        assert s2.SENTIMENT_THRESHOLD != 0.9, "Class-level attribute would share state"

    def test_defaults_are_sensible(self):
        s = Settings()
        assert s.SCAN_INTERVAL_SECONDS > 0
        assert 0.0 < s.SENTIMENT_THRESHOLD <= 1.0
        assert s.WHALE_WALLET_MIN_USD > 0


# ---------------------------------------------------------------------------
# PriceOracle
# ---------------------------------------------------------------------------

class TestPriceOracle:
    @pytest.mark.asyncio
    async def test_returns_prices_from_coingecko(self):
        oracle = PriceOracle(ttl_seconds=300)
        mock_data = {"ethereum": {"usd": 3500.0}, "binancecoin": {"usd": 450.0}}

        mock_resp = AsyncMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = AsyncMock(return_value=mock_data)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("utils.price_oracle.aiohttp.ClientSession", return_value=mock_session):
            prices = await oracle.get_prices()

        assert prices["ETH"] == pytest.approx(3500.0)
        assert prices["BNB"] == pytest.approx(450.0)

    @pytest.mark.asyncio
    async def test_uses_cache_within_ttl(self):
        oracle = PriceOracle(ttl_seconds=300)
        oracle._cache = {"ETH": 3000.0}
        oracle._fetched_at = time.monotonic()  # just fetched

        with patch("utils.price_oracle.aiohttp.ClientSession") as mock_cls:
            prices = await oracle.get_prices()
            mock_cls.assert_not_called()

        assert prices["ETH"] == pytest.approx(3000.0)

    @pytest.mark.asyncio
    async def test_fallback_on_fetch_failure(self):
        oracle = PriceOracle(ttl_seconds=1)
        oracle._fetched_at = 0  # force stale

        with patch("utils.price_oracle.aiohttp.ClientSession", side_effect=Exception("network error")):
            prices = await oracle.get_prices()

        # Should fall back to hardcoded defaults
        assert "ETH" in prices
        assert prices["ETH"] > 0

    @pytest.mark.asyncio
    async def test_get_single_symbol(self):
        oracle = PriceOracle(ttl_seconds=300)
        oracle._cache = {"ETH": 3200.0, "BNB": 420.0}
        oracle._fetched_at = time.monotonic()

        price = await oracle.get("ETH")
        assert price == pytest.approx(3200.0)


# ---------------------------------------------------------------------------
# TokenDetector._score_pair
# ---------------------------------------------------------------------------

class TestTokenDetectorScorePair:
    def test_strong_price_change_above_100(self, settings):
        detector = TokenDetector(settings)
        pair = {"priceChange": {"h24": "120"}, "volume": {"h24": "0"}}
        score, _ = detector._score_pair(pair)
        assert score == pytest.approx(0.30)

    def test_moderate_price_change_50_to_100(self, settings):
        detector = TokenDetector(settings)
        pair = {"priceChange": {"h24": "75"}, "volume": {"h24": "0"}}
        score, _ = detector._score_pair(pair)
        assert score == pytest.approx(0.20)

    def test_no_double_count_regression(self, settings):
        """120% must score higher than 75% — not equal as the old bug caused."""
        detector = TokenDetector(settings)
        score_120, _ = detector._score_pair({"priceChange": {"h24": "120"}, "volume": {"h24": "0"}})
        score_75, _ = detector._score_pair({"priceChange": {"h24": "75"}, "volume": {"h24": "0"}})
        assert score_120 > score_75

    def test_high_volume_bonus(self, settings):
        detector = TokenDetector(settings)
        pair = {"priceChange": {"h24": "0"}, "volume": {"h24": "1500000"}}
        score, reasons = detector._score_pair(pair)
        assert score == pytest.approx(0.20)
        assert any("volume" in r for r in reasons)

    def test_rug_risk_penalty(self, settings):
        """Extremely high vol/liq ratio should reduce score."""
        detector = TokenDetector(settings)
        pair = {
            "priceChange": {"h24": "200"},
            "volume": {"h24": "5000000"},
            "liquidity": {"usd": "10000"},  # vol/liq = 500x
        }
        score, reasons = detector._score_pair(pair)
        assert any("rug" in r.lower() for r in reasons)
        # Penalty should reduce score vs same pair with healthy liquidity
        pair_healthy = {
            "priceChange": {"h24": "200"},
            "volume": {"h24": "5000000"},
            "liquidity": {"usd": "2000000"},
        }
        score_healthy, _ = detector._score_pair(pair_healthy)
        assert score < score_healthy

    def test_buy_pressure_bonus(self, settings):
        detector = TokenDetector(settings)
        pair = {
            "priceChange": {"h24": "0"},
            "volume": {"h24": "0"},
            "txns": {"h1": {"buys": 80, "sells": 20}},
        }
        score, reasons = detector._score_pair(pair)
        assert score > 0
        assert any("buys" in r for r in reasons)

    def test_new_pair_freshness_bonus(self, settings):
        detector = TokenDetector(settings)
        created_recently = (time.time() - 3600) * 1000  # 1 hour ago
        pair = {
            "priceChange": {"h24": "0"},
            "volume": {"h24": "0"},
            "pairCreatedAt": created_recently,
        }
        score, reasons = detector._score_pair(pair)
        assert score > 0
        assert any("new pair" in r for r in reasons)

    def test_score_capped_at_one(self, settings):
        detector = TokenDetector(settings)
        pair = {
            "priceChange": {"h24": "500"},
            "volume": {"h24": "9999999"},
            "liquidity": {"usd": "9999999"},
            "txns": {"h1": {"buys": 99, "sells": 1}},
            "pairCreatedAt": (time.time() - 1800) * 1000,
        }
        score, _ = detector._score_pair(pair)
        assert score <= 1.0

    def test_missing_fields_returns_zero(self, settings):
        detector = TokenDetector(settings)
        score, _ = detector._score_pair({})
        assert score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TokenDetector.scan
# ---------------------------------------------------------------------------

class TestTokenDetectorScan:
    @pytest.mark.asyncio
    async def test_scan_returns_list(self, settings):
        detector = TokenDetector(settings)
        with patch.object(detector, "_fetch_all_pairs", new=AsyncMock(return_value=[])):
            result = await detector.scan()
            assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_scan_deduplicates_by_pair_address(self, settings):
        detector = TokenDetector(settings)
        pair = {
            "pairAddress": "0xabc",
            "priceChange": {"h24": "200"},
            "volume": {"h24": "2000000"},
            "liquidity": {"usd": "500000"},
            "baseToken": {"symbol": "MOON"},
        }
        with patch.object(detector, "_fetch_all_pairs", new=AsyncMock(return_value=[pair, pair])):
            result = await detector.scan()
            assert len(result) <= 1

    @pytest.mark.asyncio
    async def test_scan_includes_reasons(self, settings):
        detector = TokenDetector(settings)
        pair = {
            "pairAddress": "0xdef",
            "priceChange": {"h24": "120"},
            "volume": {"h24": "1500000"},
            "liquidity": {"usd": "600000"},
            "baseToken": {"symbol": "GEM"},
        }
        with patch.object(detector, "_fetch_all_pairs", new=AsyncMock(return_value=[pair])):
            result = await detector.scan()
            if result:
                assert "reasons" in result[0]
                assert isinstance(result[0]["reasons"], list)


# ---------------------------------------------------------------------------
# WalletMonitor
# ---------------------------------------------------------------------------

class TestWalletMonitor:
    def test_load_watchlist_from_file(self, settings, tmp_path):
        wl = {"wallets": ["0xabc123", "0xdef456"]}
        wl_file = tmp_path / "whales.json"
        wl_file.write_text(json.dumps(wl))
        settings.WHALE_WATCHLIST_PATH = str(wl_file)
        monitor = WalletMonitor(settings)
        assert len(monitor.known_whales) == 2
        assert "0xabc123" in monitor.known_whales

    def test_missing_watchlist_returns_empty(self, settings):
        settings.WHALE_WATCHLIST_PATH = "/nonexistent/path.json"
        monitor = WalletMonitor(settings)
        assert monitor.known_whales == []

    @pytest.mark.asyncio
    async def test_scan_skips_when_no_api_key(self, settings):
        settings.ETHERSCAN_API_KEY = ""
        monitor = WalletMonitor(settings)
        result = await monitor.scan()
        assert result == []

    @pytest.mark.asyncio
    async def test_scan_skips_when_watchlist_empty(self, settings):
        settings.WHALE_WATCHLIST_PATH = "/nonexistent/path.json"
        monitor = WalletMonitor(settings)
        result = await monitor.scan()
        assert result == []

    @pytest.mark.asyncio
    async def test_rate_limit_returns_empty_list(self, settings):
        monitor = WalletMonitor(settings)
        payload = {"status": "0", "message": "Max rate limit reached", "result": "Max rate limit reached"}

        mock_resp = AsyncMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = AsyncMock(return_value=payload)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("modules.onchain.wallet_monitor.aiohttp.ClientSession", return_value=mock_session):
            result = await monitor._fetch_txs("0xdeadbeef", "txlist")

        assert result == []

    @pytest.mark.asyncio
    async def test_successful_txlist_response(self, settings):
        monitor = WalletMonitor(settings)
        tx = {"hash": "0xabc", "value": str(int(1e18))}
        payload = {"status": "1", "message": "OK", "result": [tx]}

        mock_resp = AsyncMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = AsyncMock(return_value=payload)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("modules.onchain.wallet_monitor.aiohttp.ClientSession", return_value=mock_session):
            result = await monitor._fetch_txs("0xdeadbeef", "txlist")

        assert result == [tx]

    def test_evaluate_eth_tx_above_threshold(self, settings):
        monitor = WalletMonitor(settings)
        tx = {"value": str(int(50 * 1e18)), "hash": "0xabc", "to": "0x1234"}
        signal = monitor._evaluate_eth_tx(tx, "0xwallet", eth_price=3000.0)
        assert signal is not None
        assert signal["value_usd"] == pytest.approx(150_000.0)
        assert signal["token"] == "ETH"

    def test_evaluate_eth_tx_below_threshold(self, settings):
        monitor = WalletMonitor(settings)
        tx = {"value": str(int(1 * 1e18)), "hash": "0xabc", "to": "0x1234"}
        signal = monitor._evaluate_eth_tx(tx, "0xwallet", eth_price=3000.0)
        assert signal is None  # $3k < $100k threshold

    def test_evaluate_erc20_stablecoin(self, settings):
        monitor = WalletMonitor(settings)
        tx = {
            "value": str(int(500_000 * 1e6)),  # 500k USDC (6 decimals)
            "tokenDecimal": "6",
            "tokenSymbol": "USDC",
            "hash": "0xabc",
            "to": "0x1234",
            "contractAddress": "0xusdc",
        }
        signal = monitor._evaluate_erc20_tx(tx, "0xwallet", eth_price=3000.0)
        assert signal is not None
        assert signal["value_usd"] == pytest.approx(500_000.0)


# ---------------------------------------------------------------------------
# SentimentAnalyzer
# ---------------------------------------------------------------------------

class TestSentimentAnalyzer:
    def test_no_token_when_bearer_missing(self, settings):
        settings.TWITTER_BEARER_TOKEN = ""
        analyzer = SentimentAnalyzer(settings)
        assert analyzer._client is None

    @pytest.mark.asyncio
    async def test_scan_returns_empty_without_client(self, settings):
        settings.TWITTER_BEARER_TOKEN = ""
        analyzer = SentimentAnalyzer(settings)
        result = await analyzer.scan()
        assert result == []

    def test_crypto_lexicon_applied(self, settings):
        settings.TWITTER_BEARER_TOKEN = ""
        analyzer = SentimentAnalyzer(settings)
        # "wagmi" should score positive after lexicon injection
        score = analyzer.vader.polarity_scores("wagmi this is going to moon")
        assert score["compound"] > 0

    def test_negative_crypto_slang(self, settings):
        settings.TWITTER_BEARER_TOKEN = ""
        analyzer = SentimentAnalyzer(settings)
        score = analyzer.vader.polarity_scores("total rugpull, everyone got rekt")
        assert score["compound"] < 0


# ---------------------------------------------------------------------------
# AlphaScoutAgent
# ---------------------------------------------------------------------------

class TestAlphaScoutAgent:
    @pytest.mark.asyncio
    async def test_scan_cycle_calls_all_modules(self, settings):
        with patch("agents.alpha_scout.start_http_server"):
            agent = AlphaScoutAgent(settings)
        agent.sentiment.scan = AsyncMock(return_value=[])
        agent.wallet_monitor.scan = AsyncMock(return_value=[])
        agent.token_detector.scan = AsyncMock(return_value=[])
        agent.notifier.send_alert = AsyncMock()

        await agent._scan_cycle()

        agent.sentiment.scan.assert_called_once()
        agent.wallet_monitor.scan.assert_called_once()
        agent.token_detector.scan.assert_called_once()

    @pytest.mark.asyncio
    async def test_merge_signals_filters_by_threshold(self, settings):
        with patch("agents.alpha_scout.start_http_server"):
            agent = AlphaScoutAgent(settings)
        high = {"confidence": 0.9, "source": "dexscreener", "token": "MOON"}
        low = {"confidence": 0.1, "source": "twitter", "token": "MOON"}
        alerts = agent._merge_signals([low], [], [high])
        assert high in alerts
        assert low not in alerts

    @pytest.mark.asyncio
    async def test_multi_source_correlation_boosts_confidence(self, settings):
        with patch("agents.alpha_scout.start_http_server"):
            agent = AlphaScoutAgent(settings)
        # Same token from two sources — should get a confidence boost
        twitter_sig = {"source": "twitter", "token": "ETH", "confidence": 0.65, "sentiment": "bullish"}
        dex_sig = {"source": "dexscreener", "token": "ETH", "confidence": 0.65}
        alerts = await agent._correlate_and_filter([twitter_sig], [], [dex_sig])
        assert len(alerts) == 1
        assert alerts[0]["confidence"] > 0.65
        assert "corroborated_by" in alerts[0]

    @pytest.mark.asyncio
    async def test_dedup_suppresses_repeat_alert(self, settings):
        with patch("agents.alpha_scout.start_http_server"):
            agent = AlphaScoutAgent(settings)
        key = "pair:0xabc"
        await agent._mark_seen(key)
        assert await agent._is_duplicate(key)

    @pytest.mark.asyncio
    async def test_dedup_expires_after_ttl(self, settings):
        with patch("agents.alpha_scout.start_http_server"):
            agent = AlphaScoutAgent(settings)
        key = "pair:0xexpired"
        # Manually set an already-expired entry
        agent._dedup_cache[key] = time.monotonic() - 1
        assert not await agent._is_duplicate(key)

    def test_dedup_key_uses_tx_hash(self, settings):
        with patch("agents.alpha_scout.start_http_server"):
            agent = AlphaScoutAgent(settings)
        signal = {"source": "onchain", "tx_hash": "0xdeadbeef", "token": "ETH"}
        assert agent._dedup_key(signal) == "tx:0xdeadbeef"

    def test_dedup_key_uses_pair_address(self, settings):
        with patch("agents.alpha_scout.start_http_server"):
            agent = AlphaScoutAgent(settings)
        signal = {"source": "dexscreener", "pair_address": "0xpair123", "token": "MOON"}
        assert agent._dedup_key(signal) == "pair:0xpair123"


# ---------------------------------------------------------------------------
# Notifier
# ---------------------------------------------------------------------------

class TestNotifier:
    def test_disabled_when_no_credentials(self):
        notifier = Notifier(bot_token="", chat_id="")
        assert not notifier._enabled

    def test_enabled_with_credentials(self):
        notifier = Notifier(bot_token="abc:123", chat_id="-100123")
        assert notifier._enabled

    @pytest.mark.asyncio
    async def test_send_alert_logs_when_disabled(self):
        notifier = Notifier(bot_token="", chat_id="")
        signal = {"source": "dexscreener", "token": "MOON", "confidence": 0.85, "reasons": ["test"]}
        # Should not raise even without Telegram configured
        await notifier.send_alert(signal)

    def test_format_dexscreener_signal(self):
        notifier = Notifier(bot_token="", chat_id="")
        signal = {
            "source": "dexscreener",
            "token": "PEPE",
            "confidence": 0.85,
            "chain": "ethereum",
            "price_usd": "0.0001",
            "price_change_24h": "120",
            "volume_24h": 1_500_000,
            "liquidity_usd": 600_000,
            "reasons": ["+120% 24h", "$1.5M volume"],
            "dexscreener_url": "https://dexscreener.com/ethereum/0xabc",
        }
        msg = notifier._format(signal)
        assert "PEPE" in msg
        assert "85%" in msg
        assert "ethereum" in msg

    def test_format_onchain_signal(self):
        notifier = Notifier(bot_token="", chat_id="")
        signal = {
            "source": "onchain",
            "token": "ETH",
            "confidence": 0.75,
            "type": "eth_transfer",
            "wallet": "0xd8da6bf26964af9d7eed9e03e53415d37aa96045",
            "to": "0xabc123",
            "value_usd": 250_000,
            "tx_hash": "0xdeadbeef",
        }
        msg = notifier._format(signal)
        assert "ETH" in msg
        assert "250,000" in msg
        assert "Etherscan" in msg
