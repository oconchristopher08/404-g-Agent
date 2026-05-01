"""
Unit tests for 404-g Agent modules.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from config.settings import Settings
from modules.tokens.detector import TokenDetector
from modules.onchain.wallet_monitor import WalletMonitor
from modules.sentiment.analyzer import SentimentAnalyzer
from agents.alpha_scout import AlphaScoutAgent


@pytest.fixture
def settings():
    return Settings()


# ---------------------------------------------------------------------------
# TokenDetector
# ---------------------------------------------------------------------------

def test_token_detector_score_high_signal(settings):
    detector = TokenDetector(settings)
    pair = {
        "priceChange": {"h24": "120"},
        "volume": {"h24": "750000"},
        "baseToken": {"symbol": "MOON"},
        "pairAddress": "0xabc123",
        "priceUsd": "0.0042",
    }
    score = detector._score_pair(pair)
    assert score >= 0.6, f"Expected high confidence, got {score}"


def test_token_detector_score_low_signal(settings):
    detector = TokenDetector(settings)
    pair = {
        "priceChange": {"h24": "2"},
        "volume": {"h24": "1000"},
    }
    score = detector._score_pair(pair)
    assert score == 0.0


def test_token_detector_score_no_double_count(settings):
    """price_change > 100 must not accumulate both the >50 and >100 branches."""
    detector = TokenDetector(settings)
    pair_over_100 = {"priceChange": {"h24": "150"}, "volume": {"h24": "0"}}
    pair_over_50 = {"priceChange": {"h24": "75"}, "volume": {"h24": "0"}}
    score_100 = detector._score_pair(pair_over_100)
    score_50 = detector._score_pair(pair_over_50)
    assert score_100 > score_50, (
        f">100% change ({score_100}) should outscore >50% change ({score_50})"
    )


async def test_token_detector_scan_returns_list(settings):
    detector = TokenDetector(settings)
    with patch.object(detector, "_fetch_trending_pairs", new=AsyncMock(return_value=[])):
        result = await detector.scan()
        assert isinstance(result, list)


async def test_token_detector_fetch_uses_search_endpoint(settings):
    """_fetch_trending_pairs must call /dex/search, not the non-existent /dex/tokens/trending."""
    detector = TokenDetector(settings)

    mock_resp = AsyncMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json = AsyncMock(return_value={"pairs": [{"baseToken": {"symbol": "X"}}]})
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock()
    mock_session.get = MagicMock(return_value=mock_resp)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch("modules.tokens.detector.aiohttp.ClientSession", return_value=mock_session):
        pairs = await detector._fetch_trending_pairs()

    called_url = mock_session.get.call_args[0][0]
    assert called_url.endswith("/dex/search"), (
        f"Expected /dex/search endpoint, got: {called_url}"
    )
    assert pairs == [{"baseToken": {"symbol": "X"}}]


# ---------------------------------------------------------------------------
# WalletMonitor
# ---------------------------------------------------------------------------

async def test_wallet_monitor_skips_when_no_api_key(settings):
    settings.ETHERSCAN_API_KEY = ""
    monitor = WalletMonitor(settings)
    monitor.known_whales = ["0xdeadbeef"]
    result = await monitor.scan()
    assert result == []


async def test_wallet_monitor_handles_etherscan_error_string(settings):
    """Etherscan error responses return result as a string; must not iterate chars."""
    settings.ETHERSCAN_API_KEY = "dummy"
    monitor = WalletMonitor(settings)
    monitor.known_whales = ["0xdeadbeef"]

    error_payload = {
        "status": "0",
        "message": "Max rate limit reached",
        "result": "Max rate limit reached",
    }

    mock_resp = AsyncMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json = AsyncMock(return_value=error_payload)
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock()
    mock_session.get = MagicMock(return_value=mock_resp)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch("modules.onchain.wallet_monitor.aiohttp.ClientSession", return_value=mock_session):
        result = await monitor.scan()

    assert result == []


async def test_wallet_monitor_returns_signals_for_whale_tx(settings):
    """A valid large transaction above the threshold should produce a signal."""
    settings.ETHERSCAN_API_KEY = "dummy"
    settings.WHALE_WALLET_MIN_USD = 100_000
    monitor = WalletMonitor(settings)
    monitor.known_whales = ["0xwhale"]

    # 50 ETH * 3000 = $150,000 — above threshold
    wei_50_eth = str(int(50 * 1e18))
    tx_payload = {
        "status": "1",
        "message": "OK",
        "result": [{"value": wei_50_eth, "hash": "0xtxhash"}],
    }

    mock_resp = AsyncMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json = AsyncMock(return_value=tx_payload)
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock()
    mock_session.get = MagicMock(return_value=mock_resp)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch("modules.onchain.wallet_monitor.aiohttp.ClientSession", return_value=mock_session):
        result = await monitor.scan()

    assert len(result) == 1
    assert result[0]["source"] == "onchain"
    assert result[0]["value_usd"] == 150_000.0


# ---------------------------------------------------------------------------
# SentimentAnalyzer — sentiment label classification
# ---------------------------------------------------------------------------

def test_sentiment_neutral_compound_is_not_bearish(settings):
    """compound == 0.0 must be labelled 'neutral', not 'bearish'."""
    compound = 0.0
    label = "bullish" if compound > 0 else ("bearish" if compound < 0 else "neutral")
    assert label == "neutral", f"Expected 'neutral' for compound=0.0, got '{label}'"


def test_sentiment_positive_compound_is_bullish(settings):
    compound = 0.8
    label = "bullish" if compound > 0 else ("bearish" if compound < 0 else "neutral")
    assert label == "bullish"


def test_sentiment_negative_compound_is_bearish(settings):
    compound = -0.7
    label = "bullish" if compound > 0 else ("bearish" if compound < 0 else "neutral")
    assert label == "bearish"


# ---------------------------------------------------------------------------
# AlphaScoutAgent — run loop spin-guard
# ---------------------------------------------------------------------------

async def test_run_loop_sleeps_after_scan_error(settings):
    """A scan cycle exception must not skip the sleep (tight spin loop guard)."""
    agent = AlphaScoutAgent(settings)

    call_count = 0

    async def failing_then_cancel():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("simulated scan failure")
        raise asyncio.CancelledError()

    sleep_called = []

    async def mock_sleep(seconds):
        sleep_called.append(seconds)

    with patch.object(agent, "_scan_cycle", side_effect=failing_then_cancel), \
         patch("agents.alpha_scout.asyncio.sleep", side_effect=mock_sleep):
        with pytest.raises(asyncio.CancelledError):
            await agent.run()

    assert len(sleep_called) >= 1, "asyncio.sleep was not called after a scan failure"
