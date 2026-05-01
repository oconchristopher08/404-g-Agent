"""Tests for HyperliquidScanner."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from config.settings import Settings
from modules.hyperliquid.scanner import HyperliquidScanner


@pytest.fixture
def settings():
    s = Settings()
    s.SENTIMENT_THRESHOLD = 0.6
    s.HL_MIN_OI_USD = 5_000_000
    s.HL_FUNDING_THRESHOLD = 0.0003
    s.METRICS_PORT = 9999
    return s


def _make_meta_ctxs(coins, ctxs):
    """Build the [universe, asset_ctxs] structure returned by metaAndAssetCtxs."""
    universe = [{"name": c, "szDecimals": 4, "maxLeverage": 50} for c in coins]
    return [{"universe": universe, "marginTables": []}, ctxs]


class TestHyperliquidScannerFundingSignals:
    def test_extreme_funding_flagged(self, settings):
        scanner = HyperliquidScanner(settings)
        coins = ["NEWCOIN"]
        ctxs = [{
            "funding": "0.0010",   # 10x above threshold
            "markPx": "5.0",
            "openInterest": "2000000",  # $10M OI
            "prevDayPx": "4.5",
            "dayNtlVlm": "1000000",
        }]
        signals = scanner._detect_funding_signals(coins, ctxs, [])
        assert len(signals) == 1
        assert signals[0]["type"] == "extreme_funding"
        assert signals[0]["token"] == "NEWCOIN"
        assert signals[0]["direction"] == "longs paying"

    def test_negative_funding_flagged_as_shorts_paying(self, settings):
        scanner = HyperliquidScanner(settings)
        coins = ["BEAR"]
        ctxs = [{
            "funding": "-0.0008",
            "markPx": "10.0",
            "openInterest": "1000000",  # $10M OI
            "prevDayPx": "10.0",
            "dayNtlVlm": "500000",
        }]
        signals = scanner._detect_funding_signals(coins, ctxs, [])
        assert any(s["direction"] == "shorts paying" for s in signals)

    def test_below_threshold_not_flagged(self, settings):
        scanner = HyperliquidScanner(settings)
        coins = ["QUIET"]
        ctxs = [{
            "funding": "0.0001",   # below 0.0003 threshold
            "markPx": "5.0",
            "openInterest": "2000000",
            "prevDayPx": "5.0",
            "dayNtlVlm": "1000000",
        }]
        signals = scanner._detect_funding_signals(coins, ctxs, [])
        assert signals == []

    def test_low_oi_skipped(self, settings):
        scanner = HyperliquidScanner(settings)
        coins = ["TINY"]
        ctxs = [{
            "funding": "0.0020",   # extreme rate
            "markPx": "1.0",
            "openInterest": "100",  # only $100 OI — below min
            "prevDayPx": "1.0",
            "dayNtlVlm": "1000",
        }]
        signals = scanner._detect_funding_signals(coins, ctxs, [])
        assert signals == []

    def test_funding_divergence_detected(self, settings):
        scanner = HyperliquidScanner(settings)
        coins = ["ARB"]
        ctxs = [{
            "funding": "0.0015",   # HL rate
            "markPx": "2.0",
            "openInterest": "5000000",  # $10M OI
            "prevDayPx": "2.0",
            "dayNtlVlm": "2000000",
        }]
        # CEX average is 0.0003 — divergence = 4x
        predicted = [
            ["ARB", [
                ["BinPerp", {"fundingRate": "0.0003", "nextFundingTime": 9999}],
                ["BybitPerp", {"fundingRate": "0.0003", "nextFundingTime": 9999}],
            ]]
        ]
        signals = scanner._detect_funding_signals(coins, ctxs, predicted)
        divergence_sigs = [s for s in signals if s["type"] == "funding_divergence"]
        assert len(divergence_sigs) == 1
        assert divergence_sigs[0]["divergence_ratio"] > 1.5


class TestHyperliquidScannerOISignals:
    def test_pre_listing_oi_flagged_for_non_major_coin(self, settings):
        scanner = HyperliquidScanner(settings)
        coins = ["NEWGEM"]
        ctxs = [{
            "markPx": "1.0",
            "openInterest": "15000000",  # $15M OI — 3x min
            "prevDayPx": "0.9",
            "dayNtlVlm": "3000000",
            "funding": "0.0001",
        }]
        signals = scanner._detect_oi_signals(coins, ctxs)
        assert len(signals) == 1
        assert signals[0]["type"] == "pre_listing_oi"
        assert signals[0]["token"] == "NEWGEM"

    def test_major_coin_not_flagged_as_pre_listing(self, settings):
        scanner = HyperliquidScanner(settings)
        coins = ["BTC"]
        ctxs = [{
            "markPx": "60000.0",
            "openInterest": "1000",  # $60M OI
            "prevDayPx": "59000.0",
            "dayNtlVlm": "100000000",
            "funding": "0.0001",
        }]
        signals = scanner._detect_oi_signals(coins, ctxs)
        assert signals == []


class TestHyperliquidScannerTopMovers:
    def test_top_mover_above_10pct(self, settings):
        scanner = HyperliquidScanner(settings)
        coins = ["MOON", "FLAT"]
        ctxs = [
            {
                "markPx": "1.20",
                "prevDayPx": "1.00",
                "openInterest": "5000000",  # $6M OI
                "dayNtlVlm": "2000000",
                "funding": "0.0001",
            },
            {
                "markPx": "1.00",
                "prevDayPx": "1.00",
                "openInterest": "5000000",
                "dayNtlVlm": "1000000",
                "funding": "0.0001",
            },
        ]
        signals = scanner._detect_top_movers(coins, ctxs)
        assert len(signals) == 1
        assert signals[0]["token"] == "MOON"
        assert signals[0]["price_change_24h"] == pytest.approx(20.0)

    def test_mover_below_10pct_excluded(self, settings):
        scanner = HyperliquidScanner(settings)
        coins = ["SLOW"]
        ctxs = [{
            "markPx": "1.05",
            "prevDayPx": "1.00",
            "openInterest": "5000000",
            "dayNtlVlm": "1000000",
            "funding": "0.0001",
        }]
        signals = scanner._detect_top_movers(coins, ctxs)
        assert signals == []


class TestHyperliquidScannerIntegration:
    @pytest.mark.asyncio
    async def test_scan_returns_list_on_api_failure(self, settings):
        scanner = HyperliquidScanner(settings)
        with patch.object(scanner, "_fetch_market_data", new=AsyncMock(return_value=(None, []))):
            result = await scanner.scan()
        assert isinstance(result, list)
        assert result == []

    @pytest.mark.asyncio
    async def test_scan_returns_signals_from_all_detectors(self, settings):
        scanner = HyperliquidScanner(settings)
        coins = ["NEWGEM"]
        ctxs = [{
            "funding": "0.0015",
            "markPx": "2.0",
            "openInterest": "8000000",
            "prevDayPx": "1.6",
            "dayNtlVlm": "3000000",
        }]
        meta = _make_meta_ctxs(coins, ctxs)
        with patch.object(scanner, "_fetch_market_data", new=AsyncMock(return_value=(meta, []))):
            result = await scanner.scan()
        assert isinstance(result, list)
        assert len(result) > 0
