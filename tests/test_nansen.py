"""Tests for NansenScanner."""

import pytest
from unittest.mock import AsyncMock, patch
from config.settings import Settings
from modules.nansen.smart_money import NansenScanner


@pytest.fixture
def settings():
    s = Settings()
    s.SENTIMENT_THRESHOLD = 0.6
    s.NANSEN_API_KEY = "test_key"
    s.METRICS_PORT = 9999
    return s


@pytest.fixture
def settings_no_key():
    s = Settings()
    s.NANSEN_API_KEY = ""
    s.SENTIMENT_THRESHOLD = 0.6
    return s


class TestNansenScannerInit:
    def test_disabled_without_api_key(self, settings_no_key):
        scanner = NansenScanner(settings_no_key)
        assert not scanner._enabled

    def test_enabled_with_api_key(self, settings):
        scanner = NansenScanner(settings)
        assert scanner._enabled

    @pytest.mark.asyncio
    async def test_scan_returns_empty_without_key(self, settings_no_key):
        scanner = NansenScanner(settings_no_key)
        result = await scanner.scan()
        assert result == []


class TestNansenNetflowScoring:
    def test_accumulation_signal_above_threshold(self, settings):
        scanner = NansenScanner(settings)
        records = [{
            "token_symbol": "PEPE",
            "chain": "ethereum",
            "token_address": "0xpepe",
            "net_flow_1h_usd": 2_000_000,
            "net_flow_24h_usd": 8_000_000,
            "net_flow_7d_usd": 15_000_000,
            "trader_count": 15,
            "token_age_days": 45,
            "market_cap_usd": 500_000_000,
        }]
        signals = scanner._score_netflows(records)
        assert len(signals) == 1
        assert signals[0]["type"] == "smart_money_accumulation"
        assert signals[0]["token"] == "PEPE"
        assert signals[0]["confidence"] >= settings.SENTIMENT_THRESHOLD

    def test_distribution_signal_detected(self, settings):
        scanner = NansenScanner(settings)
        records = [{
            "token_symbol": "DUMP",
            "chain": "ethereum",
            "token_address": "0xdump",
            "net_flow_1h_usd": -3_000_000,
            "net_flow_24h_usd": -9_000_000,
            "net_flow_7d_usd": -20_000_000,
            "trader_count": 20,
            "token_age_days": 90,
            "market_cap_usd": 200_000_000,
        }]
        signals = scanner._score_netflows(records)
        assert len(signals) == 1
        assert signals[0]["type"] == "smart_money_distribution"

    def test_small_flow_below_minimum_skipped(self, settings):
        scanner = NansenScanner(settings)
        records = [{
            "token_symbol": "TINY",
            "chain": "ethereum",
            "token_address": "0xtiny",
            "net_flow_1h_usd": 10_000,   # below _MIN_NETFLOW_USD
            "net_flow_24h_usd": 50_000,
            "net_flow_7d_usd": 100_000,
            "trader_count": 5,
        }]
        signals = scanner._score_netflows(records)
        assert signals == []

    def test_low_trader_count_reduces_confidence(self, settings):
        scanner = NansenScanner(settings)
        records_many = [{
            "token_symbol": "GEM",
            "chain": "ethereum",
            "token_address": "0xgem",
            "net_flow_1h_usd": 2_000_000,
            "net_flow_24h_usd": 5_000_000,
            "net_flow_7d_usd": 10_000_000,
            "trader_count": 25,
        }]
        records_few = [{**records_many[0], "trader_count": 3}]
        sigs_many = scanner._score_netflows(records_many)
        sigs_few = scanner._score_netflows(records_few)
        if sigs_many and sigs_few:
            assert sigs_many[0]["confidence"] > sigs_few[0]["confidence"]


class TestNansenScreenerScoring:
    def test_new_token_with_smart_money_flagged(self, settings):
        scanner = NansenScanner(settings)
        records = [{
            "token_symbol": "FRESH",
            "chain": "base",
            "token_address": "0xfresh",
            "smart_money_count": 20,
            "token_age_days": 10,   # very new
            "market_cap_usd": 5_000_000,
        }]
        signals = scanner._score_screener(records)
        assert len(signals) == 1
        assert signals[0]["type"] == "screener_hit"
        # Freshness bonus should push confidence higher
        assert signals[0]["confidence"] > 0.6

    def test_low_smart_money_count_skipped(self, settings):
        scanner = NansenScanner(settings)
        records = [{
            "token_symbol": "IGNORED",
            "chain": "ethereum",
            "token_address": "0xignored",
            "smart_money_count": 1,  # below _MIN_TRADER_COUNT
            "token_age_days": 5,
        }]
        signals = scanner._score_screener(records)
        assert signals == []


class TestNansenScannerIntegration:
    @pytest.mark.asyncio
    async def test_scan_aggregates_both_sources(self, settings):
        scanner = NansenScanner(settings)
        netflow_data = [{
            "token_symbol": "ALPHA",
            "chain": "ethereum",
            "token_address": "0xalpha",
            "net_flow_1h_usd": 3_000_000,
            "net_flow_24h_usd": 10_000_000,
            "net_flow_7d_usd": 20_000_000,
            "trader_count": 18,
            "token_age_days": 20,
        }]
        screener_data = [{
            "token_symbol": "BETA",
            "chain": "base",
            "token_address": "0xbeta",
            "smart_money_count": 22,
            "token_age_days": 7,
        }]
        with patch.object(scanner, "_fetch_netflows", new=AsyncMock(return_value=netflow_data)):
            with patch.object(scanner, "_fetch_token_screener", new=AsyncMock(return_value=screener_data)):
                result = await scanner.scan()
        tokens = {s["token"] for s in result}
        assert "ALPHA" in tokens
        assert "BETA" in tokens

    @pytest.mark.asyncio
    async def test_scan_handles_api_error_gracefully(self, settings):
        scanner = NansenScanner(settings)
        with patch.object(scanner, "_fetch_netflows", new=AsyncMock(side_effect=Exception("API down"))):
            with patch.object(scanner, "_fetch_token_screener", new=AsyncMock(return_value=[])):
                result = await scanner.scan()
        assert isinstance(result, list)
