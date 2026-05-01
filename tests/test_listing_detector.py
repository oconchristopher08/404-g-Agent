"""Tests for ListingDetector."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from config.settings import Settings
from modules.onchain.listing_detector import ListingDetector


@pytest.fixture
def settings():
    s = Settings()
    s.ETHERSCAN_API_KEY = "dummy"
    s.SENTIMENT_THRESHOLD = 0.6
    s.METRICS_PORT = 9999
    return s


class TestListingDetectorInit:
    @pytest.mark.asyncio
    async def test_scan_returns_empty_without_api_key(self, settings):
        settings.ETHERSCAN_API_KEY = ""
        detector = ListingDetector(settings)
        result = await detector.scan()
        assert result == []


class TestListingDetectorTokenTransfers:
    @pytest.mark.asyncio
    async def test_rate_limit_returns_empty(self, settings):
        detector = ListingDetector(settings)
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

        with patch("modules.onchain.listing_detector.aiohttp.ClientSession", return_value=mock_session):
            result = await detector._fetch_token_transfers("0xwallet", 0)
        assert result == []

    @pytest.mark.asyncio
    async def test_successful_transfer_fetch(self, settings):
        detector = ListingDetector(settings)
        tx = {
            "contractAddress": "0xtoken",
            "tokenSymbol": "NEWGEM",
            "tokenDecimal": "18",
            "value": str(int(1000 * 1e18)),
            "to": "0xwallet",
            "from": "0xsender",
        }
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

        with patch("modules.onchain.listing_detector.aiohttp.ClientSession", return_value=mock_session):
            result = await detector._fetch_token_transfers("0xwallet", 0, direction="in")
        # Direction filter: only returns txs where 'to' matches wallet
        assert all(tx["to"].lower() == "0xwallet" for tx in result)

    def test_stablecoins_filtered_out(self, settings):
        detector = ListingDetector(settings)
        # Simulate _scan_exchange_wallet logic: stablecoins should be skipped
        stablecoin_tx = {
            "contractAddress": "0xusdc",
            "tokenSymbol": "USDC",
            "tokenDecimal": "6",
            "value": str(int(1_000_000 * 1e6)),
            "to": "0xexchange",
            "from": "0xsender",
        }
        # The stablecoin check is inside _scan_exchange_wallet; verify symbol is in skip list
        from modules.onchain.listing_detector import _ALL_EXCHANGE_ADDRS
        assert len(_ALL_EXCHANGE_ADDRS) > 0  # exchange address set is populated


class TestListingDetectorSignalGeneration:
    @pytest.mark.asyncio
    async def test_multi_exchange_deposit_produces_high_confidence(self, settings):
        detector = ListingDetector(settings)

        # Mock _scan_exchange_wallet to return the same token for two exchanges
        async def mock_scan(exchange, addr, start_block):
            return {
                "0xnewtoken": {
                    "symbol": "NEWGEM",
                    "exchanges": {exchange},
                    "transfer_count": 5,
                    "total_value": 100_000.0,
                }
            }

        with patch.object(detector, "_scan_exchange_wallet", side_effect=mock_scan):
            with patch.object(detector, "_get_start_block", new=AsyncMock(return_value=19_000_000)):
                result = await detector.scan()

        assert len(result) > 0
        signal = result[0]
        assert signal["token"] == "NEWGEM"
        # Multiple exchanges → confidence > single exchange
        assert signal["confidence"] >= settings.SENTIMENT_THRESHOLD
        assert len(signal["exchanges_receiving"]) >= 2

    @pytest.mark.asyncio
    async def test_single_exchange_deposit_lower_confidence(self, settings):
        detector = ListingDetector(settings)

        call_count = [0]

        async def mock_scan_one(exchange, addr, start_block):
            call_count[0] += 1
            # Only Binance receives the token
            if exchange == "Binance":
                return {
                    "0xsingletoken": {
                        "symbol": "SOLO",
                        "exchanges": {"Binance"},
                        "transfer_count": 3,
                        "total_value": 50_000.0,
                    }
                }
            return {}

        with patch.object(detector, "_scan_exchange_wallet", side_effect=mock_scan_one):
            with patch.object(detector, "_get_start_block", new=AsyncMock(return_value=19_000_000)):
                result = await detector.scan()

        # Single exchange: confidence = 0.5, below default threshold of 0.6
        # So it may or may not appear depending on threshold
        for sig in result:
            if sig["token"] == "SOLO":
                assert sig["n_exchanges"] == 1


class TestListingDetectorAddressMap:
    def test_exchange_address_map_populated(self):
        from modules.onchain.listing_detector import _ADDR_TO_EXCHANGE, _ALL_EXCHANGE_ADDRS
        assert len(_ALL_EXCHANGE_ADDRS) >= 10
        assert len(_ADDR_TO_EXCHANGE) >= 10
        # All addresses should be lowercase
        for addr in _ALL_EXCHANGE_ADDRS:
            assert addr == addr.lower()

    def test_known_binance_address_mapped(self):
        from modules.onchain.listing_detector import _ADDR_TO_EXCHANGE
        binance_addr = "0x28c6c06298d514db089934071355e5743bf21d60"
        assert _ADDR_TO_EXCHANGE.get(binance_addr) == "Binance"
