"""
Basic unit tests for 404-g Agent modules.
"""

import pytest
from unittest.mock import AsyncMock, patch
from config.settings import Settings
from modules.tokens.detector import TokenDetector


@pytest.fixture
def settings():
    return Settings()


@pytest.mark.asyncio
async def test_token_detector_score_high_signal(settings):
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


@pytest.mark.asyncio
async def test_token_detector_score_low_signal(settings):
    detector = TokenDetector(settings)
    pair = {
        "priceChange": {"h24": "2"},
        "volume": {"h24": "1000"},
    }
    score = detector._score_pair(pair)
    assert score == 0.0


@pytest.mark.asyncio
async def test_token_detector_scan_returns_list(settings):
    detector = TokenDetector(settings)
    with patch.object(detector, "_fetch_trending_pairs", new=AsyncMock(return_value=[])):
        result = await detector.scan()
        assert isinstance(result, list)
