"""Tests for mock LLM provider."""

import json

import numpy as np
import pytest

from server.llm.config import LLMConfig
from server.llm.providers.mock import MockProvider
from server.llm.types import LLMMessage, LLMRequest


class TestMockProvider:
    """Tests for MockProvider."""

    @pytest.fixture
    def config(self) -> LLMConfig:
        """Create test configuration."""
        return LLMConfig(provider="mock")

    @pytest.fixture
    def provider(self, config: LLMConfig) -> MockProvider:
        """Create mock provider instance."""
        return MockProvider(config)

    def test_init(self, provider: MockProvider) -> None:
        """Test provider initialization."""
        assert provider.is_available() is True
        assert provider._simulated_latency_ms == 50.0

    @pytest.mark.asyncio
    async def test_complete_default_response(self, provider: MockProvider) -> None:
        """Test complete returns default response."""
        request = LLMRequest(
            messages=[LLMMessage(role="user", content="Hello")]
        )

        response = await provider.complete(request)

        assert response is not None
        assert response.provider == "mock"
        assert response.model == "mock-model"
        # Default response should be valid JSON
        data = json.loads(response.content)
        assert "description" in data

    @pytest.mark.asyncio
    async def test_complete_custom_responses(self, provider: MockProvider) -> None:
        """Test complete with custom response queue."""
        provider.set_responses(["response 1", "response 2", "response 3"])

        request = LLMRequest(
            messages=[LLMMessage(role="user", content="Hello")]
        )

        r1 = await provider.complete(request)
        r2 = await provider.complete(request)
        r3 = await provider.complete(request)

        assert r1.content == "response 1"
        assert r2.content == "response 2"
        assert r3.content == "response 3"

    @pytest.mark.asyncio
    async def test_complete_cycles_responses(self, provider: MockProvider) -> None:
        """Test response queue cycles."""
        provider.set_responses(["a", "b"])

        request = LLMRequest(
            messages=[LLMMessage(role="user", content="Hello")]
        )

        r1 = await provider.complete(request)
        r2 = await provider.complete(request)
        r3 = await provider.complete(request)  # Should cycle back to "a"

        assert r1.content == "a"
        assert r2.content == "b"
        assert r3.content == "a"

    @pytest.mark.asyncio
    async def test_complete_with_vision(self, provider: MockProvider) -> None:
        """Test vision completion."""
        request = LLMRequest(
            messages=[LLMMessage(role="user", content="What do you see?")]
        )
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        response = await provider.complete_with_vision(request, image)

        assert response is not None
        assert response.model == "mock-vision-model"

    @pytest.mark.asyncio
    async def test_call_history(self, provider: MockProvider) -> None:
        """Test call history tracking."""
        request = LLMRequest(
            messages=[LLMMessage(role="user", content="Test")]
        )

        await provider.complete(request)
        await provider.complete(request)

        history = provider.get_call_history()
        assert len(history) == 2
        assert all(h["type"] == "complete" for h in history)

    @pytest.mark.asyncio
    async def test_call_history_with_vision(self, provider: MockProvider) -> None:
        """Test call history includes vision calls."""
        request = LLMRequest(
            messages=[LLMMessage(role="user", content="Test")]
        )
        image = np.zeros((50, 50, 3), dtype=np.uint8)

        await provider.complete_with_vision(request, image)

        history = provider.get_call_history()
        assert len(history) == 1
        assert history[0]["type"] == "complete_with_vision"
        assert history[0]["image_shape"] == (50, 50, 3)

    def test_set_available(self, provider: MockProvider) -> None:
        """Test setting availability."""
        assert provider.is_available() is True

        provider.set_available(False)
        assert provider.is_available() is False

    @pytest.mark.asyncio
    async def test_health_check(self, provider: MockProvider) -> None:
        """Test health check."""
        assert await provider.health_check() is True

        provider.set_healthy(False)
        assert await provider.health_check() is False

    def test_clear_history(self, provider: MockProvider) -> None:
        """Test clearing call history."""
        provider._call_history.append({"type": "test"})
        assert len(provider.get_call_history()) == 1

        provider.clear_history()
        assert len(provider.get_call_history()) == 0

    @pytest.mark.asyncio
    async def test_stats_tracking(self, provider: MockProvider) -> None:
        """Test statistics are tracked."""
        request = LLMRequest(
            messages=[LLMMessage(role="user", content="Test")]
        )

        await provider.complete(request)
        await provider.complete(request)

        stats = provider.get_stats()
        assert stats["calls"] == 2
        assert stats["avg_latency_ms"] > 0
