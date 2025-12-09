"""Tests for LLM service."""

import pytest

from server.llm.config import LLMConfig
from server.llm.services.llm_service import LLMService
from server.llm.types import LLMMessage


class TestLLMService:
    """Tests for LLMService."""

    @pytest.fixture
    def mock_config(self) -> LLMConfig:
        """Create mock provider configuration."""
        return LLMConfig(
            provider="mock",
            vision_enabled=True,
            reasoning_enabled=True,
            max_requests_per_minute=100,
            cache_ttl_seconds=10.0,
        )

    @pytest.fixture
    def service(self, mock_config: LLMConfig) -> LLMService:
        """Create LLM service with mock provider."""
        return LLMService(mock_config)

    def test_init(self, service: LLMService) -> None:
        """Test service initialization."""
        assert service.provider_name == "mock"
        assert service.is_initialized is False
        assert service.is_available is True  # Mock always available

    @pytest.mark.asyncio
    async def test_complete_initializes_on_first_call(self, service: LLMService) -> None:
        """Test service initializes lazily on first call."""
        assert service.is_initialized is False

        messages = [LLMMessage(role="user", content="Hello")]
        response = await service.complete(messages)

        assert service.is_initialized is True
        assert response is not None

    @pytest.mark.asyncio
    async def test_complete_returns_response(self, service: LLMService) -> None:
        """Test complete returns proper response."""
        messages = [LLMMessage(role="user", content="Hello")]

        response = await service.complete(messages)

        assert response is not None
        assert response.content is not None
        assert response.provider == "mock"

    @pytest.mark.asyncio
    async def test_complete_with_caching(self, service: LLMService) -> None:
        """Test response caching."""
        messages = [LLMMessage(role="user", content="Hello")]

        # First call - not cached
        r1 = await service.complete(messages, cache_key="test")
        assert r1.cached is False

        # Second call with same key - should be cached
        r2 = await service.complete(messages, cache_key="test")
        assert r2.cached is True
        assert r2.content == r1.content

    @pytest.mark.asyncio
    async def test_complete_skip_cache(self, service: LLMService) -> None:
        """Test skipping cache."""
        messages = [LLMMessage(role="user", content="Hello")]

        await service.complete(messages, cache_key="test")
        r2 = await service.complete(messages, cache_key="test", skip_cache=True)

        assert r2.cached is False

    @pytest.mark.asyncio
    async def test_rate_limiting(self) -> None:
        """Test rate limiting."""
        config = LLMConfig(
            provider="mock",
            max_requests_per_minute=2,
        )
        service = LLMService(config)

        messages = [LLMMessage(role="user", content="Hello")]

        r1 = await service.complete(messages)
        r2 = await service.complete(messages)
        r3 = await service.complete(messages)  # Should be rate limited

        assert r1 is not None
        assert r2 is not None
        assert r3 is None  # Rate limited

    @pytest.mark.asyncio
    async def test_stats_tracking(self, service: LLMService) -> None:
        """Test statistics tracking."""
        messages = [LLMMessage(role="user", content="Hello")]

        await service.complete(messages, cache_key="a")
        await service.complete(messages, cache_key="a")  # Cache hit
        await service.complete(messages, cache_key="b")

        stats = service.get_stats()
        assert stats["requests"] == 3
        assert stats["cache_hits"] == 1

    @pytest.mark.asyncio
    async def test_graceful_degradation_unavailable_provider(self) -> None:
        """Test graceful degradation when provider unavailable."""
        config = LLMConfig(
            provider="nanogpt",
            nanogpt_api_key=None,  # Invalid - no key
        )
        service = LLMService(config)

        messages = [LLMMessage(role="user", content="Hello")]
        response = await service.complete(messages)

        # Should return None, not raise exception
        assert response is None

    @pytest.mark.asyncio
    async def test_close(self, service: LLMService) -> None:
        """Test cleanup on close."""
        messages = [LLMMessage(role="user", content="Hello")]
        await service.complete(messages)

        await service.close()

        assert service._provider is None

    def test_repr(self, service: LLMService) -> None:
        """Test string representation."""
        repr_str = repr(service)
        assert "LLMService" in repr_str
        assert "mock" in repr_str
