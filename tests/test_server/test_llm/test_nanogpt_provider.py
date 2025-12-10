"""Tests for NanoGPT LLM provider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from server.llm.config import LLMConfig
from server.llm.providers.nanogpt import NanoGPTProvider
from server.llm.types import LLMMessage, LLMRequest

# Check for optional dependencies
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False


class TestNanoGPTProviderInit:
    """Tests for NanoGPTProvider initialization."""

    def test_init_with_config(self):
        """Test initialization with config."""
        config = LLMConfig(
            nanogpt_api_key="test-api-key",
            nanogpt_base_url="https://nano-gpt.com/api/v1",
            nanogpt_model="gpt-4o-mini",
            nanogpt_vision_model="gpt-4o-mini",
        )
        provider = NanoGPTProvider(config)

        assert provider._base_url == "https://nano-gpt.com/api/v1"
        assert provider._api_key == "test-api-key"
        assert provider._text_model == "gpt-4o-mini"
        assert provider._vision_model == "gpt-4o-mini"
        assert provider._session is None

    def test_init_strips_trailing_slash(self):
        """Test URL trailing slash is stripped."""
        config = LLMConfig(nanogpt_base_url="https://nano-gpt.com/api/v1/")
        provider = NanoGPTProvider(config)

        assert provider._base_url == "https://nano-gpt.com/api/v1"

    def test_repr_hides_api_key(self):
        """Test string representation hides API key."""
        config = LLMConfig(nanogpt_api_key="secret-key")
        provider = NanoGPTProvider(config)

        repr_str = repr(provider)
        assert "secret-key" not in repr_str
        assert "***" in repr_str

    def test_repr_shows_none_for_no_key(self):
        """Test repr shows None when no API key."""
        config = LLMConfig(nanogpt_api_key=None)
        provider = NanoGPTProvider(config)

        repr_str = repr(provider)
        assert "api_key=None" in repr_str


class TestNanoGPTProviderAvailability:
    """Tests for provider availability."""

    def test_is_available_with_api_key(self):
        """Test is_available returns True with API key."""
        config = LLMConfig(nanogpt_api_key="test-key")
        provider = NanoGPTProvider(config)

        assert provider.is_available() is True

    def test_is_available_without_api_key(self):
        """Test is_available returns False without API key."""
        config = LLMConfig(nanogpt_api_key=None)
        provider = NanoGPTProvider(config)

        assert provider.is_available() is False

    def test_is_available_with_empty_api_key(self):
        """Test is_available returns False with empty API key."""
        config = LLMConfig(nanogpt_api_key="")
        provider = NanoGPTProvider(config)

        assert provider.is_available() is False


class TestNanoGPTProviderComplete:
    """Tests for complete method."""

    @pytest.fixture
    def provider(self):
        """Create NanoGPT provider."""
        config = LLMConfig(
            nanogpt_api_key="test-key",
            nanogpt_base_url="https://nano-gpt.com/api/v1",
            nanogpt_model="gpt-4o-mini",
        )
        return NanoGPTProvider(config)

    @pytest.fixture
    def mock_response(self):
        """Create mock HTTP response."""
        response = AsyncMock()
        response.status = 200
        response.raise_for_status = MagicMock()
        response.json = AsyncMock(
            return_value={
                "choices": [
                    {"message": {"content": "Hello! How can I help?"}}
                ],
                "model": "gpt-4o-mini",
                "usage": {
                    "prompt_tokens": 15,
                    "completion_tokens": 25,
                    "total_tokens": 40,
                },
            }
        )
        return response

    @pytest.mark.asyncio
    async def test_complete_success(self, provider, mock_response):
        """Test successful completion request."""
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response)))

        with patch.object(provider, "_ensure_session", return_value=mock_session):
            request = LLMRequest(
                messages=[LLMMessage(role="user", content="Hello")],
                temperature=0.7,
                max_tokens=100,
            )
            response = await provider.complete(request)

        assert response.content == "Hello! How can I help?"
        assert response.model == "gpt-4o-mini"
        assert response.provider == "nanogpt"
        assert response.usage["prompt_tokens"] == 15
        assert response.usage["completion_tokens"] == 25

    @pytest.mark.asyncio
    async def test_complete_openai_format(self, provider, mock_response):
        """Test request uses OpenAI-compatible format."""
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response)))

        with patch.object(provider, "_ensure_session", return_value=mock_session):
            request = LLMRequest(
                messages=[
                    LLMMessage(role="system", content="You are helpful."),
                    LLMMessage(role="user", content="Hello"),
                ],
                temperature=0.5,
                max_tokens=200,
            )
            await provider.complete(request)

        call_args = mock_session.post.call_args
        payload = call_args[1]["json"]

        assert payload["model"] == "gpt-4o-mini"
        assert payload["temperature"] == 0.5
        assert payload["max_tokens"] == 200
        assert len(payload["messages"]) == 2

    @pytest.mark.asyncio
    async def test_complete_records_stats(self, provider, mock_response):
        """Test completion records statistics."""
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response)))

        initial_calls = provider._stats.calls

        with patch.object(provider, "_ensure_session", return_value=mock_session):
            request = LLMRequest(
                messages=[LLMMessage(role="user", content="Hello")],
            )
            await provider.complete(request)

        assert provider._stats.calls == initial_calls + 1

    @pytest.mark.asyncio
    async def test_complete_error_records_error_stat(self, provider):
        """Test completion error records error statistic."""
        mock_session = AsyncMock()
        mock_session.post = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(side_effect=Exception("API error"))
            )
        )

        initial_errors = provider._stats.errors

        with patch.object(provider, "_ensure_session", return_value=mock_session):
            request = LLMRequest(
                messages=[LLMMessage(role="user", content="Hello")],
            )
            with pytest.raises(Exception):
                await provider.complete(request)

        assert provider._stats.errors == initial_errors + 1


class TestNanoGPTProviderVision:
    """Tests for vision completion."""

    @pytest.fixture
    def provider(self):
        """Create NanoGPT provider."""
        config = LLMConfig(
            nanogpt_api_key="test-key",
            nanogpt_vision_model="gpt-4o-mini",
        )
        return NanoGPTProvider(config)

    @pytest.fixture
    def mock_vision_response(self):
        """Create mock vision response."""
        response = AsyncMock()
        response.status = 200
        response.raise_for_status = MagicMock()
        response.json = AsyncMock(
            return_value={
                "choices": [
                    {"message": {"content": "I see a red ball in the image."}}
                ],
                "model": "gpt-4o-mini",
                "usage": {
                    "prompt_tokens": 200,
                    "completion_tokens": 30,
                },
            }
        )
        return response

    @pytest.fixture
    def test_image(self):
        """Create test image."""
        return np.zeros((100, 100, 3), dtype=np.uint8)

    @pytest.mark.asyncio
    async def test_complete_with_vision_success(
        self, provider, mock_vision_response, test_image
    ):
        """Test successful vision completion."""
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_vision_response)))

        with patch.object(provider, "_ensure_session", return_value=mock_session):
            with patch.object(provider, "_encode_image", return_value="base64data"):
                request = LLMRequest(
                    messages=[LLMMessage(role="user", content="What do you see?")],
                )
                response = await provider.complete_with_vision(request, test_image)

        assert response.content == "I see a red ball in the image."
        assert response.provider == "nanogpt"

    @pytest.mark.asyncio
    async def test_complete_with_vision_multimodal_format(
        self, provider, mock_vision_response, test_image
    ):
        """Test vision request uses multimodal content format."""
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_vision_response)))

        with patch.object(provider, "_ensure_session", return_value=mock_session):
            with patch.object(provider, "_encode_image", return_value="base64data"):
                request = LLMRequest(
                    messages=[LLMMessage(role="user", content="What do you see?")],
                )
                await provider.complete_with_vision(request, test_image)

        call_args = mock_session.post.call_args
        payload = call_args[1]["json"]
        message = payload["messages"][0]

        # Should use multimodal content array format
        assert isinstance(message["content"], list)
        assert message["content"][0]["type"] == "text"
        assert message["content"][1]["type"] == "image_url"
        assert "data:image/jpeg;base64" in message["content"][1]["image_url"]["url"]

    @pytest.mark.asyncio
    async def test_complete_with_vision_only_last_user_message_has_image(
        self, provider, mock_vision_response, test_image
    ):
        """Test only the last user message gets the image."""
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_vision_response)))

        with patch.object(provider, "_ensure_session", return_value=mock_session):
            with patch.object(provider, "_encode_image", return_value="base64data"):
                request = LLMRequest(
                    messages=[
                        LLMMessage(role="system", content="You are helpful."),
                        LLMMessage(role="user", content="Look at this."),
                    ],
                )
                await provider.complete_with_vision(request, test_image)

        call_args = mock_session.post.call_args
        payload = call_args[1]["json"]

        # System message should be plain dict
        assert isinstance(payload["messages"][0]["content"], str)

        # Last user message should be multimodal
        assert isinstance(payload["messages"][1]["content"], list)


class TestNanoGPTProviderHealthCheck:
    """Tests for health check."""

    @pytest.fixture
    def provider(self):
        """Create NanoGPT provider."""
        config = LLMConfig(nanogpt_api_key="test-key")
        return NanoGPTProvider(config)

    @pytest.mark.asyncio
    async def test_health_check_success(self, provider):
        """Test health check returns True on success."""
        mock_response = AsyncMock()
        mock_response.status = 200

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response)))

        with patch.object(provider, "_ensure_session", return_value=mock_session):
            result = await provider.health_check()

        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_without_api_key(self):
        """Test health check returns False without API key."""
        config = LLMConfig(nanogpt_api_key=None)
        provider = NanoGPTProvider(config)

        result = await provider.health_check()

        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_connection_error(self, provider):
        """Test health check returns False on connection error."""
        mock_session = AsyncMock()
        mock_session.get = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(side_effect=Exception("Connection failed"))
            )
        )

        with patch.object(provider, "_ensure_session", return_value=mock_session):
            result = await provider.health_check()

        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_401_unauthorized(self, provider):
        """Test health check returns False on 401."""
        mock_response = AsyncMock()
        mock_response.status = 401

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response)))

        with patch.object(provider, "_ensure_session", return_value=mock_session):
            result = await provider.health_check()

        assert result is False


class TestNanoGPTProviderListModels:
    """Tests for model listing."""

    @pytest.fixture
    def provider(self):
        """Create NanoGPT provider."""
        config = LLMConfig(nanogpt_api_key="test-key")
        return NanoGPTProvider(config)

    @pytest.mark.asyncio
    async def test_list_models_success(self, provider):
        """Test listing models successfully."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(
            return_value={
                "data": [
                    {"id": "gpt-4o-mini", "created": 123456},
                    {"id": "claude-3-sonnet", "created": 123457},
                ]
            }
        )

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response)))

        with patch.object(provider, "_ensure_session", return_value=mock_session):
            models = await provider.list_models()

        assert len(models) == 2
        assert models[0]["id"] == "gpt-4o-mini"
        assert models[1]["id"] == "claude-3-sonnet"

    @pytest.mark.asyncio
    async def test_list_models_without_api_key(self):
        """Test listing models without API key returns empty."""
        config = LLMConfig(nanogpt_api_key=None)
        provider = NanoGPTProvider(config)

        models = await provider.list_models()

        assert models == []

    @pytest.mark.asyncio
    async def test_list_models_error(self, provider):
        """Test listing models on error returns empty list."""
        mock_session = AsyncMock()
        mock_session.get = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(side_effect=Exception("API error"))
            )
        )

        with patch.object(provider, "_ensure_session", return_value=mock_session):
            models = await provider.list_models()

        assert models == []


class TestNanoGPTProviderSession:
    """Tests for session management."""

    @pytest.fixture
    def provider(self):
        """Create NanoGPT provider."""
        config = LLMConfig(nanogpt_api_key="test-key")
        return NanoGPTProvider(config)

    @pytest.mark.asyncio
    @pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
    async def test_ensure_session_creates_with_auth(self, provider):
        """Test session includes Bearer token auth."""
        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_session.closed = False
            mock_session_class.return_value = mock_session

            await provider._ensure_session()

            # Verify headers include Authorization
            call_kwargs = mock_session_class.call_args[1]
            assert "Authorization" in call_kwargs["headers"]
            assert "Bearer test-key" in call_kwargs["headers"]["Authorization"]

    @pytest.mark.asyncio
    async def test_close_closes_session(self, provider):
        """Test close closes the session."""
        mock_session = AsyncMock()
        mock_session.closed = False
        provider._session = mock_session

        await provider.close()

        mock_session.close.assert_called_once()
        assert provider._session is None


class TestNanoGPTProviderEncodeImage:
    """Tests for image encoding."""

    @pytest.fixture
    def provider(self):
        """Create NanoGPT provider."""
        config = LLMConfig(nanogpt_api_key="test-key")
        return NanoGPTProvider(config)

    @pytest.mark.skipif(not HAS_CV2, reason="cv2 not installed")
    def test_encode_image_rgb(self, provider):
        """Test encoding RGB image."""
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        image[25, 25] = [255, 0, 0]  # Red pixel

        result = provider._encode_image(image)

        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.skipif(not HAS_CV2, reason="cv2 not installed")
    def test_encode_image_grayscale(self, provider):
        """Test encoding grayscale image."""
        image = np.zeros((50, 50), dtype=np.uint8)

        result = provider._encode_image(image)

        assert isinstance(result, str)
        assert len(result) > 0
