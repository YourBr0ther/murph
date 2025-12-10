"""Tests for Ollama LLM provider."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from server.llm.config import LLMConfig
from server.llm.providers.ollama import OllamaProvider
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


class TestOllamaProviderInit:
    """Tests for OllamaProvider initialization."""

    def test_init_with_config(self):
        """Test initialization with config."""
        config = LLMConfig(
            ollama_base_url="http://localhost:11434",
            ollama_model="llama3.2",
            ollama_vision_model="llama3.2-vision",
        )
        provider = OllamaProvider(config)

        assert provider._base_url == "http://localhost:11434"
        assert provider._text_model == "llama3.2"
        assert provider._vision_model == "llama3.2-vision"
        assert provider._session is None

    def test_init_strips_trailing_slash(self):
        """Test URL trailing slash is stripped."""
        config = LLMConfig(ollama_base_url="http://localhost:11434/")
        provider = OllamaProvider(config)

        assert provider._base_url == "http://localhost:11434"

    def test_repr(self):
        """Test string representation."""
        config = LLMConfig()
        provider = OllamaProvider(config)

        repr_str = repr(provider)
        assert "OllamaProvider" in repr_str
        assert "http://localhost:11434" in repr_str


class TestOllamaProviderAvailability:
    """Tests for provider availability."""

    def test_is_available_always_true(self):
        """Test is_available always returns True."""
        config = LLMConfig()
        provider = OllamaProvider(config)

        assert provider.is_available() is True


class TestOllamaProviderComplete:
    """Tests for complete method."""

    @pytest.fixture
    def provider(self):
        """Create Ollama provider."""
        config = LLMConfig(
            ollama_base_url="http://localhost:11434",
            ollama_model="llama3.2",
        )
        return OllamaProvider(config)

    @pytest.fixture
    def mock_response(self):
        """Create mock HTTP response."""
        response = AsyncMock()
        response.status = 200
        response.raise_for_status = MagicMock()
        response.json = AsyncMock(
            return_value={
                "message": {"content": "Hello! How can I help?"},
                "model": "llama3.2",
                "prompt_eval_count": 10,
                "eval_count": 20,
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
        assert response.model == "llama3.2"
        assert response.provider == "ollama"
        assert response.usage["prompt_tokens"] == 10
        assert response.usage["completion_tokens"] == 20
        assert response.usage["total_tokens"] == 30

    @pytest.mark.asyncio
    async def test_complete_with_model_override(self, provider, mock_response):
        """Test completion with model override."""
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response)))

        with patch.object(provider, "_ensure_session", return_value=mock_session):
            request = LLMRequest(
                messages=[LLMMessage(role="user", content="Hello")],
                model="different-model",
            )
            await provider.complete(request)

        # Verify model was passed in request
        call_args = mock_session.post.call_args
        assert call_args[1]["json"]["model"] == "different-model"

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
                __aenter__=AsyncMock(side_effect=Exception("Connection failed"))
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


class TestOllamaProviderVision:
    """Tests for vision completion."""

    @pytest.fixture
    def provider(self):
        """Create Ollama provider."""
        config = LLMConfig(
            ollama_base_url="http://localhost:11434",
            ollama_vision_model="llama3.2-vision",
        )
        return OllamaProvider(config)

    @pytest.fixture
    def mock_vision_response(self):
        """Create mock vision response."""
        response = AsyncMock()
        response.status = 200
        response.raise_for_status = MagicMock()
        response.json = AsyncMock(
            return_value={
                "message": {"content": "I see a cat in the image."},
                "model": "llama3.2-vision",
                "prompt_eval_count": 100,
                "eval_count": 30,
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

        assert response.content == "I see a cat in the image."
        assert response.model == "llama3.2-vision"
        assert response.provider == "ollama"

    @pytest.mark.asyncio
    async def test_complete_with_vision_attaches_image(
        self, provider, mock_vision_response, test_image
    ):
        """Test image is attached to the last user message."""
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_vision_response)))

        with patch.object(provider, "_ensure_session", return_value=mock_session):
            with patch.object(provider, "_encode_image", return_value="base64data"):
                request = LLMRequest(
                    messages=[
                        LLMMessage(role="system", content="You are helpful."),
                        LLMMessage(role="user", content="What do you see?"),
                    ],
                )
                await provider.complete_with_vision(request, test_image)

        # Verify image is attached to the last message
        call_args = mock_session.post.call_args
        messages = call_args[1]["json"]["messages"]
        assert "images" not in messages[0]  # System message has no image
        assert "images" in messages[1]  # User message has image
        assert messages[1]["images"] == ["base64data"]

    @pytest.mark.asyncio
    async def test_complete_with_vision_records_stats(
        self, provider, mock_vision_response, test_image
    ):
        """Test vision completion records statistics."""
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_vision_response)))

        initial_calls = provider._stats.calls

        with patch.object(provider, "_ensure_session", return_value=mock_session):
            with patch.object(provider, "_encode_image", return_value="base64data"):
                request = LLMRequest(
                    messages=[LLMMessage(role="user", content="What?")],
                )
                await provider.complete_with_vision(request, test_image)

        assert provider._stats.calls == initial_calls + 1


class TestOllamaProviderEncodeImage:
    """Tests for image encoding."""

    @pytest.fixture
    def provider(self):
        """Create Ollama provider."""
        config = LLMConfig()
        return OllamaProvider(config)

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


class TestOllamaProviderHealthCheck:
    """Tests for health check."""

    @pytest.fixture
    def provider(self):
        """Create Ollama provider."""
        config = LLMConfig()
        return OllamaProvider(config)

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
        assert provider._health_check_result is True

    @pytest.mark.asyncio
    async def test_health_check_failure_non_200(self, provider):
        """Test health check returns False on non-200."""
        mock_response = AsyncMock()
        mock_response.status = 500

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response)))

        with patch.object(provider, "_ensure_session", return_value=mock_session):
            result = await provider.health_check()

        assert result is False
        assert provider._health_check_result is False

    @pytest.mark.asyncio
    async def test_health_check_connection_error(self, provider):
        """Test health check returns False on connection error."""
        mock_session = AsyncMock()
        mock_session.get = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(side_effect=Exception("Connection refused"))
            )
        )

        with patch.object(provider, "_ensure_session", return_value=mock_session):
            result = await provider.health_check()

        assert result is False


class TestOllamaProviderListModels:
    """Tests for model listing."""

    @pytest.fixture
    def provider(self):
        """Create Ollama provider."""
        config = LLMConfig()
        return OllamaProvider(config)

    @pytest.mark.asyncio
    async def test_list_models_success(self, provider):
        """Test listing models successfully."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(
            return_value={
                "models": [
                    {"name": "llama3.2"},
                    {"name": "llama3.2-vision"},
                    {"name": "codellama"},
                ]
            }
        )

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response)))

        with patch.object(provider, "_ensure_session", return_value=mock_session):
            models = await provider.list_models()

        assert len(models) == 3
        assert "llama3.2" in models
        assert "llama3.2-vision" in models
        assert "codellama" in models

    @pytest.mark.asyncio
    async def test_list_models_empty(self, provider):
        """Test listing models with empty result."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(return_value={"models": []})

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_response)))

        with patch.object(provider, "_ensure_session", return_value=mock_session):
            models = await provider.list_models()

        assert models == []

    @pytest.mark.asyncio
    async def test_list_models_error(self, provider):
        """Test listing models on error returns empty list."""
        mock_session = AsyncMock()
        mock_session.get = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(side_effect=Exception("Server error"))
            )
        )

        with patch.object(provider, "_ensure_session", return_value=mock_session):
            models = await provider.list_models()

        assert models == []


class TestOllamaProviderSession:
    """Tests for session management."""

    @pytest.fixture
    def provider(self):
        """Create Ollama provider."""
        config = LLMConfig()
        return OllamaProvider(config)

    @pytest.mark.asyncio
    @pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
    async def test_ensure_session_creates_new(self, provider):
        """Test _ensure_session creates new session."""
        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_session.closed = False
            mock_session_class.return_value = mock_session

            result = await provider._ensure_session()

            assert result is mock_session
            assert provider._session is mock_session

    @pytest.mark.asyncio
    async def test_ensure_session_reuses_existing(self, provider):
        """Test _ensure_session reuses existing open session."""
        mock_session = MagicMock()
        mock_session.closed = False
        provider._session = mock_session

        result = await provider._ensure_session()

        assert result is mock_session

    @pytest.mark.asyncio
    async def test_close_closes_session(self, provider):
        """Test close closes the session."""
        mock_session = AsyncMock()
        mock_session.closed = False
        provider._session = mock_session

        await provider.close()

        mock_session.close.assert_called_once()
        assert provider._session is None

    @pytest.mark.asyncio
    async def test_close_handles_no_session(self, provider):
        """Test close handles no session gracefully."""
        provider._session = None

        # Should not raise
        await provider.close()
