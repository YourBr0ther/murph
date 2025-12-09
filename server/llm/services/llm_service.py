"""
Murph - LLM Service
Main LLM service orchestrating providers, caching, and rate limiting.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from ..cache import ResponseCache
from ..rate_limiter import RateLimiter

if TYPE_CHECKING:
    from ..config import LLMConfig
    from ..providers.base import LLMProvider
    from ..types import LLMMessage, LLMRequest, LLMResponse

logger = logging.getLogger(__name__)


class LLMService:
    """
    Main LLM service orchestrating providers and caching.

    Features:
    - Multi-provider support (Ollama, NanoGPT, Mock)
    - Lazy initialization (like VisionProcessor)
    - Response caching with TTL
    - Rate limiting
    - Graceful degradation (returns None when unavailable)
    - Statistics tracking

    Usage:
        service = LLMService(config)
        response = await service.complete(messages)
        if response:
            print(response.content)
    """

    def __init__(self, config: LLMConfig | None = None) -> None:
        """
        Initialize LLM service.

        Args:
            config: LLM configuration (uses env if None)
        """
        from ..config import LLMConfig

        self._config = config or LLMConfig.from_env()
        self._provider: LLMProvider | None = None
        self._cache: ResponseCache[LLMResponse] = ResponseCache(
            ttl_seconds=self._config.cache_ttl_seconds,
            max_entries=self._config.cache_max_entries,
        )
        self._rate_limiter = RateLimiter(
            max_requests_per_minute=self._config.max_requests_per_minute
        )

        # State
        self._initialized = False
        self._available = False

        # Stats
        self._requests = 0
        self._cache_hits = 0
        self._rate_limited = 0
        self._errors = 0

        logger.info(f"LLMService created: {self._config}")

    async def _ensure_initialized(self) -> bool:
        """
        Lazy-initialize the provider.

        Returns:
            True if provider is available
        """
        if self._initialized:
            return self._available

        try:
            self._provider = self._create_provider()
            self._available = await self._provider.health_check()
            self._initialized = True

            if self._available:
                logger.info(
                    f"LLMService initialized with {self._config.provider} provider"
                )
            else:
                logger.warning(
                    f"LLM provider {self._config.provider} not available"
                )

            return self._available

        except Exception as e:
            logger.error(f"Failed to initialize LLM provider: {e}")
            self._initialized = True
            self._available = False
            return False

    def _create_provider(self) -> LLMProvider:
        """Create the appropriate provider based on config."""
        from ..providers import MockProvider, NanoGPTProvider, OllamaProvider

        if self._config.provider == "nanogpt":
            return NanoGPTProvider(self._config)
        elif self._config.provider == "ollama":
            return OllamaProvider(self._config)
        elif self._config.provider == "mock":
            return MockProvider(self._config)
        else:
            raise ValueError(f"Unknown provider: {self._config.provider}")

    async def complete(
        self,
        messages: list[LLMMessage],
        cache_key: str | None = None,
        skip_cache: bool = False,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> LLMResponse | None:
        """
        Send a completion request with caching and rate limiting.

        Args:
            messages: List of chat messages
            cache_key: Optional cache key (uses message hash if None)
            skip_cache: Skip cache lookup
            model: Override model
            temperature: Sampling temperature
            max_tokens: Max tokens to generate

        Returns:
            LLMResponse, or None if unavailable/rate-limited
        """
        from ..types import LLMRequest, LLMResponse

        if not await self._ensure_initialized():
            return None

        self._requests += 1

        # Check cache
        if cache_key and not skip_cache:
            cached = self._cache.get(cache_key)
            if cached:
                self._cache_hits += 1
                # Mark as cached
                return LLMResponse(
                    content=cached.content,
                    model=cached.model,
                    usage=cached.usage,
                    latency_ms=0,
                    cached=True,
                    provider=cached.provider,
                )

        # Check rate limit
        if not self._rate_limiter.acquire():
            self._rate_limited += 1
            logger.warning("LLM request rate-limited")
            return None

        try:
            request = LLMRequest(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            response = await self._provider.complete(request)

            # Cache response
            if cache_key:
                self._cache.set(cache_key, response)

            return response

        except Exception as e:
            self._errors += 1
            logger.error(f"LLM completion error: {e}")
            return None

    async def complete_with_vision(
        self,
        messages: list[LLMMessage],
        image: np.ndarray,
        cache_key: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> LLMResponse | None:
        """
        Send a vision completion request.

        Vision requests are typically not cached since frames change.

        Args:
            messages: List of chat messages
            image: RGB numpy array (H, W, 3)
            cache_key: Optional cache key (typically None for vision)
            temperature: Sampling temperature
            max_tokens: Max tokens to generate

        Returns:
            LLMResponse, or None if unavailable/rate-limited
        """
        from ..types import LLMRequest

        if not await self._ensure_initialized():
            return None

        self._requests += 1

        # Check rate limit
        if not self._rate_limiter.acquire():
            self._rate_limited += 1
            logger.warning("LLM vision request rate-limited")
            return None

        try:
            request = LLMRequest(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return await self._provider.complete_with_vision(request, image)

        except Exception as e:
            self._errors += 1
            logger.error(f"LLM vision error: {e}")
            return None

    @property
    def is_available(self) -> bool:
        """Check if service is available (may not be initialized yet)."""
        if not self._initialized:
            # Quick check without network
            if self._config.provider == "nanogpt":
                return self._config.nanogpt_api_key is not None
            return True  # Ollama/Mock always potentially available
        return self._available

    @property
    def is_initialized(self) -> bool:
        """Check if service has been initialized."""
        return self._initialized

    @property
    def provider_name(self) -> str:
        """Get current provider name."""
        return self._config.provider

    def get_stats(self) -> dict[str, Any]:
        """Get service statistics."""
        stats = {
            "provider": self._config.provider,
            "initialized": self._initialized,
            "available": self._available,
            "requests": self._requests,
            "cache_hits": self._cache_hits,
            "rate_limited": self._rate_limited,
            "errors": self._errors,
            "cache": self._cache.get_stats(),
            "rate_limiter": self._rate_limiter.get_stats(),
        }
        if self._provider:
            stats["provider_stats"] = self._provider.get_stats()
        return stats

    async def close(self) -> None:
        """Clean up resources."""
        if self._provider:
            await self._provider.close()
            self._provider = None

        self._cache.clear()
        logger.info("LLMService closed")

    def __repr__(self) -> str:
        return (
            f"LLMService(provider={self._config.provider}, "
            f"initialized={self._initialized}, available={self._available})"
        )
