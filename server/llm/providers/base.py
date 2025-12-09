"""
Murph - LLM Provider Base
Abstract base class for LLM providers.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..config import LLMConfig
    from ..types import LLMRequest, LLMResponse


@dataclass
class ProviderStats:
    """Statistics for an LLM provider."""

    calls: int = 0
    errors: int = 0
    total_latency_ms: float = 0.0
    total_tokens_in: int = 0
    total_tokens_out: int = 0

    @property
    def avg_latency_ms(self) -> float:
        """Average latency in milliseconds."""
        if self.calls == 0:
            return 0.0
        return self.total_latency_ms / self.calls

    def record_call(self, latency_ms: float, tokens_in: int = 0, tokens_out: int = 0) -> None:
        """Record a successful call."""
        self.calls += 1
        self.total_latency_ms += latency_ms
        self.total_tokens_in += tokens_in
        self.total_tokens_out += tokens_out

    def record_error(self) -> None:
        """Record a failed call."""
        self.errors += 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "calls": self.calls,
            "errors": self.errors,
            "avg_latency_ms": self.avg_latency_ms,
            "total_tokens_in": self.total_tokens_in,
            "total_tokens_out": self.total_tokens_out,
        }


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    All providers must implement:
    - complete(): Text completion
    - complete_with_vision(): Completion with image input
    - health_check(): Verify provider is available
    - is_available(): Quick availability check

    Providers should:
    - Track statistics via self._stats
    - Handle errors gracefully
    - Support async operations
    """

    def __init__(self, config: LLMConfig) -> None:
        """
        Initialize provider with configuration.

        Args:
            config: LLM configuration
        """
        self._config = config
        self._stats = ProviderStats()
        self._initialized = False
        self._last_health_check: float | None = None
        self._health_check_result: bool = False

    @abstractmethod
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """
        Send a completion request.

        Args:
            request: The completion request

        Returns:
            LLM response

        Raises:
            Exception: If the request fails
        """
        ...

    @abstractmethod
    async def complete_with_vision(
        self,
        request: LLMRequest,
        image: np.ndarray,
    ) -> LLMResponse:
        """
        Send a completion request with an image.

        Args:
            request: The completion request (last message describes the image)
            image: RGB numpy array (H, W, 3)

        Returns:
            LLM response

        Raises:
            Exception: If the request fails
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """
        Quick check if provider is available.

        This should not make network requests. Use health_check() for
        that.

        Returns:
            True if provider appears to be available
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Verify provider is reachable and working.

        Makes a network request to verify connectivity.

        Returns:
            True if provider is healthy
        """
        ...

    async def close(self) -> None:
        """
        Clean up resources.

        Override in subclasses if cleanup is needed.
        """
        pass

    def get_stats(self) -> dict[str, Any]:
        """Get provider statistics."""
        return self._stats.to_dict()

    @property
    def name(self) -> str:
        """Provider name for logging."""
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"{self.name}(available={self.is_available()})"
