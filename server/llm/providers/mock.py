"""
Murph - Mock LLM Provider
Mock provider for testing without real LLM calls.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

import numpy as np

from .base import LLMProvider

if TYPE_CHECKING:
    from ..config import LLMConfig
    from ..types import LLMRequest, LLMResponse


# Default mock responses
DEFAULT_TEXT_RESPONSE = json.dumps({
    "description": "A test scene",
    "objects": ["table", "chair"],
    "activities": [],
    "mood": [],
    "triggers": [],
    "confidence": 0.8,
})

DEFAULT_BEHAVIOR_RESPONSE = json.dumps({
    "behavior": "idle",
    "reasoning": "No specific action needed",
    "confidence": 0.7,
    "alternatives": ["explore", "rest"],
})


class MockProvider(LLMProvider):
    """
    Mock LLM provider for testing.

    Features:
    - Configurable response queue
    - Simulated latency
    - Call recording for test assertions
    - Always available

    Usage:
        provider = MockProvider(config)
        provider.set_responses(["response1", "response2"])
        response = await provider.complete(request)  # Returns "response1"
        response = await provider.complete(request)  # Returns "response2"
    """

    def __init__(
        self,
        config: LLMConfig,
        default_response: str = DEFAULT_TEXT_RESPONSE,
        simulated_latency_ms: float = 50.0,
    ) -> None:
        """
        Initialize mock provider.

        Args:
            config: LLM configuration
            default_response: Default response when queue is empty
            simulated_latency_ms: Simulated latency per request
        """
        super().__init__(config)
        self._default_response = default_response
        self._simulated_latency_ms = simulated_latency_ms
        self._response_queue: list[str] = []
        self._response_index = 0
        self._call_history: list[dict] = []
        self._available = True
        self._healthy = True
        self._initialized = True

    def set_responses(self, responses: list[str]) -> None:
        """
        Set canned responses for upcoming requests.

        Args:
            responses: List of response strings (will cycle through)
        """
        self._response_queue = responses
        self._response_index = 0

    def set_available(self, available: bool) -> None:
        """Set whether provider appears available."""
        self._available = available

    def set_healthy(self, healthy: bool) -> None:
        """Set whether health check passes."""
        self._healthy = healthy

    def get_call_history(self) -> list[dict]:
        """Get history of all calls made."""
        return self._call_history.copy()

    def clear_history(self) -> None:
        """Clear call history."""
        self._call_history.clear()

    def _get_next_response(self) -> str:
        """Get next response from queue or default."""
        if not self._response_queue:
            return self._default_response

        response = self._response_queue[self._response_index % len(self._response_queue)]
        self._response_index += 1
        return response

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """
        Return mock completion.

        Args:
            request: The completion request

        Returns:
            Mock response from queue or default
        """
        from ..types import LLMResponse

        # Record call
        self._call_history.append({
            "type": "complete",
            "messages": [m.to_dict() for m in request.messages],
            "timestamp": time.time(),
        })

        # Simulate latency
        if self._simulated_latency_ms > 0:
            import asyncio
            await asyncio.sleep(self._simulated_latency_ms / 1000.0)

        content = self._get_next_response()

        # Update stats
        self._stats.record_call(
            latency_ms=self._simulated_latency_ms,
            tokens_in=sum(len(m.content.split()) for m in request.messages),
            tokens_out=len(content.split()),
        )

        return LLMResponse(
            content=content,
            model="mock-model",
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
            latency_ms=self._simulated_latency_ms,
            provider="mock",
        )

    async def complete_with_vision(
        self,
        request: LLMRequest,
        image: np.ndarray,
    ) -> LLMResponse:
        """
        Return mock vision completion.

        Args:
            request: The completion request
            image: RGB image array (ignored in mock)

        Returns:
            Mock response from queue or default
        """
        from ..types import LLMResponse

        # Record call with image shape
        self._call_history.append({
            "type": "complete_with_vision",
            "messages": [m.to_dict() for m in request.messages],
            "image_shape": image.shape if image is not None else None,
            "timestamp": time.time(),
        })

        # Simulate latency (vision typically slower)
        latency = self._simulated_latency_ms * 2
        if latency > 0:
            import asyncio
            await asyncio.sleep(latency / 1000.0)

        content = self._get_next_response()

        # Update stats
        self._stats.record_call(
            latency_ms=latency,
            tokens_in=sum(len(m.content.split()) for m in request.messages),
            tokens_out=len(content.split()),
        )

        return LLMResponse(
            content=content,
            model="mock-vision-model",
            usage={
                "prompt_tokens": 100,  # Image tokens
                "completion_tokens": 20,
                "total_tokens": 120,
            },
            latency_ms=latency,
            provider="mock",
        )

    def is_available(self) -> bool:
        """Check if mock provider is available."""
        return self._available

    async def health_check(self) -> bool:
        """Check if mock provider is healthy."""
        return self._healthy

    async def close(self) -> None:
        """No cleanup needed for mock."""
        pass

    def __repr__(self) -> str:
        return (
            f"MockProvider(available={self._available}, "
            f"responses={len(self._response_queue)}, "
            f"calls={len(self._call_history)})"
        )
