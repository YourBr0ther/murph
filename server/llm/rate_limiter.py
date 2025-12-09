"""
Murph - LLM Rate Limiter
Token bucket rate limiter for LLM requests.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RateLimiter:
    """
    Token bucket rate limiter.

    Limits the number of requests per minute using a token bucket algorithm.
    Tokens refill continuously over time.

    Usage:
        limiter = RateLimiter(max_requests_per_minute=20)
        if limiter.acquire():
            # Make request
            ...
        else:
            # Rate limited, try again later
            ...
    """

    max_requests_per_minute: int = 20
    _tokens: float = field(init=False)
    _last_refill: float = field(init=False)
    _total_acquired: int = field(init=False, default=0)
    _total_rejected: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        """Initialize tokens to max."""
        self._tokens = float(self.max_requests_per_minute)
        self._last_refill = time.time()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_refill

        # Calculate tokens to add (max_rpm / 60 tokens per second)
        tokens_per_second = self.max_requests_per_minute / 60.0
        refill = elapsed * tokens_per_second

        # Add tokens up to max
        self._tokens = min(self.max_requests_per_minute, self._tokens + refill)
        self._last_refill = now

    def acquire(self) -> bool:
        """
        Try to acquire a token.

        Returns:
            True if token acquired (request allowed), False if rate limited
        """
        self._refill()

        if self._tokens >= 1.0:
            self._tokens -= 1.0
            self._total_acquired += 1
            return True

        self._total_rejected += 1
        return False

    def wait_time(self) -> float:
        """
        Get time to wait until next token is available.

        Returns:
            Seconds to wait (0 if token available now)
        """
        self._refill()

        if self._tokens >= 1.0:
            return 0.0

        # Calculate time to get 1 token
        tokens_needed = 1.0 - self._tokens
        tokens_per_second = self.max_requests_per_minute / 60.0
        return tokens_needed / tokens_per_second

    @property
    def available_tokens(self) -> float:
        """Get current available tokens."""
        self._refill()
        return self._tokens

    def reset(self) -> None:
        """Reset to full tokens."""
        self._tokens = float(self.max_requests_per_minute)
        self._last_refill = time.time()

    def get_stats(self) -> dict[str, Any]:
        """Get rate limiter statistics."""
        self._refill()
        return {
            "max_rpm": self.max_requests_per_minute,
            "available_tokens": self._tokens,
            "total_acquired": self._total_acquired,
            "total_rejected": self._total_rejected,
            "rejection_rate": (
                self._total_rejected / max(1, self._total_acquired + self._total_rejected)
            ),
        }

    def __repr__(self) -> str:
        return (
            f"RateLimiter(max_rpm={self.max_requests_per_minute}, "
            f"tokens={self._tokens:.1f})"
        )
