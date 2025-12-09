"""Tests for LLM rate limiter."""

import time

import pytest

from server.llm.rate_limiter import RateLimiter


class TestRateLimiter:
    """Tests for RateLimiter."""

    def test_init(self) -> None:
        """Test rate limiter initialization."""
        limiter = RateLimiter(max_requests_per_minute=60)

        assert limiter.max_requests_per_minute == 60
        assert limiter.available_tokens == 60

    def test_acquire_success(self) -> None:
        """Test successful token acquisition."""
        limiter = RateLimiter(max_requests_per_minute=10)

        assert limiter.acquire() is True
        assert limiter.available_tokens < 10

    def test_acquire_depleted(self) -> None:
        """Test acquiring when tokens depleted."""
        limiter = RateLimiter(max_requests_per_minute=2)

        assert limiter.acquire() is True
        assert limiter.acquire() is True
        assert limiter.acquire() is False  # Depleted

    def test_token_refill(self) -> None:
        """Test tokens refill over time."""
        limiter = RateLimiter(max_requests_per_minute=60)  # 1 per second

        # Deplete some tokens
        limiter.acquire()
        limiter.acquire()
        initial_tokens = limiter.available_tokens

        # Wait a bit for refill
        time.sleep(0.1)
        assert limiter.available_tokens > initial_tokens

    def test_wait_time(self) -> None:
        """Test wait time calculation."""
        limiter = RateLimiter(max_requests_per_minute=60)  # 1 per second

        # When tokens available
        assert limiter.wait_time() == 0.0

        # Deplete all tokens
        for _ in range(60):
            limiter.acquire()

        # Should have wait time
        wait = limiter.wait_time()
        assert wait > 0

    def test_reset(self) -> None:
        """Test reset to full tokens."""
        limiter = RateLimiter(max_requests_per_minute=10)

        # Deplete some tokens
        limiter.acquire()
        limiter.acquire()
        assert limiter.available_tokens < 10

        limiter.reset()
        assert limiter.available_tokens == 10

    def test_stats(self) -> None:
        """Test statistics tracking."""
        limiter = RateLimiter(max_requests_per_minute=2)

        limiter.acquire()  # Acquired
        limiter.acquire()  # Acquired
        limiter.acquire()  # Rejected

        stats = limiter.get_stats()
        assert stats["total_acquired"] == 2
        assert stats["total_rejected"] == 1
        assert stats["max_rpm"] == 2

    def test_repr(self) -> None:
        """Test string representation."""
        limiter = RateLimiter(max_requests_per_minute=30)

        repr_str = repr(limiter)
        assert "30" in repr_str
        assert "RateLimiter" in repr_str
