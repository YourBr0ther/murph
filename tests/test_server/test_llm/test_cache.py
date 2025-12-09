"""Tests for LLM response cache."""

import time

import pytest

from server.llm.cache import ResponseCache
from server.llm.types import LLMResponse


class TestResponseCache:
    """Tests for ResponseCache."""

    def test_init(self) -> None:
        """Test cache initialization."""
        cache: ResponseCache[str] = ResponseCache(ttl_seconds=10.0, max_entries=50)

        assert cache.size == 0
        assert cache._ttl == 10.0
        assert cache._max_entries == 50

    def test_set_and_get(self) -> None:
        """Test basic set and get operations."""
        cache: ResponseCache[str] = ResponseCache()

        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_get_nonexistent_key(self) -> None:
        """Test getting nonexistent key returns None."""
        cache: ResponseCache[str] = ResponseCache()

        result = cache.get("nonexistent")
        assert result is None

    def test_ttl_expiration(self) -> None:
        """Test entries expire after TTL."""
        cache: ResponseCache[str] = ResponseCache(ttl_seconds=0.1)

        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        # Wait for TTL to expire
        time.sleep(0.15)
        assert cache.get("key1") is None

    def test_lru_eviction(self) -> None:
        """Test LRU eviction when max entries exceeded."""
        cache: ResponseCache[str] = ResponseCache(max_entries=3)

        cache.set("a", "1")
        cache.set("b", "2")
        cache.set("c", "3")
        assert cache.size == 3

        # Access 'a' to make it recently used
        cache.get("a")

        # Add new entry, should evict 'b' (least recently used)
        cache.set("d", "4")
        assert cache.size == 3
        assert cache.get("a") == "1"  # Still there (recently used)
        assert cache.get("b") is None  # Evicted
        assert cache.get("c") == "3"
        assert cache.get("d") == "4"

    def test_contains(self) -> None:
        """Test __contains__ respects TTL."""
        cache: ResponseCache[str] = ResponseCache(ttl_seconds=0.1)

        cache.set("key1", "value1")
        assert "key1" in cache

        time.sleep(0.15)
        assert "key1" not in cache

    def test_remove(self) -> None:
        """Test explicit entry removal."""
        cache: ResponseCache[str] = ResponseCache()

        cache.set("key1", "value1")
        assert cache.remove("key1") is True
        assert cache.get("key1") is None
        assert cache.remove("key1") is False  # Already removed

    def test_clear(self) -> None:
        """Test clearing all entries."""
        cache: ResponseCache[str] = ResponseCache()

        cache.set("a", "1")
        cache.set("b", "2")
        cache.clear()

        assert cache.size == 0
        assert cache.get("a") is None

    def test_prune_expired(self) -> None:
        """Test pruning expired entries."""
        cache: ResponseCache[str] = ResponseCache(ttl_seconds=0.1)

        cache.set("a", "1")
        cache.set("b", "2")
        time.sleep(0.15)
        cache.set("c", "3")  # Not expired

        removed = cache.prune_expired()
        assert removed == 2
        assert cache.size == 1
        assert cache.get("c") == "3"

    def test_stats_tracking(self) -> None:
        """Test statistics tracking."""
        cache: ResponseCache[str] = ResponseCache()

        cache.set("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key1")  # Hit
        cache.get("missing")  # Miss

        stats = cache.get_stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == pytest.approx(2 / 3)

    def test_with_llm_response(self) -> None:
        """Test cache works with LLMResponse objects."""
        cache: ResponseCache[LLMResponse] = ResponseCache()

        response = LLMResponse(
            content="Hello, world!",
            model="test-model",
            usage={"total_tokens": 10},
            latency_ms=100.0,
            provider="mock",
        )

        cache.set("test", response)
        retrieved = cache.get("test")

        assert retrieved is not None
        assert retrieved.content == "Hello, world!"
        assert retrieved.model == "test-model"
