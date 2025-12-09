"""
Murph - LLM Response Cache
TTL-based LRU cache for LLM responses.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

T = TypeVar("T")


@dataclass
class CacheEntry(Generic[T]):
    """A cached item with timestamp."""

    value: T
    timestamp: float = field(default_factory=time.time)

    def age_seconds(self) -> float:
        """Get age of this entry in seconds."""
        return time.time() - self.timestamp


class ResponseCache(Generic[T]):
    """
    TTL-based LRU cache for LLM responses.

    Features:
    - Time-to-live (TTL) expiration
    - Least Recently Used (LRU) eviction
    - Configurable max entries
    - Statistics tracking
    """

    def __init__(
        self,
        ttl_seconds: float = 30.0,
        max_entries: int = 100,
    ) -> None:
        """
        Initialize the cache.

        Args:
            ttl_seconds: How long entries remain valid
            max_entries: Maximum number of entries before LRU eviction
        """
        self._ttl = ttl_seconds
        self._max_entries = max_entries
        self._cache: OrderedDict[str, CacheEntry[T]] = OrderedDict()

        # Stats
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: str) -> T | None:
        """
        Get cached value if not expired.

        Args:
            key: Cache key

        Returns:
            Cached value, or None if not found or expired
        """
        if key not in self._cache:
            self._misses += 1
            return None

        entry = self._cache[key]

        # Check TTL
        if entry.age_seconds() > self._ttl:
            del self._cache[key]
            self._misses += 1
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)
        self._hits += 1
        return entry.value

    def set(self, key: str, value: T) -> None:
        """
        Cache a value.

        Args:
            key: Cache key
            value: Value to cache
        """
        # Remove if exists (to update timestamp)
        if key in self._cache:
            del self._cache[key]

        # Add new entry
        self._cache[key] = CacheEntry(value=value)
        self._cache.move_to_end(key)

        # Evict oldest if over limit
        while len(self._cache) > self._max_entries:
            self._cache.popitem(last=False)
            self._evictions += 1

    def remove(self, key: str) -> bool:
        """
        Remove an entry from cache.

        Args:
            key: Cache key

        Returns:
            True if entry was removed, False if not found
        """
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()

    def prune_expired(self) -> int:
        """
        Remove all expired entries.

        Returns:
            Number of entries removed
        """
        now = time.time()
        expired_keys = [
            key
            for key, entry in self._cache.items()
            if now - entry.timestamp > self._ttl
        ]

        for key in expired_keys:
            del self._cache[key]

        return len(expired_keys)

    @property
    def size(self) -> int:
        """Current number of entries."""
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        """Cache hit rate (0.0 to 1.0)."""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "entries": len(self._cache),
            "max_entries": self._max_entries,
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "hit_rate": self.hit_rate,
            "ttl_seconds": self._ttl,
        }

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        if key not in self._cache:
            return False
        entry = self._cache[key]
        return entry.age_seconds() <= self._ttl

    def __repr__(self) -> str:
        return (
            f"ResponseCache(entries={len(self._cache)}, "
            f"ttl={self._ttl}s, hit_rate={self.hit_rate:.1%})"
        )
