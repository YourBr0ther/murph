"""
Murph - Frame Buffer
Thread-safe single-frame buffer with latest-frame semantics.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from shared.constants import VISION_FRAME_STALE_MS

logger = logging.getLogger(__name__)


@dataclass
class FrameStats:
    """Statistics about frame buffer usage."""

    frames_received: int = 0
    frames_dropped: int = 0
    last_frame_time: float = 0.0
    avg_frame_interval_ms: float = 0.0


class FrameBuffer:
    """
    Thread-safe single-frame buffer with latest-frame semantics.

    This buffer stores only the most recent video frame, allowing
    the WebRTC callback to write frames and the perception loop
    to read the latest frame without blocking.

    Features:
    - Thread-safe: WebRTC callbacks may run on different threads
    - Latest-frame semantics: Always returns newest frame, drops old ones
    - Staleness detection: Tracks frame age to detect stale data
    - Async event: Signals when new frames arrive

    Usage:
        buffer = FrameBuffer()

        # WebRTC callback writes frames
        buffer.put(frame)

        # Perception loop reads latest
        frame, timestamp = buffer.get_latest()
        if frame is not None and not buffer.is_stale():
            process(frame)
    """

    def __init__(self, stale_threshold_ms: float = VISION_FRAME_STALE_MS) -> None:
        """
        Initialize frame buffer.

        Args:
            stale_threshold_ms: Consider frame stale after this many milliseconds
        """
        self._frame: np.ndarray | None = None
        self._timestamp: float = 0.0
        self._lock = threading.Lock()
        self._stale_threshold = stale_threshold_ms / 1000.0

        # Stats tracking
        self._stats = FrameStats()
        self._frame_intervals: list[float] = []

        # Async event for waiting on new frames
        self._new_frame_event: asyncio.Event | None = None
        self._event_loop: asyncio.AbstractEventLoop | None = None

        logger.debug(f"FrameBuffer initialized (stale_threshold={stale_threshold_ms}ms)")

    def put(self, frame: np.ndarray) -> None:
        """
        Store a new frame (called from WebRTC track callback).

        This method is thread-safe and can be called from any thread.
        If a frame already exists, it will be replaced (dropped).

        Args:
            frame: RGB numpy array with shape (H, W, 3)
        """
        now = time.time()

        with self._lock:
            # Track dropped frames
            if self._frame is not None:
                self._stats.frames_dropped += 1

            # Store new frame (copy to avoid reference issues)
            self._frame = frame.copy()

            # Update timing
            if self._timestamp > 0:
                interval = now - self._timestamp
                self._frame_intervals.append(interval)
                # Keep only recent intervals for averaging
                if len(self._frame_intervals) > 100:
                    self._frame_intervals.pop(0)
                self._stats.avg_frame_interval_ms = (
                    sum(self._frame_intervals) / len(self._frame_intervals) * 1000
                )

            self._timestamp = now
            self._stats.frames_received += 1
            self._stats.last_frame_time = now

        # Signal async consumers (thread-safe event set)
        self._signal_new_frame()

    def get_latest(self) -> tuple[np.ndarray | None, float]:
        """
        Get the latest frame without blocking.

        Returns:
            Tuple of (frame, timestamp) where frame may be None if no frame available
        """
        with self._lock:
            if self._frame is None:
                return None, 0.0
            # Return copy to prevent modification
            return self._frame.copy(), self._timestamp

    def peek(self) -> tuple[np.ndarray | None, float]:
        """
        Peek at the latest frame without copying.

        WARNING: The returned frame should not be modified.
        Use get_latest() if you need a safe copy.

        Returns:
            Tuple of (frame, timestamp) - frame may be None
        """
        with self._lock:
            return self._frame, self._timestamp

    def is_stale(self, max_age_seconds: float | None = None) -> bool:
        """
        Check if the current frame is stale.

        Args:
            max_age_seconds: Override default stale threshold

        Returns:
            True if frame is stale or no frame available
        """
        with self._lock:
            return self._is_stale_unlocked(max_age_seconds)

    def _is_stale_unlocked(self, max_age_seconds: float | None = None) -> bool:
        """Check staleness without acquiring lock (for use when lock already held)."""
        threshold = max_age_seconds if max_age_seconds is not None else self._stale_threshold

        if self._frame is None or self._timestamp == 0:
            return True
        return (time.time() - self._timestamp) > threshold

    def clear(self) -> None:
        """Clear the buffer, removing any stored frame."""
        with self._lock:
            self._frame = None
            self._timestamp = 0.0

    async def wait_for_frame(self, timeout: float = 1.0) -> np.ndarray | None:
        """
        Wait for a new frame asynchronously.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            The new frame, or None if timeout
        """
        # Lazy initialize event for current event loop
        if self._new_frame_event is None or self._event_loop != asyncio.get_event_loop():
            self._event_loop = asyncio.get_event_loop()
            self._new_frame_event = asyncio.Event()

        try:
            await asyncio.wait_for(self._new_frame_event.wait(), timeout)
            self._new_frame_event.clear()
            frame, _ = self.get_latest()
            return frame
        except asyncio.TimeoutError:
            return None

    def _signal_new_frame(self) -> None:
        """Signal that a new frame is available (thread-safe)."""
        if self._new_frame_event is not None and self._event_loop is not None:
            try:
                # Thread-safe way to set event from another thread
                self._event_loop.call_soon_threadsafe(self._new_frame_event.set)
            except RuntimeError:
                # Event loop not running or closed
                pass

    def get_stats(self) -> dict[str, Any]:
        """Get buffer statistics."""
        with self._lock:
            age_ms = (time.time() - self._timestamp) * 1000 if self._timestamp > 0 else -1
            return {
                "frames_received": self._stats.frames_received,
                "frames_dropped": self._stats.frames_dropped,
                "has_frame": self._frame is not None,
                "frame_age_ms": age_ms,
                "is_stale": self._is_stale_unlocked(),
                "avg_frame_interval_ms": self._stats.avg_frame_interval_ms,
            }

    @property
    def has_frame(self) -> bool:
        """Check if a frame is available."""
        with self._lock:
            return self._frame is not None

    @property
    def frame_age(self) -> float:
        """Get age of current frame in seconds, or -1 if no frame."""
        with self._lock:
            if self._timestamp == 0:
                return -1.0
            return time.time() - self._timestamp

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"FrameBuffer(has_frame={stats['has_frame']}, "
            f"age_ms={stats['frame_age_ms']:.1f}, "
            f"received={stats['frames_received']})"
        )
