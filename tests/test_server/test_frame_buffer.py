"""Tests for the FrameBuffer class."""

import asyncio
import threading
import time

import numpy as np
import pytest

from server.video.frame_buffer import FrameBuffer


class TestFrameBuffer:
    """Tests for FrameBuffer."""

    def test_init(self) -> None:
        """Test buffer initialization."""
        buffer = FrameBuffer()
        assert buffer.has_frame is False
        assert buffer.frame_age == -1.0
        assert buffer.is_stale() is True

    def test_put_and_get(self) -> None:
        """Test storing and retrieving frames."""
        buffer = FrameBuffer()
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        buffer.put(frame)

        assert buffer.has_frame is True
        retrieved, timestamp = buffer.get_latest()
        assert retrieved is not None
        assert np.array_equal(retrieved, frame)
        assert timestamp > 0

    def test_latest_frame_semantics(self) -> None:
        """Test that only the latest frame is kept."""
        buffer = FrameBuffer()

        frame1 = np.ones((480, 640, 3), dtype=np.uint8) * 100
        frame2 = np.ones((480, 640, 3), dtype=np.uint8) * 200

        buffer.put(frame1)
        buffer.put(frame2)

        retrieved, _ = buffer.get_latest()
        assert np.array_equal(retrieved, frame2)

        stats = buffer.get_stats()
        assert stats["frames_received"] == 2
        assert stats["frames_dropped"] == 1

    def test_frame_copy_on_put(self) -> None:
        """Test that frames are copied on put."""
        buffer = FrameBuffer()
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 100

        buffer.put(frame)
        frame[:] = 200  # Modify original

        retrieved, _ = buffer.get_latest()
        assert np.all(retrieved == 100)  # Should be unchanged

    def test_frame_copy_on_get(self) -> None:
        """Test that frames are copied on get."""
        buffer = FrameBuffer()
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 100

        buffer.put(frame)
        retrieved1, _ = buffer.get_latest()
        retrieved1[:] = 200  # Modify retrieved

        retrieved2, _ = buffer.get_latest()
        assert np.all(retrieved2 == 100)  # Should be unchanged

    def test_peek_no_copy(self) -> None:
        """Test that peek returns reference without copying."""
        buffer = FrameBuffer()
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 100

        buffer.put(frame)
        peeked1, _ = buffer.peek()
        peeked2, _ = buffer.peek()

        # Should be the same object (no copy)
        assert peeked1 is peeked2

    def test_staleness_detection(self) -> None:
        """Test frame staleness detection."""
        buffer = FrameBuffer(stale_threshold_ms=100)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        buffer.put(frame)
        assert buffer.is_stale() is False

        # Wait for frame to become stale
        time.sleep(0.15)
        assert buffer.is_stale() is True

    def test_custom_stale_threshold(self) -> None:
        """Test staleness with custom threshold."""
        buffer = FrameBuffer(stale_threshold_ms=500)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        buffer.put(frame)

        # Check with default threshold - should not be stale
        assert buffer.is_stale() is False

        # Wait a bit to ensure some time passes
        time.sleep(0.005)  # 5ms

        # Check with custom max_age - should be stale after 5ms with 1ms threshold
        assert buffer.is_stale(max_age_seconds=0.001) is True

    def test_clear(self) -> None:
        """Test clearing the buffer."""
        buffer = FrameBuffer()
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        buffer.put(frame)
        assert buffer.has_frame is True

        buffer.clear()
        assert buffer.has_frame is False
        retrieved, _ = buffer.get_latest()
        assert retrieved is None

    def test_thread_safety(self) -> None:
        """Test thread-safe operation."""
        buffer = FrameBuffer()
        frames_written = []
        frames_read = []
        num_frames = 100

        def writer():
            for i in range(num_frames):
                frame = np.ones((10, 10, 3), dtype=np.uint8) * i
                buffer.put(frame)
                frames_written.append(i)
                time.sleep(0.001)

        def reader():
            for _ in range(num_frames):
                frame, _ = buffer.get_latest()
                if frame is not None:
                    frames_read.append(frame[0, 0, 0])
                time.sleep(0.001)

        writer_thread = threading.Thread(target=writer)
        reader_thread = threading.Thread(target=reader)

        writer_thread.start()
        reader_thread.start()

        writer_thread.join()
        reader_thread.join()

        # All writes should complete
        assert len(frames_written) == num_frames
        # Some frames should be read (not necessarily all due to latest-frame semantics)
        assert len(frames_read) > 0

    def test_stats(self) -> None:
        """Test statistics tracking."""
        buffer = FrameBuffer()

        # Initial stats
        stats = buffer.get_stats()
        assert stats["frames_received"] == 0
        assert stats["frames_dropped"] == 0
        assert stats["has_frame"] is False

        # After adding frames
        for i in range(5):
            frame = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
            buffer.put(frame)
            time.sleep(0.01)

        stats = buffer.get_stats()
        assert stats["frames_received"] == 5
        assert stats["frames_dropped"] == 4
        assert stats["has_frame"] is True
        assert stats["avg_frame_interval_ms"] > 0

    @pytest.mark.asyncio
    async def test_wait_for_frame(self) -> None:
        """Test async waiting for frames."""
        buffer = FrameBuffer()

        async def writer():
            await asyncio.sleep(0.05)
            frame = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
            buffer.put(frame)

        # Start writer task
        asyncio.create_task(writer())

        # Wait for frame
        frame = await buffer.wait_for_frame(timeout=1.0)
        assert frame is not None

    @pytest.mark.asyncio
    async def test_wait_for_frame_timeout(self) -> None:
        """Test timeout when no frame arrives."""
        buffer = FrameBuffer()

        # Wait with short timeout
        frame = await buffer.wait_for_frame(timeout=0.1)
        assert frame is None

    def test_repr(self) -> None:
        """Test string representation."""
        buffer = FrameBuffer()
        repr_str = repr(buffer)
        assert "FrameBuffer" in repr_str
        assert "has_frame=False" in repr_str
