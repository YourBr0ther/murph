"""
Murph - Camera Manager
PiCamera2 wrapper with H.264 encoding for video streaming.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, Iterator

import numpy as np

from shared.constants import VIDEO_FPS, VIDEO_HEIGHT, VIDEO_WIDTH

if TYPE_CHECKING:
    from aiortc.mediastreams import MediaStreamTrack
    from av import VideoFrame

logger = logging.getLogger(__name__)


class CameraManager:
    """
    Manages PiCamera2 for video capture with H.264 encoding.

    Provides frames compatible with aiortc VideoStreamTrack for
    WebRTC streaming to the server.

    Usage:
        camera = CameraManager()
        if await camera.initialize():
            track = camera.create_video_track()
            # Add track to WebRTC peer connection
    """

    def __init__(
        self,
        width: int = VIDEO_WIDTH,
        height: int = VIDEO_HEIGHT,
        fps: int = VIDEO_FPS,
    ) -> None:
        """
        Initialize camera manager.

        Args:
            width: Frame width
            height: Frame height
            fps: Target frames per second
        """
        self._width = width
        self._height = height
        self._fps = fps
        self._camera = None
        self._initialized = False
        self._running = False

        logger.info(f"CameraManager created ({width}x{height} @ {fps}fps)")

    async def initialize(self) -> bool:
        """
        Initialize the camera hardware.

        Returns:
            True if camera initialized successfully
        """
        try:
            from picamera2 import Picamera2

            self._camera = Picamera2()

            # Configure for video streaming
            config = self._camera.create_video_configuration(
                main={"size": (self._width, self._height), "format": "RGB888"},
                controls={"FrameRate": self._fps},
            )
            self._camera.configure(config)

            self._initialized = True
            logger.info("Camera initialized successfully")
            return True

        except ImportError:
            logger.error("picamera2 not available - camera disabled")
            return False
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            return False

    async def start(self) -> None:
        """Start the camera capture."""
        if not self._initialized:
            logger.warning("Cannot start camera - not initialized")
            return

        try:
            self._camera.start()
            self._running = True
            logger.info("Camera started")
        except Exception as e:
            logger.error(f"Failed to start camera: {e}")

    async def stop(self) -> None:
        """Stop the camera capture."""
        self._running = False
        if self._camera:
            try:
                self._camera.stop()
                logger.info("Camera stopped")
            except Exception as e:
                logger.warning(f"Error stopping camera: {e}")

    async def shutdown(self) -> None:
        """Shutdown and release camera resources."""
        await self.stop()
        if self._camera:
            try:
                self._camera.close()
                self._camera = None
                self._initialized = False
                logger.info("Camera shutdown complete")
            except Exception as e:
                logger.warning(f"Error shutting down camera: {e}")

    def capture_frame(self) -> np.ndarray | None:
        """
        Capture a single frame.

        Returns:
            RGB numpy array or None if capture failed
        """
        if not self._initialized or not self._running:
            return None

        try:
            frame = self._camera.capture_array()
            return frame
        except Exception as e:
            logger.error(f"Frame capture error: {e}")
            return None

    def frame_generator(self) -> Iterator[np.ndarray]:
        """
        Generate frames continuously.

        Yields:
            RGB numpy arrays
        """
        interval = 1.0 / self._fps

        while self._running:
            start = time.time()

            frame = self.capture_frame()
            if frame is not None:
                yield frame

            elapsed = time.time() - start
            if elapsed < interval:
                time.sleep(interval - elapsed)

    def create_video_track(self) -> MediaStreamTrack:
        """
        Create an aiortc-compatible VideoStreamTrack.

        Returns:
            MediaStreamTrack that can be added to RTCPeerConnection
        """
        return CameraVideoTrack(self)

    @property
    def is_initialized(self) -> bool:
        """Check if camera is initialized."""
        return self._initialized

    @property
    def is_running(self) -> bool:
        """Check if camera is running."""
        return self._running

    def get_status(self) -> dict[str, Any]:
        """Get camera status."""
        return {
            "initialized": self._initialized,
            "running": self._running,
            "width": self._width,
            "height": self._height,
            "fps": self._fps,
        }


class CameraVideoTrack:
    """
    aiortc-compatible VideoStreamTrack wrapper for CameraManager.

    This class implements the interface expected by aiortc for
    sending video frames over WebRTC.
    """

    kind = "video"

    def __init__(self, camera: CameraManager) -> None:
        """
        Initialize video track.

        Args:
            camera: CameraManager instance
        """
        self._camera = camera
        self._timestamp = 0
        self._start_time = time.time()

    async def recv(self) -> VideoFrame:
        """
        Receive the next video frame.

        This is called by aiortc to get frames for transmission.

        Returns:
            VideoFrame for WebRTC transmission
        """
        from av import VideoFrame as AVVideoFrame

        # Get frame from camera
        frame = self._camera.capture_frame()

        if frame is None:
            # Generate blank frame if capture failed
            frame = np.zeros(
                (self._camera._height, self._camera._width, 3),
                dtype=np.uint8,
            )

        # Convert to av VideoFrame
        video_frame = AVVideoFrame.from_ndarray(frame, format="rgb24")

        # Set timestamp (pts) based on frame rate
        elapsed = time.time() - self._start_time
        video_frame.pts = int(elapsed * 90000)  # 90kHz timebase
        video_frame.time_base = "1/90000"

        return video_frame

    def stop(self) -> None:
        """Stop the video track."""
        pass  # Camera stop is handled by CameraManager


class MockCameraManager:
    """
    Mock camera manager for testing without hardware.

    Generates synthetic test frames.
    """

    def __init__(
        self,
        width: int = VIDEO_WIDTH,
        height: int = VIDEO_HEIGHT,
        fps: int = VIDEO_FPS,
    ) -> None:
        """
        Initialize mock camera.

        Args:
            width: Frame width
            height: Frame height
            fps: Target frames per second
        """
        self._width = width
        self._height = height
        self._fps = fps
        self._initialized = False
        self._running = False
        self._frame_count = 0

        logger.info(f"MockCameraManager created ({width}x{height} @ {fps}fps)")

    async def initialize(self) -> bool:
        """Initialize mock camera (always succeeds)."""
        self._initialized = True
        logger.info("MockCameraManager initialized")
        return True

    async def start(self) -> None:
        """Start mock camera."""
        self._running = True
        logger.info("MockCameraManager started")

    async def stop(self) -> None:
        """Stop mock camera."""
        self._running = False
        logger.info("MockCameraManager stopped")

    async def shutdown(self) -> None:
        """Shutdown mock camera."""
        await self.stop()
        self._initialized = False
        logger.info("MockCameraManager shutdown")

    def capture_frame(self) -> np.ndarray | None:
        """
        Generate a synthetic test frame.

        Returns:
            RGB numpy array with test pattern
        """
        if not self._initialized or not self._running:
            return None

        # Generate test pattern
        frame = np.zeros((self._height, self._width, 3), dtype=np.uint8)

        # Gradient background
        for y in range(self._height):
            frame[y, :, 0] = int(255 * y / self._height)  # Red gradient

        # Moving rectangle (simulates face)
        rect_size = 100
        x = (self._frame_count * 3) % (self._width - rect_size)
        y = (self._frame_count * 2) % (self._height - rect_size)
        frame[y : y + rect_size, x : x + rect_size] = [200, 180, 150]  # Skin-ish color

        # Frame counter overlay (simple pattern)
        frame[10:20, 10 : 10 + (self._frame_count % 100)] = [255, 255, 255]

        self._frame_count += 1
        return frame

    def frame_generator(self) -> Iterator[np.ndarray]:
        """Generate frames continuously."""
        interval = 1.0 / self._fps

        while self._running:
            start = time.time()

            frame = self.capture_frame()
            if frame is not None:
                yield frame

            elapsed = time.time() - start
            if elapsed < interval:
                time.sleep(interval - elapsed)

    def create_video_track(self) -> MediaStreamTrack:
        """Create mock video track."""
        return MockVideoTrack(self)

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def is_running(self) -> bool:
        return self._running

    def get_status(self) -> dict[str, Any]:
        return {
            "initialized": self._initialized,
            "running": self._running,
            "width": self._width,
            "height": self._height,
            "fps": self._fps,
            "mock": True,
            "frame_count": self._frame_count,
        }


class MockVideoTrack:
    """Mock video track for testing."""

    kind = "video"

    def __init__(self, camera: MockCameraManager) -> None:
        self._camera = camera
        self._start_time = time.time()

    async def recv(self) -> VideoFrame:
        """Receive mock frame."""
        from av import VideoFrame as AVVideoFrame

        frame = self._camera.capture_frame()
        if frame is None:
            frame = np.zeros(
                (self._camera._height, self._camera._width, 3),
                dtype=np.uint8,
            )

        video_frame = AVVideoFrame.from_ndarray(frame, format="rgb24")
        elapsed = time.time() - self._start_time
        video_frame.pts = int(elapsed * 90000)
        video_frame.time_base = "1/90000"

        # Add small delay to simulate real camera timing
        await asyncio.sleep(1.0 / self._camera._fps)

        return video_frame

    def stop(self) -> None:
        pass
