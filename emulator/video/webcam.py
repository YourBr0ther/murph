"""
Murph - Emulator Webcam Camera
USB webcam capture using OpenCV for video streaming.
"""

from __future__ import annotations

import asyncio
import logging
import time
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Iterator

import numpy as np

from shared.constants import VIDEO_FPS, VIDEO_HEIGHT, VIDEO_WIDTH

if TYPE_CHECKING:
    from aiortc.mediastreams import MediaStreamTrack
    from av import VideoFrame

logger = logging.getLogger(__name__)


class WebcamCamera:
    """
    USB webcam capture using OpenCV.

    Provides frames compatible with aiortc VideoStreamTrack for
    WebRTC streaming to the server. Mirrors CameraManager API from pi/video/.

    Usage:
        camera = WebcamCamera()
        if await camera.initialize():
            track = camera.create_video_track()
            # Add track to WebRTC peer connection
    """

    def __init__(
        self,
        device_id: int = 0,
        width: int = VIDEO_WIDTH,
        height: int = VIDEO_HEIGHT,
        fps: int = VIDEO_FPS,
    ) -> None:
        """
        Initialize webcam camera.

        Args:
            device_id: OpenCV camera device ID (0 = default webcam)
            width: Frame width
            height: Frame height
            fps: Target frames per second
        """
        self._device_id = device_id
        self._width = width
        self._height = height
        self._fps = fps
        self._capture = None
        self._initialized = False
        self._running = False

        logger.info(f"WebcamCamera created (device={device_id}, {width}x{height} @ {fps}fps)")

    @classmethod
    def find_available_device(cls, max_devices: int = 5) -> int | None:
        """
        Scan for available camera devices.

        Args:
            max_devices: Maximum number of device indices to try

        Returns:
            First available device index, or None if no cameras found
        """
        try:
            import cv2
        except ImportError:
            logger.warning("OpenCV (cv2) not installed - cannot scan for cameras")
            return None

        for device_id in range(max_devices):
            cap = cv2.VideoCapture(device_id)
            if cap.isOpened():
                cap.release()
                logger.info(f"Found camera at device index {device_id}")
                return device_id
            cap.release()

        logger.warning("No camera devices found")
        return None

    async def initialize(self) -> bool:
        """
        Initialize the webcam.

        Returns:
            True if webcam opened successfully
        """
        try:
            import cv2

            self._capture = cv2.VideoCapture(self._device_id)

            if not self._capture.isOpened():
                logger.warning(f"Could not open webcam device {self._device_id}")
                self._capture = None
                return False

            # Set resolution and FPS
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            self._capture.set(cv2.CAP_PROP_FPS, self._fps)

            # Read actual settings (may differ from requested)
            actual_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self._capture.get(cv2.CAP_PROP_FPS)

            logger.info(
                f"Webcam initialized: {actual_width}x{actual_height} @ {actual_fps:.1f}fps"
            )

            self._initialized = True
            return True

        except ImportError:
            logger.warning("OpenCV (cv2) not installed - webcam disabled")
            return False
        except Exception as e:
            logger.error(f"Webcam initialization failed: {e}")
            return False

    async def start(self) -> None:
        """Start the webcam capture."""
        if not self._initialized:
            logger.warning("Cannot start webcam - not initialized")
            return

        self._running = True
        logger.info("Webcam started")

    async def stop(self) -> None:
        """Stop the webcam capture."""
        self._running = False
        logger.info("Webcam stopped")

    async def shutdown(self) -> None:
        """Shutdown and release webcam resources."""
        await self.stop()
        if self._capture is not None:
            try:
                self._capture.release()
                self._capture = None
                self._initialized = False
                logger.info("Webcam shutdown complete")
            except Exception as e:
                logger.warning(f"Error shutting down webcam: {e}")

    def capture_frame(self) -> np.ndarray | None:
        """
        Capture a single frame from webcam.

        Returns:
            RGB numpy array (H, W, 3) or None if capture failed
        """
        if not self._initialized or not self._running:
            return None

        if self._capture is None:
            return None

        try:
            import cv2

            ret, frame = self._capture.read()
            if not ret:
                logger.warning("Failed to read frame from webcam")
                return None

            # OpenCV returns BGR, convert to RGB for WebRTC
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return rgb_frame

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
        return WebcamVideoTrack(self)

    @property
    def is_initialized(self) -> bool:
        """Check if webcam is initialized."""
        return self._initialized

    @property
    def is_running(self) -> bool:
        """Check if webcam is running."""
        return self._running

    def get_status(self) -> dict[str, Any]:
        """Get webcam status."""
        return {
            "initialized": self._initialized,
            "running": self._running,
            "device_id": self._device_id,
            "width": self._width,
            "height": self._height,
            "fps": self._fps,
        }


class WebcamVideoTrack:
    """
    aiortc-compatible VideoStreamTrack wrapper for WebcamCamera.

    This class implements the interface expected by aiortc for
    sending video frames over WebRTC.
    """

    kind = "video"

    def __init__(self, camera: WebcamCamera) -> None:
        """
        Initialize video track.

        Args:
            camera: WebcamCamera instance
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

        # Add frame pacing to match target FPS
        target_interval = 1.0 / self._camera._fps
        await asyncio.sleep(target_interval)

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

        # Set timestamp (pts) based on elapsed time
        elapsed = time.time() - self._start_time
        video_frame.pts = int(elapsed * 90000)  # 90kHz timebase
        video_frame.time_base = Fraction(1, 90000)

        return video_frame

    def stop(self) -> None:
        """Stop the video track."""
        pass  # Camera stop is handled by WebcamCamera


class MockWebcamCamera:
    """
    Mock webcam camera for testing without hardware.

    Generates synthetic test frames with a moving rectangle
    to simulate a face/object in the frame.
    """

    def __init__(
        self,
        width: int = VIDEO_WIDTH,
        height: int = VIDEO_HEIGHT,
        fps: int = VIDEO_FPS,
    ) -> None:
        """
        Initialize mock webcam.

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

        logger.info(f"MockWebcamCamera created ({width}x{height} @ {fps}fps)")

    async def initialize(self) -> bool:
        """Initialize mock webcam (always succeeds)."""
        self._initialized = True
        logger.info("MockWebcamCamera initialized")
        return True

    async def start(self) -> None:
        """Start mock webcam."""
        self._running = True
        logger.info("MockWebcamCamera started")

    async def stop(self) -> None:
        """Stop mock webcam."""
        self._running = False
        logger.info("MockWebcamCamera stopped")

    async def shutdown(self) -> None:
        """Shutdown mock webcam."""
        await self.stop()
        self._initialized = False
        logger.info("MockWebcamCamera shutdown")

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

        # Gradient background (blue-ish to distinguish from Pi mock)
        for y in range(self._height):
            frame[y, :, 2] = int(255 * y / self._height)  # Blue gradient

        # Moving rectangle (simulates face)
        rect_size = 100
        x = (self._frame_count * 3) % (self._width - rect_size)
        y = (self._frame_count * 2) % (self._height - rect_size)
        frame[y : y + rect_size, x : x + rect_size] = [200, 180, 150]  # Skin-ish color

        # Frame counter overlay (simple pattern)
        counter_width = min(10 + (self._frame_count % 100), self._width - 20)
        frame[10:20, 10 : 10 + counter_width] = [255, 255, 255]

        # Add "EMU" text indicator (top right corner)
        frame[10:20, self._width - 50 : self._width - 10] = [0, 255, 0]  # Green bar

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
        return MockWebcamVideoTrack(self)

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


class MockWebcamVideoTrack:
    """Mock video track for testing."""

    kind = "video"

    def __init__(self, camera: MockWebcamCamera) -> None:
        self._camera = camera
        self._start_time = time.time()

    async def recv(self) -> VideoFrame:
        """Receive mock frame."""
        from av import VideoFrame as AVVideoFrame

        # Add small delay to simulate real camera timing
        await asyncio.sleep(1.0 / self._camera._fps)

        frame = self._camera.capture_frame()
        if frame is None:
            frame = np.zeros(
                (self._camera._height, self._camera._width, 3),
                dtype=np.uint8,
            )

        video_frame = AVVideoFrame.from_ndarray(frame, format="rgb24")
        elapsed = time.time() - self._start_time
        video_frame.pts = int(elapsed * 90000)
        video_frame.time_base = Fraction(1, 90000)

        return video_frame

    def stop(self) -> None:
        pass
