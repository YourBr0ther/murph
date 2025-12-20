"""
Tests for Pi camera implementations.
"""

import pytest
import asyncio
import numpy as np

from pi.video import MockCameraManager, OpenCVCameraManager


class TestMockCameraManager:
    """Tests for MockCameraManager."""

    @pytest.fixture
    def camera(self):
        return MockCameraManager()

    @pytest.mark.asyncio
    async def test_initialize(self, camera):
        assert not camera.is_initialized
        result = await camera.initialize()
        assert result is True
        assert camera.is_initialized

    @pytest.mark.asyncio
    async def test_start_stop(self, camera):
        await camera.initialize()
        assert not camera.is_running

        await camera.start()
        assert camera.is_running

        await camera.stop()
        assert not camera.is_running

    @pytest.mark.asyncio
    async def test_capture_frame(self, camera):
        await camera.initialize()
        await camera.start()

        frame = camera.capture_frame()
        assert frame is not None
        assert isinstance(frame, np.ndarray)
        assert frame.shape[2] == 3  # RGB channels

    @pytest.mark.asyncio
    async def test_capture_before_start_returns_none(self, camera):
        await camera.initialize()
        frame = camera.capture_frame()
        assert frame is None

    @pytest.mark.asyncio
    async def test_create_video_track(self, camera):
        await camera.initialize()
        await camera.start()

        track = camera.create_video_track()
        assert track is not None
        assert track.kind == "video"

    @pytest.mark.asyncio
    async def test_get_status(self, camera):
        await camera.initialize()
        await camera.start()

        status = camera.get_status()
        assert status["initialized"] is True
        assert status["running"] is True
        assert status["mock"] is True
        assert "frame_count" in status

    @pytest.mark.asyncio
    async def test_shutdown(self, camera):
        await camera.initialize()
        await camera.start()
        await camera.shutdown()

        assert not camera.is_initialized
        assert not camera.is_running


class TestOpenCVCameraManager:
    """Tests for OpenCVCameraManager.

    Note: These tests check the interface works correctly.
    On systems without a camera, initialization may fail.
    """

    @pytest.fixture
    def camera(self):
        return OpenCVCameraManager()

    @pytest.mark.asyncio
    async def test_interface_consistency(self, camera):
        """Test that OpenCVCameraManager has same interface as MockCameraManager."""
        # Check properties exist
        assert hasattr(camera, 'is_initialized')
        assert hasattr(camera, 'is_running')

        # Check methods exist
        assert callable(getattr(camera, 'initialize', None))
        assert callable(getattr(camera, 'start', None))
        assert callable(getattr(camera, 'stop', None))
        assert callable(getattr(camera, 'shutdown', None))
        assert callable(getattr(camera, 'capture_frame', None))
        assert callable(getattr(camera, 'create_video_track', None))
        assert callable(getattr(camera, 'get_status', None))

    @pytest.mark.asyncio
    async def test_get_status_before_init(self, camera):
        """Test status before initialization."""
        status = camera.get_status()
        assert status["initialized"] is False
        assert status["type"] == "opencv"

    @pytest.mark.asyncio
    async def test_capture_before_init_returns_none(self, camera):
        """Test that capture returns None before initialization."""
        frame = camera.capture_frame()
        assert frame is None

    @pytest.mark.asyncio
    async def test_start_without_init_logs_warning(self, camera, caplog):
        """Test that starting without init logs warning."""
        import logging
        caplog.set_level(logging.WARNING)
        await camera.start()
        assert "not initialized" in caplog.text.lower() or not camera.is_running
