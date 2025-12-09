"""
Tests for emulator webcam and video streaming.
"""

import asyncio

import numpy as np
import pytest

from emulator.video.webcam import MockWebcamCamera, WebcamCamera
from emulator.video.streamer import EmulatorVideoStreamer
from shared.messages import MessageType, RobotMessage


class TestMockWebcamCamera:
    """Tests for MockWebcamCamera."""

    @pytest.mark.asyncio
    async def test_mock_webcam_initialization(self):
        """Test mock webcam always initializes successfully."""
        camera = MockWebcamCamera()
        result = await camera.initialize()
        assert result is True
        assert camera.is_initialized is True
        await camera.shutdown()

    @pytest.mark.asyncio
    async def test_mock_webcam_start_stop(self):
        """Test mock webcam start and stop."""
        camera = MockWebcamCamera()
        await camera.initialize()

        assert camera.is_running is False
        await camera.start()
        assert camera.is_running is True
        await camera.stop()
        assert camera.is_running is False

        await camera.shutdown()

    @pytest.mark.asyncio
    async def test_mock_webcam_captures_frames(self):
        """Test mock webcam generates frames."""
        camera = MockWebcamCamera()
        await camera.initialize()
        await camera.start()

        frame = camera.capture_frame()
        assert frame is not None
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (480, 640, 3)
        assert frame.dtype == np.uint8

        await camera.shutdown()

    @pytest.mark.asyncio
    async def test_mock_webcam_frame_changes(self):
        """Test mock webcam generates different frames over time."""
        camera = MockWebcamCamera()
        await camera.initialize()
        await camera.start()

        frame1 = camera.capture_frame()
        frame2 = camera.capture_frame()

        # Frames should be different (moving rectangle)
        assert not np.array_equal(frame1, frame2)

        await camera.shutdown()

    @pytest.mark.asyncio
    async def test_mock_webcam_no_frame_when_not_running(self):
        """Test mock webcam returns None when not running."""
        camera = MockWebcamCamera()
        await camera.initialize()
        # Don't start

        frame = camera.capture_frame()
        assert frame is None

        await camera.shutdown()

    @pytest.mark.asyncio
    async def test_mock_webcam_status(self):
        """Test mock webcam status reporting."""
        camera = MockWebcamCamera()
        await camera.initialize()

        status = camera.get_status()
        assert status["initialized"] is True
        assert status["running"] is False
        assert status["mock"] is True
        assert status["width"] == 640
        assert status["height"] == 480
        assert status["fps"] == 10

        await camera.shutdown()

    @pytest.mark.asyncio
    async def test_mock_webcam_video_track_creation(self):
        """Test mock webcam can create video track."""
        camera = MockWebcamCamera()
        await camera.initialize()
        await camera.start()

        track = camera.create_video_track()
        assert track is not None
        assert track.kind == "video"

        await camera.shutdown()


class TestWebcamCamera:
    """Tests for WebcamCamera (real webcam)."""

    @pytest.mark.asyncio
    async def test_webcam_initialization_without_hardware(self):
        """Test webcam initialization returns False when no hardware."""
        # This test may pass or fail depending on hardware
        camera = WebcamCamera(device_id=99)  # Unlikely to exist
        result = await camera.initialize()
        # Should return False for non-existent device
        assert result is False
        assert camera.is_initialized is False

    @pytest.mark.asyncio
    async def test_webcam_status_when_not_initialized(self):
        """Test webcam status when not initialized."""
        camera = WebcamCamera()
        status = camera.get_status()
        assert status["initialized"] is False
        assert status["running"] is False
        assert status["device_id"] == 0

    @pytest.mark.asyncio
    async def test_webcam_cannot_start_when_not_initialized(self):
        """Test webcam cannot start when not initialized."""
        camera = WebcamCamera()
        await camera.start()  # Should not crash
        assert camera.is_running is False

    @pytest.mark.asyncio
    async def test_webcam_no_frame_when_not_initialized(self):
        """Test webcam returns None when not initialized."""
        camera = WebcamCamera()
        frame = camera.capture_frame()
        assert frame is None


class TestEmulatorVideoStreamer:
    """Tests for EmulatorVideoStreamer."""

    @pytest.mark.asyncio
    async def test_streamer_creation(self):
        """Test streamer can be created."""
        camera = MockWebcamCamera()
        await camera.initialize()

        signaling_messages = []

        async def on_signaling(msg: RobotMessage):
            signaling_messages.append(msg)

        streamer = EmulatorVideoStreamer(
            camera=camera,
            on_signaling=on_signaling,
        )

        assert streamer.connection_state == "new"
        assert streamer.is_connected is False

        await camera.shutdown()

    @pytest.mark.asyncio
    async def test_streamer_stats(self):
        """Test streamer provides stats."""
        camera = MockWebcamCamera()
        await camera.initialize()

        streamer = EmulatorVideoStreamer(camera=camera)
        stats = streamer.get_stats()

        assert "running" in stats
        assert "connected" in stats
        assert "connection_state" in stats
        assert "camera_running" in stats

        assert stats["running"] is False
        assert stats["connected"] is False

        await camera.shutdown()

    @pytest.mark.asyncio
    async def test_streamer_repr(self):
        """Test streamer string representation."""
        camera = MockWebcamCamera()
        await camera.initialize()

        streamer = EmulatorVideoStreamer(camera=camera)
        repr_str = repr(streamer)

        assert "EmulatorVideoStreamer" in repr_str
        assert "connected=False" in repr_str

        await camera.shutdown()

    @pytest.mark.asyncio
    async def test_streamer_sends_offer_on_start(self):
        """Test streamer sends WebRTC offer on start."""
        # Skip if aiortc not installed
        try:
            import aiortc  # noqa: F401
        except ImportError:
            pytest.skip("aiortc not installed")

        camera = MockWebcamCamera()
        await camera.initialize()

        signaling_messages = []

        async def on_signaling(msg: RobotMessage):
            signaling_messages.append(msg)

        streamer = EmulatorVideoStreamer(
            camera=camera,
            on_signaling=on_signaling,
        )

        # Start the streamer
        await streamer.start()

        # Wait a bit for offer to be created
        await asyncio.sleep(0.5)

        # Should have sent an offer
        assert len(signaling_messages) >= 1
        assert signaling_messages[0].message_type == MessageType.WEBRTC_OFFER

        await streamer.stop()
        await camera.shutdown()

    @pytest.mark.asyncio
    async def test_streamer_stop(self):
        """Test streamer can be stopped."""
        camera = MockWebcamCamera()
        await camera.initialize()

        streamer = EmulatorVideoStreamer(camera=camera)
        await streamer.start()
        await asyncio.sleep(0.2)

        await streamer.stop()

        stats = streamer.get_stats()
        assert stats["running"] is False

        await camera.shutdown()

    @pytest.mark.asyncio
    async def test_streamer_handle_answer_without_pc(self):
        """Test handle_answer does nothing without peer connection."""
        camera = MockWebcamCamera()
        await camera.initialize()

        streamer = EmulatorVideoStreamer(camera=camera)
        # Don't start - no peer connection

        # Should not crash
        await streamer.handle_answer("v=0\r\no=- 123 456 IN IP4 0.0.0.0\r\ns=-\r\n")

        await camera.shutdown()

    @pytest.mark.asyncio
    async def test_streamer_add_ice_candidate_without_pc(self):
        """Test add_ice_candidate does nothing without peer connection."""
        camera = MockWebcamCamera()
        await camera.initialize()

        streamer = EmulatorVideoStreamer(camera=camera)
        # Don't start - no peer connection

        # Should not crash
        await streamer.add_ice_candidate({
            "candidate": "candidate:1 1 UDP 2130706431 192.168.1.1 12345 typ host",
            "sdp_mid": "0",
            "sdp_mline_index": 0,
        })

        await camera.shutdown()


class TestVideoTrack:
    """Tests for video track functionality."""

    @pytest.mark.asyncio
    async def test_mock_video_track_recv(self):
        """Test mock video track can receive frames."""
        camera = MockWebcamCamera()
        await camera.initialize()
        await camera.start()

        track = camera.create_video_track()

        # Receive a frame (this uses aiortc/av)
        try:
            frame = await asyncio.wait_for(track.recv(), timeout=2.0)
            assert frame is not None
            assert frame.format.name == "rgb24"
        except ImportError:
            pytest.skip("aiortc/av not installed")

        await camera.shutdown()
