"""
Tests for Pi microphone capture implementations.
"""

import pytest
import asyncio

from pi.audio import MockMicrophoneCapture, MicrophoneCapture


class TestMockMicrophoneCapture:
    """Tests for MockMicrophoneCapture."""

    @pytest.fixture
    def microphone(self):
        return MockMicrophoneCapture()

    @pytest.mark.asyncio
    async def test_is_available(self, microphone):
        """Mock microphone should always be available."""
        assert microphone.is_available is True

    @pytest.mark.asyncio
    async def test_start_stop(self, microphone):
        """Test starting and stopping audio capture."""
        result = await microphone.start()
        assert result is True
        assert microphone.get_state().is_running is True

        await microphone.stop()
        assert microphone.get_state().is_running is False

    @pytest.mark.asyncio
    async def test_get_audio_level(self, microphone):
        """Test getting audio level."""
        await microphone.start()
        level = microphone.get_audio_level()
        assert 0.0 <= level <= 1.0

    @pytest.mark.asyncio
    async def test_simulate_voice(self, microphone):
        """Test simulating voice activity."""
        await microphone.start()
        microphone.simulate_voice(duration=0.5)
        # Give update loop time to process
        await asyncio.sleep(0.1)
        assert microphone.is_voice_detected() is True

        # Wait for voice to end
        await asyncio.sleep(0.6)
        assert microphone.is_voice_detected() is False

    @pytest.mark.asyncio
    async def test_simulate_silence(self, microphone):
        """Test simulating silence."""
        await microphone.start()
        microphone.simulate_voice(duration=10.0)
        # Give update loop time to process
        await asyncio.sleep(0.1)
        assert microphone.is_voice_detected() is True

        microphone.simulate_silence()
        # Give update loop time to process
        await asyncio.sleep(0.1)
        assert microphone.is_voice_detected() is False

    @pytest.mark.asyncio
    async def test_get_audio_chunk(self, microphone):
        """Test getting audio chunk."""
        await microphone.start()
        chunk = microphone.get_audio_chunk()
        assert chunk is not None
        assert isinstance(chunk, bytes)
        # 20ms at 16kHz = 320 samples * 2 bytes = 640 bytes
        assert len(chunk) == 640

    @pytest.mark.asyncio
    async def test_create_audio_track(self, microphone):
        """Test creating audio track."""
        track = microphone.create_audio_track()
        assert track is not None
        assert track.kind == "audio"

    @pytest.mark.asyncio
    async def test_get_state(self, microphone):
        """Test getting state dataclass."""
        await microphone.start()
        state = microphone.get_state()

        assert hasattr(state, 'is_running')
        assert hasattr(state, 'audio_level')
        assert hasattr(state, 'is_voice_detected')
        assert hasattr(state, 'samples_captured')
        assert hasattr(state, 'last_update_ms')

        assert state.is_running is True


class TestMicrophoneCapture:
    """Tests for MicrophoneCapture.

    Note: Real microphone may not be available in CI/test environments.
    These tests verify the interface and fallback behavior.
    """

    @pytest.fixture
    def microphone(self):
        return MicrophoneCapture()

    def test_interface_consistency(self, microphone):
        """Test that MicrophoneCapture has same interface as MockMicrophoneCapture."""
        # Check properties exist
        assert hasattr(microphone, 'is_available')

        # Check methods exist
        assert callable(getattr(microphone, 'start', None))
        assert callable(getattr(microphone, 'stop', None))
        assert callable(getattr(microphone, 'get_audio_level', None))
        assert callable(getattr(microphone, 'is_voice_detected', None))
        assert callable(getattr(microphone, 'get_state', None))
        assert callable(getattr(microphone, 'get_audio_chunk', None))
        assert callable(getattr(microphone, 'create_audio_track', None))

    @pytest.mark.asyncio
    async def test_fallback_to_mock(self, microphone):
        """Test that MicrophoneCapture falls back to mock if hardware unavailable."""
        # Start should succeed (either real or mock)
        result = await microphone.start()
        assert result is True

        # Should be able to get state
        state = microphone.get_state()
        assert state.is_running is True

        await microphone.stop()

    @pytest.mark.asyncio
    async def test_create_audio_track_works(self, microphone):
        """Test that create_audio_track works regardless of hardware."""
        track = microphone.create_audio_track()
        assert track is not None
        assert track.kind == "audio"
