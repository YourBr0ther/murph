"""Tests for emulator audio track (WebRTC compatible)."""

import asyncio

import pytest

from emulator.audio.microphone import MockMicrophoneCapture
from emulator.audio.track import MicrophoneAudioTrack


class TestMicrophoneAudioTrack:
    """Tests for MicrophoneAudioTrack class."""

    def test_track_kind_is_audio(self) -> None:
        """Test track has kind='audio' for WebRTC compatibility."""
        mic = MockMicrophoneCapture()
        track = MicrophoneAudioTrack(mic)
        assert track.kind == "audio"

    def test_track_ready_state(self) -> None:
        """Test track has readyState for aiortc compatibility."""
        mic = MockMicrophoneCapture()
        track = MicrophoneAudioTrack(mic)
        assert track.readyState == "live"

    def test_track_initial_state(self) -> None:
        """Test track initializes with correct state."""
        mic = MockMicrophoneCapture()
        track = MicrophoneAudioTrack(mic)

        assert track.is_speaking is False
        assert track.SAMPLES_PER_FRAME == 320  # 20ms at 16kHz

    @pytest.mark.asyncio
    async def test_recv_returns_audio_frame(self) -> None:
        """Test recv() returns a valid audio frame."""
        mic = MockMicrophoneCapture()
        await mic.start()
        track = MicrophoneAudioTrack(mic)

        # recv() should return an AudioFrame
        try:
            from av import AudioFrame

            frame = await track.recv()

            assert frame is not None
            assert isinstance(frame, AudioFrame)
            assert frame.sample_rate == 16000
            assert frame.samples == 320

        except ImportError:
            pytest.skip("PyAV not available")
        finally:
            await mic.stop()

    @pytest.mark.asyncio
    async def test_recv_voice_detection(self) -> None:
        """Test recv() tracks voice activity."""
        mic = MockMicrophoneCapture()
        await mic.start()
        track = MicrophoneAudioTrack(mic)

        try:
            from av import AudioFrame

            # Initially not speaking
            assert track.is_speaking is False

            # Simulate voice
            mic.simulate_voice(duration=2.0)
            await asyncio.sleep(0.05)

            # Get a frame while voice active
            frame = await track.recv()
            assert track.is_speaking is True

        except ImportError:
            pytest.skip("PyAV not available")
        finally:
            await mic.stop()

    @pytest.mark.asyncio
    async def test_recv_multiple_frames(self) -> None:
        """Test recv() can be called multiple times."""
        mic = MockMicrophoneCapture()
        await mic.start()
        track = MicrophoneAudioTrack(mic)

        try:
            from av import AudioFrame

            # Get multiple frames
            frames = []
            for _ in range(5):
                frame = await track.recv()
                frames.append(frame)

            assert len(frames) == 5
            for frame in frames:
                assert isinstance(frame, AudioFrame)
                assert frame.samples == 320

        except ImportError:
            pytest.skip("PyAV not available")
        finally:
            await mic.stop()

    def test_track_stop(self) -> None:
        """Test track stop() method."""
        mic = MockMicrophoneCapture()
        track = MicrophoneAudioTrack(mic)

        # Should not raise
        track.stop()

    def test_track_repr(self) -> None:
        """Test track string representation."""
        mic = MockMicrophoneCapture()
        track = MicrophoneAudioTrack(mic)

        repr_str = repr(track)
        assert "MicrophoneAudioTrack" in repr_str
        assert "rate=16000" in repr_str


class TestMicrophoneAudioTrackSilence:
    """Tests for silence frame generation."""

    def test_create_silence(self) -> None:
        """Test _create_silence returns correct size."""
        mic = MockMicrophoneCapture()
        track = MicrophoneAudioTrack(mic)

        silence = track._create_silence()

        # 320 samples * 2 bytes = 640 bytes
        assert len(silence) == 640
        assert silence == bytes(640)  # All zeros


class TestMicrophoneAudioTrackVAD:
    """Tests for VAD (Voice Activity Detection) gating."""

    @pytest.mark.asyncio
    async def test_speaking_state_transitions(self) -> None:
        """Test is_speaking state transitions correctly."""
        mic = MockMicrophoneCapture()
        await mic.start()
        track = MicrophoneAudioTrack(mic)

        try:
            from av import AudioFrame

            # Start silent
            assert track.is_speaking is False

            # Simulate voice
            mic.simulate_voice(duration=0.5)
            await asyncio.sleep(0.05)
            await track.recv()
            assert track.is_speaking is True

            # Stop voice and wait for silence threshold
            mic.simulate_silence()
            # Need to call recv() multiple times over VAD_SILENCE_DURATION_MS (500ms)
            # Each recv() sleeps ~20ms, so we need ~30 calls to exceed 500ms
            for _ in range(30):
                await track.recv()
            assert track.is_speaking is False

        except ImportError:
            pytest.skip("PyAV not available")
        finally:
            await mic.stop()
