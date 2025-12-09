"""Tests for emulator microphone simulation."""

import asyncio
import time

import pytest

from emulator.audio.microphone import (
    AudioState,
    MicrophoneCapture,
    MockMicrophoneCapture,
)


class TestMockMicrophoneCapture:
    """Tests for MockMicrophoneCapture class."""

    def test_initialization(self) -> None:
        """Test mock microphone initializes correctly."""
        mic = MockMicrophoneCapture()
        assert mic.is_available
        state = mic.get_state()
        assert not state.is_running
        assert state.audio_level == 0.0
        assert not state.is_voice_detected

    @pytest.mark.asyncio
    async def test_start_stop(self) -> None:
        """Test starting and stopping mock microphone."""
        mic = MockMicrophoneCapture()

        result = await mic.start()
        assert result is True
        assert mic.get_state().is_running

        await mic.stop()
        assert not mic.get_state().is_running

    @pytest.mark.asyncio
    async def test_audio_levels_update(self) -> None:
        """Test that audio levels update when running."""
        mic = MockMicrophoneCapture()
        await mic.start()

        # Wait for a few updates
        await asyncio.sleep(0.2)

        state = mic.get_state()
        assert state.samples_captured > 0
        assert state.last_update_ms > 0

        await mic.stop()

    @pytest.mark.asyncio
    async def test_simulate_voice(self) -> None:
        """Test manual voice simulation."""
        mic = MockMicrophoneCapture()
        await mic.start()

        # Wait for initial update
        await asyncio.sleep(0.1)

        # Simulate voice
        mic.simulate_voice(duration=0.5)

        # Should detect voice immediately after next update
        await asyncio.sleep(0.1)
        assert mic.is_voice_detected()
        assert mic.get_audio_level() > 0.2

        await mic.stop()

    @pytest.mark.asyncio
    async def test_simulate_silence(self) -> None:
        """Test manual silence simulation."""
        mic = MockMicrophoneCapture()
        await mic.start()

        mic.simulate_voice(duration=10.0)  # Long voice
        await asyncio.sleep(0.1)
        assert mic.is_voice_detected()

        mic.simulate_silence()
        await asyncio.sleep(0.1)
        assert not mic.is_voice_detected()

        await mic.stop()

    def test_audio_level_in_range(self) -> None:
        """Test audio level stays within 0.0-1.0 range."""
        mic = MockMicrophoneCapture(base_noise_level=0.05)

        # Manually trigger many updates
        for _ in range(100):
            mic._update_audio_state()
            level = mic.get_audio_level()
            assert 0.0 <= level <= 1.0


class TestMicrophoneCapture:
    """Tests for MicrophoneCapture class."""

    def test_initialization(self) -> None:
        """Test microphone initializes (may fall back to mock)."""
        mic = MicrophoneCapture()
        # Should either be available or have mock fallback
        state = mic.get_state()
        assert not state.is_running

    @pytest.mark.asyncio
    async def test_start_stop(self) -> None:
        """Test starting and stopping microphone (may use mock)."""
        mic = MicrophoneCapture()

        result = await mic.start()
        # Should succeed even without real hardware (uses mock)
        assert result is True
        assert mic.get_state().is_running

        await mic.stop()
        assert not mic.get_state().is_running

    @pytest.mark.asyncio
    async def test_get_audio_level(self) -> None:
        """Test getting audio level."""
        mic = MicrophoneCapture()
        await mic.start()

        # Wait for updates
        await asyncio.sleep(0.2)

        level = mic.get_audio_level()
        assert 0.0 <= level <= 1.0

        await mic.stop()


class TestAudioState:
    """Tests for AudioState dataclass."""

    def test_default_values(self) -> None:
        """Test AudioState default values."""
        state = AudioState()
        assert not state.is_running
        assert state.audio_level == 0.0
        assert not state.is_voice_detected
        assert state.samples_captured == 0
        assert state.last_update_ms == 0

    def test_custom_values(self) -> None:
        """Test AudioState with custom values."""
        state = AudioState(
            is_running=True,
            audio_level=0.5,
            is_voice_detected=True,
            samples_captured=1000,
            last_update_ms=12345,
        )
        assert state.is_running
        assert state.audio_level == 0.5
        assert state.is_voice_detected
        assert state.samples_captured == 1000
        assert state.last_update_ms == 12345
