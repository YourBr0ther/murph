"""Tests for VirtualPi audio integration."""

import asyncio

import pytest

from emulator.config import EmulatorConfig
from emulator.virtual_pi import VirtualPi, VirtualRobotState


class TestVirtualRobotStateAudio:
    """Tests for VirtualRobotState audio fields."""

    def test_default_audio_state(self) -> None:
        """Test default audio state values."""
        state = VirtualRobotState()
        assert state.audio_enabled is False
        assert state.audio_running is False
        assert state.audio_level == 0.0
        assert state.is_voice_detected is False

    def test_audio_state_in_to_dict(self) -> None:
        """Test audio fields included in to_dict output."""
        state = VirtualRobotState(
            audio_enabled=True,
            audio_running=True,
            audio_level=0.5,
            is_voice_detected=True,
        )
        state_dict = state.to_dict()

        assert "audio_enabled" in state_dict
        assert "audio_running" in state_dict
        assert "audio_level" in state_dict
        assert "is_voice_detected" in state_dict

        assert state_dict["audio_enabled"] is True
        assert state_dict["audio_running"] is True
        assert state_dict["audio_level"] == 0.5
        assert state_dict["is_voice_detected"] is True


class TestVirtualPiAudioIntegration:
    """Tests for VirtualPi audio integration."""

    def test_audio_disabled_by_default(self) -> None:
        """Test audio is disabled by default."""
        vpi = VirtualPi()
        assert vpi.state.audio_enabled is False
        assert vpi._microphone is None

    def test_audio_enabled_via_config(self) -> None:
        """Test audio enabled via EmulatorConfig."""
        config = EmulatorConfig(audio_enabled=True)
        vpi = VirtualPi(config=config)
        assert vpi.state.audio_enabled is True
        # Microphone not initialized until start()
        assert vpi._microphone is None

    @pytest.mark.asyncio
    async def test_audio_init_on_start(self) -> None:
        """Test audio initializes when enabled and start() called."""
        config = EmulatorConfig(audio_enabled=True)
        vpi = VirtualPi(config=config)

        # Start VirtualPi (without connecting to server)
        await vpi.start()

        # Audio should be initialized
        assert vpi._microphone is not None
        assert vpi.state.audio_running is True

        await vpi.stop()
        assert vpi._microphone is None
        assert vpi.state.audio_running is False

    @pytest.mark.asyncio
    async def test_audio_not_init_when_disabled(self) -> None:
        """Test audio not initialized when disabled."""
        config = EmulatorConfig(audio_enabled=False)
        vpi = VirtualPi(config=config)

        await vpi.start()

        assert vpi._microphone is None
        assert vpi.state.audio_running is False

        await vpi.stop()

    @pytest.mark.asyncio
    async def test_audio_state_updates_in_sensor_loop(self) -> None:
        """Test audio state updates from microphone."""
        config = EmulatorConfig(audio_enabled=True)
        vpi = VirtualPi(config=config)

        await vpi.start()

        # Wait for sensor loop to update audio state
        await asyncio.sleep(0.3)

        # Audio level should have been updated from microphone
        # The mock microphone generates varying levels
        assert vpi.state.audio_running is True
        # Audio level will be >= 0 (mock generates noise)
        assert vpi.state.audio_level >= 0.0

        await vpi.stop()

    @pytest.mark.asyncio
    async def test_audio_graceful_shutdown(self) -> None:
        """Test audio shuts down gracefully."""
        config = EmulatorConfig(audio_enabled=True)
        vpi = VirtualPi(config=config)

        await vpi.start()
        assert vpi._microphone is not None

        # Shutdown
        await vpi.stop()

        # Microphone should be cleaned up
        assert vpi._microphone is None
        assert vpi.state.audio_running is False
