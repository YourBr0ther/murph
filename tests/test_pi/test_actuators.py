"""
Tests for Pi actuator mock implementations.
"""

import pytest
import asyncio

from pi.actuators import (
    MockMotorController,
    MockDisplayController,
    MockAudioController,
)


class TestMockMotorController:
    """Tests for MockMotorController."""

    @pytest.fixture
    def motor(self):
        return MockMotorController()

    @pytest.mark.asyncio
    async def test_initialize(self, motor):
        assert not motor.is_ready()
        result = await motor.initialize()
        assert result is True
        assert motor.is_ready()

    @pytest.mark.asyncio
    async def test_shutdown(self, motor):
        await motor.initialize()
        await motor.shutdown()
        assert not motor.is_ready()

    @pytest.mark.asyncio
    async def test_move_forward(self, motor):
        await motor.initialize()
        await motor.move("forward", 0.5, 0)
        state = motor.get_state()
        assert state["direction"] == "forward"
        assert state["speed"] == 0.5
        assert state["is_moving"] is True

    @pytest.mark.asyncio
    async def test_move_with_duration(self, motor):
        await motor.initialize()
        await motor.move("forward", 0.5, 100)  # 100ms
        state = motor.get_state()
        assert state["is_moving"] is True

        # Wait for auto-stop
        await asyncio.sleep(0.15)
        state = motor.get_state()
        assert state["is_moving"] is False

    @pytest.mark.asyncio
    async def test_turn(self, motor):
        await motor.initialize()
        # Turn 90 degrees - should take some time
        await motor.turn(90, 0.5)
        state = motor.get_state()
        # After turn completes, should be stopped
        assert state["direction"] in ("left", "right", "stop")

    @pytest.mark.asyncio
    async def test_stop(self, motor):
        await motor.initialize()
        await motor.move("forward", 0.8, 0)
        await motor.stop()
        state = motor.get_state()
        assert state["is_moving"] is False
        assert state["speed"] == 0.0

    @pytest.mark.asyncio
    async def test_invalid_direction(self, motor):
        await motor.initialize()
        await motor.move("invalid_direction", 0.5, 0)
        state = motor.get_state()
        # Should ignore invalid direction
        assert state["is_moving"] is False

    @pytest.mark.asyncio
    async def test_speed_clamping(self, motor):
        await motor.initialize()
        await motor.move("forward", 1.5, 0)  # Over max
        state = motor.get_state()
        assert state["speed"] == 1.0

        await motor.move("forward", -0.5, 0)  # Negative
        state = motor.get_state()
        assert state["speed"] == 0.0

    @pytest.mark.asyncio
    async def test_differential_drive(self, motor):
        await motor.initialize()

        await motor.move("forward", 0.5, 0)
        state = motor.get_state()
        assert state["left_speed"] == 0.5
        assert state["right_speed"] == 0.5

        await motor.move("left", 0.5, 0)
        state = motor.get_state()
        assert state["left_speed"] == -0.5
        assert state["right_speed"] == 0.5


class TestMockDisplayController:
    """Tests for MockDisplayController."""

    @pytest.fixture
    def display(self):
        return MockDisplayController()

    @pytest.mark.asyncio
    async def test_initialize(self, display):
        result = await display.initialize()
        assert result is True
        assert display.is_ready()
        assert display.get_current_expression() == "neutral"

    @pytest.mark.asyncio
    async def test_set_expression(self, display):
        await display.initialize()
        await display.set_expression("happy")
        assert display.get_current_expression() == "happy"

    @pytest.mark.asyncio
    async def test_unknown_expression_fallback(self, display):
        await display.initialize()
        await display.set_expression("unknown_expression")
        assert display.get_current_expression() == "neutral"

    @pytest.mark.asyncio
    async def test_clear(self, display):
        await display.initialize()
        await display.set_expression("sad")
        await display.clear()
        assert display.get_current_expression() == "neutral"

    @pytest.mark.asyncio
    async def test_get_expression_art(self, display):
        await display.initialize()
        art = display.get_expression_art()
        assert isinstance(art, list)
        assert len(art) > 0


class TestMockAudioController:
    """Tests for MockAudioController."""

    @pytest.fixture
    def speaker(self):
        return MockAudioController()

    @pytest.mark.asyncio
    async def test_initialize(self, speaker):
        result = await speaker.initialize()
        assert result is True
        assert speaker.is_ready()

    @pytest.mark.asyncio
    async def test_play_sound(self, speaker):
        await speaker.initialize()
        await speaker.play_sound("greeting", 0.8)
        assert speaker.is_playing()
        assert speaker.get_current_sound() == "greeting"
        assert speaker.get_volume() == 0.8

    @pytest.mark.asyncio
    async def test_sound_auto_stops(self, speaker):
        await speaker.initialize()
        await speaker.play_sound("alert")  # Short sound
        await asyncio.sleep(0.5)
        assert not speaker.is_playing()

    @pytest.mark.asyncio
    async def test_stop_sound(self, speaker):
        await speaker.initialize()
        await speaker.play_sound("greeting")
        await speaker.stop_sound()
        assert not speaker.is_playing()
        assert speaker.get_current_sound() is None

    @pytest.mark.asyncio
    async def test_volume_clamping(self, speaker):
        await speaker.initialize()
        await speaker.play_sound("happy", 1.5)  # Over max
        assert speaker.get_volume() == 1.0
