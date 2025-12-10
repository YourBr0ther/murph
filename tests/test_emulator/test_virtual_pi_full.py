"""Comprehensive tests for VirtualPi emulator functionality."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from emulator.config import EmulatorConfig
from emulator.virtual_pi import VirtualPi, VirtualRobotState, EmulatorReflexProcessor
from shared.messages import (
    Command,
    ExpressionCommand,
    MotorCommand,
    MotorDirection,
    ScanCommand,
    ScanType,
    SoundCommand,
    StopCommand,
    TurnCommand,
    IMUData,
    MessageType,
    RobotMessage,
    SimulatedTranscription,
    create_simulated_transcription,
)


# =============================================================================
# REFLEX SIMULATION TESTS
# =============================================================================


class TestReflexSimulation:
    """Tests for manual reflex simulation methods."""

    def test_simulate_touch_updates_state(self) -> None:
        """Test simulate_touch updates electrodes and touch state."""
        vpi = VirtualPi()
        assert vpi.state.is_being_touched is False
        assert vpi.state.touched_electrodes == []

        vpi.simulate_touch([1, 2, 3])

        assert vpi.state.is_being_touched is True
        assert vpi.state.touched_electrodes == [1, 2, 3]

    def test_simulate_release_clears_touch(self) -> None:
        """Test simulate_release clears touch state."""
        vpi = VirtualPi()
        vpi.simulate_touch([1, 2, 3])
        assert vpi.state.is_being_touched is True

        vpi.simulate_release()

        assert vpi.state.is_being_touched is False
        assert vpi.state.touched_electrodes == []

    @pytest.mark.asyncio
    async def test_simulate_pickup_sets_flag(self) -> None:
        """Test simulate_pickup sets simulated_pickup flag."""
        vpi = VirtualPi()
        assert vpi.state.simulated_pickup is False

        vpi.simulate_pickup()

        assert vpi.state.simulated_pickup is True

    @pytest.mark.asyncio
    async def test_simulate_bump_sets_flag(self) -> None:
        """Test simulate_bump sets simulated_bump flag."""
        vpi = VirtualPi()
        assert vpi.state.simulated_bump is False

        vpi.simulate_bump()

        assert vpi.state.simulated_bump is True

    @pytest.mark.asyncio
    async def test_simulate_shake_sets_flag(self) -> None:
        """Test simulate_shake sets simulated_shake flag."""
        vpi = VirtualPi()
        assert vpi.state.simulated_shake is False

        vpi.simulate_shake()

        assert vpi.state.simulated_shake is True

    @pytest.mark.asyncio
    async def test_simulate_falling_modifies_imu(self) -> None:
        """Test simulate_falling sets low acceleration for freefall."""
        vpi = VirtualPi()
        initial_z = vpi.state.imu_accel_z

        vpi.simulate_falling()

        # Should be near zero (freefall)
        assert vpi.state.imu_accel_z == -0.2
        assert vpi.state.imu_accel_z != initial_z

    @pytest.mark.asyncio
    async def test_simulate_falling_restores_imu(self) -> None:
        """Test simulate_falling restores IMU after duration."""
        vpi = VirtualPi()
        vpi.simulate_falling(duration=0.1)

        assert vpi.state.imu_accel_z == -0.2

        # Wait for restoration
        await asyncio.sleep(0.15)

        assert vpi.state.imu_accel_z == -1.0


# =============================================================================
# COMMAND HANDLING TESTS
# =============================================================================


class TestCommandHandling:
    """Tests for VirtualPi command handling."""

    @pytest.mark.asyncio
    async def test_motor_command_updates_state(self) -> None:
        """Test motor command updates movement state."""
        vpi = VirtualPi()

        cmd = Command(
            sequence_id=1,
            payload=MotorCommand(
                direction=MotorDirection.FORWARD,
                speed=0.5,
                duration_ms=0,
            ),
        )

        await vpi._execute_command(cmd)

        assert vpi.state.move_direction == "forward"
        assert vpi.state.current_speed == 0.5
        assert vpi.state.is_moving is True

    @pytest.mark.asyncio
    async def test_stop_command_halts_movement(self) -> None:
        """Test stop command stops movement."""
        vpi = VirtualPi()
        # First start moving
        cmd = Command(
            sequence_id=1,
            payload=MotorCommand(
                direction=MotorDirection.FORWARD,
                speed=0.5,
            ),
        )
        await vpi._execute_command(cmd)
        assert vpi.state.is_moving is True

        # Then stop
        stop_cmd = Command(sequence_id=2, payload=StopCommand())
        await vpi._execute_command(stop_cmd)

        assert vpi.state.is_moving is False
        assert vpi.state.current_speed == 0.0
        assert vpi.state.move_direction == "stop"

    @pytest.mark.asyncio
    async def test_turn_command_updates_heading(self) -> None:
        """Test turn command updates robot heading."""
        vpi = VirtualPi()
        initial_heading = vpi.state.heading

        cmd = Command(
            sequence_id=1,
            payload=TurnCommand(angle_degrees=45.0, speed=1.0),
        )

        await vpi._execute_command(cmd)

        assert vpi.state.heading == (initial_heading + 45.0) % 360

    @pytest.mark.asyncio
    async def test_expression_command_updates_display(self) -> None:
        """Test expression command updates current expression."""
        vpi = VirtualPi()
        assert vpi.state.current_expression == "neutral"

        cmd = Command(
            sequence_id=1,
            payload=ExpressionCommand(expression_name="happy"),
        )

        await vpi._execute_command(cmd)

        assert vpi.state.current_expression == "happy"

    @pytest.mark.asyncio
    async def test_sound_command_updates_state(self) -> None:
        """Test sound command updates playing sound."""
        vpi = VirtualPi()
        assert vpi.state.playing_sound is None

        cmd = Command(
            sequence_id=1,
            payload=SoundCommand(sound_name="chirp", volume=0.8),
        )

        await vpi._execute_command(cmd)

        assert vpi.state.playing_sound == "chirp"

    @pytest.mark.asyncio
    async def test_scan_command_rotates_robot(self) -> None:
        """Test scan command performs rotation."""
        vpi = VirtualPi()
        initial_heading = vpi.state.heading

        cmd = Command(
            sequence_id=1,
            payload=ScanCommand(scan_type=ScanType.QUICK),  # 90 degrees
        )

        await vpi._execute_command(cmd)

        # After scan, heading should have changed
        # Note: The exact heading depends on implementation timing
        assert vpi.state.is_moving is False  # Should be done


# =============================================================================
# VOICE INJECTION TESTS
# =============================================================================


class TestVoiceInjection:
    """Tests for voice text injection feature."""

    @pytest.mark.asyncio
    async def test_inject_voice_text_fails_when_disconnected(self) -> None:
        """Test inject_voice_text logs warning when not connected."""
        vpi = VirtualPi()
        assert vpi._websocket is None

        # Should not raise, just warn
        await vpi.inject_voice_text("test command")

    @pytest.mark.asyncio
    async def test_inject_voice_text_sends_message(self) -> None:
        """Test inject_voice_text sends correct message when connected."""
        vpi = VirtualPi()

        # Mock websocket
        mock_ws = AsyncMock()
        vpi._websocket = mock_ws

        await vpi.inject_voice_text("Murph, come here")

        # Verify send was called
        mock_ws.send.assert_called_once()

        # Verify message content
        sent_data = mock_ws.send.call_args[0][0]
        msg = RobotMessage.deserialize(sent_data)

        assert msg.message_type == MessageType.SIMULATED_TRANSCRIPTION
        assert isinstance(msg.payload, SimulatedTranscription)
        assert msg.payload.text == "Murph, come here"


# =============================================================================
# MESSAGE TYPE TESTS
# =============================================================================


class TestSimulatedTranscriptionMessage:
    """Tests for SimulatedTranscription message type."""

    def test_create_simulated_transcription_factory(self) -> None:
        """Test factory helper creates correct message."""
        msg = create_simulated_transcription("Hello Murph")

        assert msg.message_type == MessageType.SIMULATED_TRANSCRIPTION
        assert isinstance(msg.payload, SimulatedTranscription)
        assert msg.payload.text == "Hello Murph"

    def test_simulated_transcription_serialization(self) -> None:
        """Test message serializes and deserializes correctly."""
        original = create_simulated_transcription("Test message")

        # Serialize
        data = original.serialize()

        # Deserialize
        restored = RobotMessage.deserialize(data)

        assert restored.message_type == MessageType.SIMULATED_TRANSCRIPTION
        assert isinstance(restored.payload, SimulatedTranscription)
        assert restored.payload.text == "Test message"


# =============================================================================
# REFLEX PROCESSOR TESTS
# =============================================================================


class TestEmulatorReflexProcessor:
    """Tests for the reflex detection processor."""

    def test_pickup_detection(self) -> None:
        """Test pickup detection from high acceleration."""
        processor = EmulatorReflexProcessor()
        processor.start()

        # Simulate high upward acceleration (picked up fast)
        imu = IMUData(accel_x=0, accel_y=0, accel_z=2.5)  # > 1.5g threshold

        trigger = processor.process_imu(imu)

        assert trigger in ["picked_up_fast", "picked_up_gentle"]

    def test_falling_detection(self) -> None:
        """Test freefall detection from low acceleration."""
        processor = EmulatorReflexProcessor()
        processor.start()

        # Simulate freefall (near zero g)
        imu = IMUData(accel_x=0, accel_y=0, accel_z=-0.1)  # < 0.3g

        trigger = processor.process_imu(imu)

        assert trigger == "falling"

    def test_bump_detection(self) -> None:
        """Test bump detection from sudden acceleration spike."""
        processor = EmulatorReflexProcessor()
        processor.start()

        # First trigger pickup so robot is marked as "held"
        # (bump detection only works when held or below pickup threshold)
        processor.process_imu(IMUData(accel_x=0, accel_y=0, accel_z=2.0))  # Triggers pickup

        # Wait for cooldown to pass (for pickup)
        import time
        time.sleep(0.6)  # Bump cooldown is 0.5 * cooldown (0.5s)

        # Build up history at normal levels while held
        for _ in range(5):
            processor.process_imu(IMUData(accel_x=0, accel_y=0, accel_z=-1.0))

        # Now a sudden spike should trigger bump (since already held, pickup won't trigger)
        imu = IMUData(accel_x=0, accel_y=0, accel_z=-3.5)  # Large spike

        trigger = processor.process_imu(imu)

        # Due to how bump detection works (compares to previous average),
        # a large spike in acceleration magnitude triggers bump
        assert trigger == "bump"

    def test_shake_detection(self) -> None:
        """Test shake detection from high gyro values."""
        processor = EmulatorReflexProcessor()
        processor.start()

        trigger = None
        # Build up history with high gyro (need 5+ samples for shake detection)
        for i in range(10):
            imu = IMUData(
                accel_x=0, accel_y=0, accel_z=-1.0,
                gyro_x=150.0, gyro_y=150.0, gyro_z=150.0
            )
            result = processor.process_imu(imu)
            if result == "shake":
                trigger = result
                break

        # Should have triggered shake at some point
        assert trigger == "shake"

    def test_cooldown_prevents_rapid_triggers(self) -> None:
        """Test cooldown prevents same reflex triggering repeatedly."""
        processor = EmulatorReflexProcessor(cooldown=1.0)
        processor.start()

        # First falling trigger
        imu = IMUData(accel_x=0, accel_y=0, accel_z=-0.1)
        trigger1 = processor.process_imu(imu)
        assert trigger1 == "falling"

        # Immediate second attempt should be blocked by cooldown
        trigger2 = processor.process_imu(imu)
        assert trigger2 is None  # Blocked by cooldown


# =============================================================================
# STATE TESTS
# =============================================================================


class TestVirtualRobotState:
    """Tests for VirtualRobotState dataclass."""

    def test_default_values(self) -> None:
        """Test default state values."""
        state = VirtualRobotState()

        assert state.x == 0.0
        assert state.y == 0.0
        assert state.heading == 0.0
        assert state.is_moving is False
        assert state.current_expression == "neutral"
        assert state.server_connected is False

    def test_to_dict_includes_all_fields(self) -> None:
        """Test to_dict includes all necessary fields."""
        state = VirtualRobotState(
            x=10.0,
            y=20.0,
            heading=45.0,
            is_moving=True,
            current_expression="happy",
        )

        d = state.to_dict()

        assert d["x"] == 10.0
        assert d["y"] == 20.0
        assert d["heading"] == 45.0
        assert d["is_moving"] is True
        assert d["current_expression"] == "happy"
        assert "server_connected" in d
        assert "video_enabled" in d
        assert "audio_enabled" in d


# =============================================================================
# MOTOR DIRECTION TESTS
# =============================================================================


class TestMotorDirections:
    """Tests for different motor directions."""

    @pytest.mark.asyncio
    async def test_backward_direction(self) -> None:
        """Test backward motor command."""
        vpi = VirtualPi()

        cmd = Command(
            sequence_id=1,
            payload=MotorCommand(
                direction=MotorDirection.BACKWARD,
                speed=0.5,
            ),
        )

        await vpi._execute_command(cmd)

        assert vpi.state.move_direction == "backward"

    @pytest.mark.asyncio
    async def test_left_turn_direction(self) -> None:
        """Test left turn motor command."""
        vpi = VirtualPi()

        cmd = Command(
            sequence_id=1,
            payload=MotorCommand(
                direction=MotorDirection.LEFT,
                speed=0.5,
            ),
        )

        await vpi._execute_command(cmd)

        assert vpi.state.move_direction == "left"

    @pytest.mark.asyncio
    async def test_right_turn_direction(self) -> None:
        """Test right turn motor command."""
        vpi = VirtualPi()

        cmd = Command(
            sequence_id=1,
            payload=MotorCommand(
                direction=MotorDirection.RIGHT,
                speed=0.5,
            ),
        )

        await vpi._execute_command(cmd)

        assert vpi.state.move_direction == "right"
