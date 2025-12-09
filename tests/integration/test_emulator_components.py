"""
Integration tests for emulator components.

Tests the VirtualPi emulator and its protocol compliance.
"""

import asyncio
import sys
from pathlib import Path

import pytest

# Add project root to path for direct imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from emulator.virtual_pi import (
    VirtualPi,
    VirtualRobotState,
    EmulatorReflexProcessor,
)
from shared.messages import (
    IMUData,
    TouchData,
    MotorDirection,
    ScanType,
)


class TestVirtualRobotState:
    """Test VirtualRobotState dataclass."""

    def test_default_values(self):
        """Test default state values."""
        state = VirtualRobotState()
        assert state.x == 0.0
        assert state.y == 0.0
        assert state.heading == 0.0
        assert state.is_moving is False
        assert state.current_expression == "neutral"
        assert state.server_connected is False

    def test_to_dict(self):
        """Test state serialization."""
        state = VirtualRobotState(
            x=10.5,
            y=-5.0,
            heading=45.0,
            is_moving=True,
            current_expression="happy",
        )
        d = state.to_dict()
        assert d["x"] == 10.5
        assert d["y"] == -5.0
        assert d["heading"] == 45.0
        assert d["is_moving"] is True
        assert d["current_expression"] == "happy"

    def test_imu_values_in_dict(self):
        """Test IMU values are included in state dict."""
        state = VirtualRobotState(
            imu_accel_x=0.1,
            imu_accel_y=-0.2,
            imu_accel_z=-0.98,
        )
        d = state.to_dict()
        assert "imu_accel_x" in d
        assert d["imu_accel_x"] == 0.1
        assert d["imu_accel_y"] == -0.2
        assert d["imu_accel_z"] == -0.98


class TestEmulatorReflexProcessor:
    """Test EmulatorReflexProcessor class."""

    def test_init(self):
        """Test processor initialization."""
        processor = EmulatorReflexProcessor()
        assert processor._accel_history == []
        assert processor._gyro_history == []
        assert processor._was_held is False

    def test_normal_imu_no_trigger(self):
        """Test that normal IMU data doesn't trigger reflexes."""
        processor = EmulatorReflexProcessor()
        # Normal at-rest data
        imu = IMUData(
            accel_x=0.01,
            accel_y=0.01,
            accel_z=-1.0,
            gyro_x=0.1,
            gyro_y=0.1,
            gyro_z=0.1,
            temperature=25.0,
        )
        # Process multiple times to build history
        for _ in range(10):
            trigger = processor.process_imu(imu)
        assert trigger is None

    def test_pickup_detection(self):
        """Test pickup detection with high acceleration."""
        processor = EmulatorReflexProcessor()
        # First build normal history
        normal_imu = IMUData(
            accel_x=0.0,
            accel_y=0.0,
            accel_z=-1.0,
            gyro_x=0.0,
            gyro_y=0.0,
            gyro_z=0.0,
            temperature=25.0,
        )
        for _ in range(5):
            processor.process_imu(normal_imu)

        # Now simulate pickup with strong acceleration
        # Need magnitude > 1.5g: sqrt(1.2^2 + 1.2^2 + 0^2) = ~1.7
        pickup_imu = IMUData(
            accel_x=1.2,
            accel_y=1.2,
            accel_z=0.0,
            gyro_x=0.0,
            gyro_y=0.0,
            gyro_z=0.0,
            temperature=25.0,
        )
        trigger = processor.process_imu(pickup_imu)
        assert trigger in ["picked_up_gentle", "picked_up_fast"]

    def test_cooldown_prevents_repeated_triggers(self):
        """Test that cooldowns prevent repeated triggers."""
        processor = EmulatorReflexProcessor()
        # Trigger pickup
        pickup_imu = IMUData(
            accel_x=1.2,
            accel_y=1.2,
            accel_z=0.5,
            gyro_x=0.0,
            gyro_y=0.0,
            gyro_z=0.0,
            temperature=25.0,
        )
        trigger1 = processor.process_imu(pickup_imu)
        assert trigger1 is not None

        # Immediately try again - should be blocked by cooldown
        trigger2 = processor.process_imu(pickup_imu)
        assert trigger2 is None


class TestVirtualPi:
    """Test VirtualPi class."""

    def test_init(self):
        """Test VirtualPi initialization."""
        pi = VirtualPi(server_host="localhost", server_port=9999)
        assert pi._host == "localhost"
        assert pi._port == 9999
        assert pi.is_connected is False

    def test_state_property(self):
        """Test state property access."""
        pi = VirtualPi()
        assert isinstance(pi.state, VirtualRobotState)
        assert pi.state.x == 0.0

    def test_simulate_touch(self):
        """Test touch simulation."""
        pi = VirtualPi()
        pi.simulate_touch([3, 4, 5, 6])
        assert pi.state.is_being_touched is True
        assert pi.state.touched_electrodes == [3, 4, 5, 6]

    def test_simulate_release(self):
        """Test touch release."""
        pi = VirtualPi()
        pi.simulate_touch([1, 2])
        pi.simulate_release()
        assert pi.state.is_being_touched is False
        assert pi.state.touched_electrodes == []

    @pytest.mark.asyncio
    async def test_simulate_pickup(self):
        """Test pickup simulation sets state."""
        pi = VirtualPi()
        pi.simulate_pickup(duration=0.1)
        assert pi.state.simulated_pickup is True
        await asyncio.sleep(0.15)  # Let the clear task run
        assert pi.state.simulated_pickup is False

    @pytest.mark.asyncio
    async def test_simulate_bump(self):
        """Test bump simulation sets state."""
        pi = VirtualPi()
        pi.simulate_bump(duration=0.1)
        assert pi.state.simulated_bump is True
        await asyncio.sleep(0.15)  # Let the clear task run
        assert pi.state.simulated_bump is False

    @pytest.mark.asyncio
    async def test_simulate_shake(self):
        """Test shake simulation sets state."""
        pi = VirtualPi()
        pi.simulate_shake(duration=0.1)
        assert pi.state.simulated_shake is True
        await asyncio.sleep(0.15)  # Let the clear task run
        assert pi.state.simulated_shake is False

    def test_state_change_callback(self):
        """Test on_state_change callback."""
        states_received = []

        def callback(state):
            states_received.append(state.to_dict())

        pi = VirtualPi(on_state_change=callback)
        pi.simulate_touch([1])

        assert len(states_received) == 1
        assert states_received[0]["is_being_touched"] is True

    @pytest.mark.asyncio
    async def test_start_stop(self):
        """Test VirtualPi lifecycle without server."""
        pi = VirtualPi(server_host="localhost", server_port=59999)
        await pi.start()
        assert pi._running is True
        await asyncio.sleep(0.1)  # Let it try to connect briefly
        await pi.stop()
        assert pi._running is False


class TestVirtualPiIMUGeneration:
    """Test IMU data generation."""

    def test_imu_at_rest(self):
        """Test IMU values at rest (not moving, no simulation)."""
        pi = VirtualPi()
        imu = pi._generate_imu_data()

        # At rest, accel_z should be approximately -1.0 (gravity)
        assert -1.5 < imu.accel_z < -0.5
        # X and Y should be near 0
        assert -0.5 < imu.accel_x < 0.5
        assert -0.5 < imu.accel_y < 0.5

    def test_imu_during_simulated_pickup(self):
        """Test IMU during pickup simulation."""
        pi = VirtualPi()
        pi._state.simulated_pickup = True
        imu = pi._generate_imu_data()

        # Pickup adds +1.0 to accel_z
        assert imu.accel_z > -0.5  # -1.0 + 1.0 = ~0

    def test_imu_during_simulated_bump(self):
        """Test IMU during bump simulation."""
        pi = VirtualPi()
        pi._state.simulated_bump = True
        imu = pi._generate_imu_data()

        # Bump adds 2.0-3.0 to accel_x
        assert imu.accel_x > 1.5


class TestVirtualPiTouchGeneration:
    """Test touch data generation."""

    def test_touch_when_not_touched(self):
        """Test touch data when not being touched."""
        pi = VirtualPi()
        touch = pi._generate_touch_data()
        assert touch.is_touched is False
        assert touch.touched_electrodes == []

    def test_touch_when_touched(self):
        """Test touch data when electrodes are touched."""
        pi = VirtualPi()
        pi.simulate_touch([3, 4, 5])
        touch = pi._generate_touch_data()
        assert touch.is_touched is True
        assert touch.touched_electrodes == [3, 4, 5]
