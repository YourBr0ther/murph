"""Tests for emulator physics simulation."""

import math

import pytest

from emulator.config import EmulatorConfig
from emulator.physics import MotorPhysics, OdometryCalculator


class TestMotorPhysics:
    """Tests for MotorPhysics class."""

    def test_initialization(self) -> None:
        """Test motor physics initializes with zero state."""
        config = EmulatorConfig()
        physics = MotorPhysics(config)

        left, right = physics.get_speeds()
        assert left == 0.0
        assert right == 0.0

        left_ticks, right_ticks = physics.get_encoder_ticks()
        assert left_ticks == 0
        assert right_ticks == 0

    def test_set_target_speeds(self) -> None:
        """Test setting target speeds."""
        config = EmulatorConfig()
        physics = MotorPhysics(config)

        physics.set_target_speeds(0.5, 0.8)
        assert physics.left_motor.target_speed == 0.5
        assert physics.right_motor.target_speed == 0.8

    def test_target_speed_clamping(self) -> None:
        """Test target speeds are clamped to -1.0 to 1.0."""
        config = EmulatorConfig()
        physics = MotorPhysics(config)

        physics.set_target_speeds(1.5, -1.5)
        assert physics.left_motor.target_speed == 1.0
        assert physics.right_motor.target_speed == -1.0

    def test_gradual_acceleration(self) -> None:
        """Test that speed changes gradually, not instantly."""
        config = EmulatorConfig(acceleration_rate=0.1)
        physics = MotorPhysics(config)

        physics.set_target_speeds(1.0, 1.0)

        # After one tick, speed should NOT be at target yet
        physics.tick(0.05)  # 50ms tick
        left, right = physics.get_speeds()

        assert left < 1.0  # Not yet at full speed
        assert right < 1.0
        assert left > 0.0  # But should have started moving
        assert right > 0.0

    def test_reaches_target_speed(self) -> None:
        """Test that speed eventually reaches target."""
        config = EmulatorConfig(acceleration_rate=0.1)
        physics = MotorPhysics(config)

        physics.set_target_speeds(0.5, 0.5)

        # Run multiple ticks to reach target
        for _ in range(100):
            physics.tick(0.05)

        left, right = physics.get_speeds()
        assert abs(left - 0.5) < 0.01
        assert abs(right - 0.5) < 0.01

    def test_stop(self) -> None:
        """Test stop sets target to zero."""
        config = EmulatorConfig()
        physics = MotorPhysics(config)

        physics.set_target_speeds(1.0, 1.0)
        physics.stop()

        assert physics.left_motor.target_speed == 0.0
        assert physics.right_motor.target_speed == 0.0

    def test_encoder_ticks_increase_with_movement(self) -> None:
        """Test encoder ticks increase when moving."""
        config = EmulatorConfig()
        physics = MotorPhysics(config)

        # Run at full speed for a while
        physics.set_target_speeds(1.0, 1.0)
        for _ in range(20):
            physics.tick(0.05)

        left_ticks, right_ticks = physics.get_encoder_ticks()
        assert left_ticks > 0
        assert right_ticks > 0

    def test_encoder_ticks_negative_when_backward(self) -> None:
        """Test encoder ticks go negative when moving backward."""
        config = EmulatorConfig()
        physics = MotorPhysics(config)

        physics.set_target_speeds(-1.0, -1.0)
        for _ in range(20):
            physics.tick(0.05)

        left_ticks, right_ticks = physics.get_encoder_ticks()
        assert left_ticks < 0
        assert right_ticks < 0

    def test_reset(self) -> None:
        """Test reset clears all state."""
        config = EmulatorConfig()
        physics = MotorPhysics(config)

        # Move and accumulate state
        physics.set_target_speeds(1.0, 1.0)
        for _ in range(20):
            physics.tick(0.05)

        # Reset
        physics.reset()

        left, right = physics.get_speeds()
        assert left == 0.0
        assert right == 0.0

        left_ticks, right_ticks = physics.get_encoder_ticks()
        assert left_ticks == 0
        assert right_ticks == 0

    def test_deceleration_faster_than_acceleration(self) -> None:
        """Test deceleration can be faster than acceleration."""
        config = EmulatorConfig(acceleration_rate=0.05, deceleration_rate=0.1)
        physics = MotorPhysics(config)

        # Accelerate to full speed
        physics.set_target_speeds(1.0, 1.0)
        accel_ticks = 0
        while physics.left_motor.actual_speed < 0.99:
            physics.tick(0.05)
            accel_ticks += 1
            if accel_ticks > 1000:
                break  # Safety limit

        # Now decelerate
        physics.stop()
        decel_ticks = 0
        while physics.left_motor.actual_speed > 0.01:
            physics.tick(0.05)
            decel_ticks += 1
            if decel_ticks > 1000:
                break  # Safety limit

        # Deceleration should take fewer ticks (faster rate)
        assert decel_ticks < accel_ticks


class TestOdometryCalculator:
    """Tests for OdometryCalculator class."""

    def test_initialization(self) -> None:
        """Test odometry starts at origin."""
        config = EmulatorConfig()
        odom = OdometryCalculator(config)

        x, y, heading = odom.get_position()
        assert x == 0.0
        assert y == 0.0
        assert heading == 0.0

    def test_forward_movement(self) -> None:
        """Test straight forward movement increases x position."""
        config = EmulatorConfig(
            wheel_diameter_mm=60.0,
            encoder_ticks_per_revolution=1440,
        )
        odom = OdometryCalculator(config)

        # Simulate both wheels moving same amount forward
        wheel_circumference = math.pi * 60.0  # ~188.5mm
        ticks_per_rev = 1440
        # 1 revolution = ~188.5mm = 1440 ticks

        # Move 1 full revolution on each wheel
        x, y, heading = odom.update(ticks_per_rev, ticks_per_rev)

        # Should have moved forward about 188.5mm
        assert x > 180  # Allow some tolerance
        assert x < 200
        assert abs(y) < 1  # Should not move sideways
        assert abs(heading) < 1  # Should not rotate

    def test_backward_movement(self) -> None:
        """Test backward movement decreases x position."""
        config = EmulatorConfig()
        odom = OdometryCalculator(config)

        # Move backward (negative ticks)
        x, y, heading = odom.update(-1000, -1000)

        assert x < 0  # Moved backward
        assert abs(y) < 1  # No sideways movement

    def test_turn_in_place(self) -> None:
        """Test turning in place changes heading but not position."""
        config = EmulatorConfig()
        odom = OdometryCalculator(config)

        # Spin in place: left wheel backward, right wheel forward
        x, y, heading = odom.update(-500, 500)

        # Position should stay near origin
        assert abs(x) < 10
        assert abs(y) < 10
        # Heading should have changed
        assert abs(heading) > 1

    def test_reset(self) -> None:
        """Test reset returns to origin."""
        config = EmulatorConfig()
        odom = OdometryCalculator(config)

        # Move around
        odom.update(1000, 1000)
        odom.update(-500, 500)

        # Reset
        odom.reset()

        x, y, heading = odom.get_position()
        assert x == 0.0
        assert y == 0.0
        assert heading == 0.0

    def test_cumulative_updates(self) -> None:
        """Test position accumulates over multiple updates."""
        config = EmulatorConfig()
        odom = OdometryCalculator(config)

        # Multiple small forward movements
        for _ in range(10):
            odom.update(100, 100)

        x, y, heading = odom.get_position()

        # Should have accumulated position
        assert x > 0
