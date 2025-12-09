"""
Murph - Emulator Physics Simulation
Realistic motor acceleration/deceleration and encoder simulation.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import EmulatorConfig


@dataclass
class MotorState:
    """State of a single motor."""

    target_speed: float = 0.0  # commanded speed (-1.0 to 1.0)
    actual_speed: float = 0.0  # physical speed after acceleration
    encoder_ticks: int = 0  # total encoder ticks
    position_mm: float = 0.0  # distance traveled in mm


class MotorPhysics:
    """
    Simulates realistic motor physics with acceleration curves.

    Provides gradual speed changes rather than instant response,
    matching real motor behavior more closely.
    """

    def __init__(self, config: EmulatorConfig) -> None:
        """
        Initialize motor physics simulation.

        Args:
            config: Emulator configuration with physics parameters
        """
        self._config = config
        self._left = MotorState()
        self._right = MotorState()

    @property
    def left_motor(self) -> MotorState:
        """Get left motor state."""
        return self._left

    @property
    def right_motor(self) -> MotorState:
        """Get right motor state."""
        return self._right

    def set_target_speeds(self, left: float, right: float) -> None:
        """
        Set target speeds for both motors.

        Args:
            left: Target speed for left motor (-1.0 to 1.0)
            right: Target speed for right motor (-1.0 to 1.0)
        """
        self._left.target_speed = max(-1.0, min(1.0, left))
        self._right.target_speed = max(-1.0, min(1.0, right))

    def set_target_speed_left(self, speed: float) -> None:
        """Set target speed for left motor."""
        self._left.target_speed = max(-1.0, min(1.0, speed))

    def set_target_speed_right(self, speed: float) -> None:
        """Set target speed for right motor."""
        self._right.target_speed = max(-1.0, min(1.0, speed))

    def stop(self) -> None:
        """Set both motors to stop (target = 0)."""
        self._left.target_speed = 0.0
        self._right.target_speed = 0.0

    def tick(self, dt: float) -> tuple[float, float]:
        """
        Advance physics simulation by dt seconds.

        Args:
            dt: Time delta in seconds

        Returns:
            Tuple of (left_actual_speed, right_actual_speed)
        """
        self._update_motor(self._left, dt)
        self._update_motor(self._right, dt)
        return (self._left.actual_speed, self._right.actual_speed)

    def _update_motor(self, motor: MotorState, dt: float) -> None:
        """Update a single motor's physics."""
        speed_diff = motor.target_speed - motor.actual_speed

        if abs(speed_diff) < 0.001:
            # Close enough, snap to target
            motor.actual_speed = motor.target_speed
        else:
            # Determine acceleration or deceleration rate
            if abs(motor.target_speed) < abs(motor.actual_speed):
                # Decelerating (braking)
                rate = self._config.deceleration_rate
            else:
                # Accelerating
                rate = self._config.acceleration_rate

            # Calculate max speed change this tick
            max_change = rate * dt * 20  # rate is per-tick at 50ms, scale for dt

            # Apply change towards target
            if speed_diff > 0:
                motor.actual_speed = min(
                    motor.actual_speed + max_change, motor.target_speed
                )
            else:
                motor.actual_speed = max(
                    motor.actual_speed - max_change, motor.target_speed
                )

        # Update encoder ticks based on actual speed
        # Speed is -1.0 to 1.0, multiply by base_speed for units/second
        distance_mm = motor.actual_speed * self._config.base_speed * dt

        # Update position
        motor.position_mm += distance_mm

        # Calculate encoder ticks from distance
        wheel_circumference = math.pi * self._config.wheel_diameter_mm
        revolutions = distance_mm / wheel_circumference
        ticks = int(revolutions * self._config.encoder_ticks_per_revolution)

        # Add encoder noise
        if ticks != 0 and self._config.encoder_noise > 0:
            noise = random.gauss(0, abs(ticks) * self._config.encoder_noise)
            ticks += int(noise)

        motor.encoder_ticks += ticks

    def reset(self) -> None:
        """Reset all motor state."""
        self._left = MotorState()
        self._right = MotorState()

    def get_speeds(self) -> tuple[float, float]:
        """Get current actual speeds."""
        return (self._left.actual_speed, self._right.actual_speed)

    def get_encoder_ticks(self) -> tuple[int, int]:
        """Get current encoder tick counts."""
        return (self._left.encoder_ticks, self._right.encoder_ticks)

    def get_positions(self) -> tuple[float, float]:
        """Get current positions in mm."""
        return (self._left.position_mm, self._right.position_mm)


class OdometryCalculator:
    """
    Calculates robot position from wheel encoder data.

    Uses differential drive kinematics to estimate position
    from left and right wheel movements.
    """

    def __init__(self, config: EmulatorConfig) -> None:
        """
        Initialize odometry calculator.

        Args:
            config: Emulator configuration with wheel parameters
        """
        self._config = config
        self._x = 0.0  # position in mm
        self._y = 0.0
        self._heading = 0.0  # heading in radians
        self._last_left_ticks = 0
        self._last_right_ticks = 0

    def update(self, left_ticks: int, right_ticks: int) -> tuple[float, float, float]:
        """
        Update position estimate from encoder ticks.

        Args:
            left_ticks: Total left encoder ticks
            right_ticks: Total right encoder ticks

        Returns:
            Tuple of (x_mm, y_mm, heading_degrees)
        """
        # Calculate tick deltas
        delta_left = left_ticks - self._last_left_ticks
        delta_right = right_ticks - self._last_right_ticks
        self._last_left_ticks = left_ticks
        self._last_right_ticks = right_ticks

        # Convert ticks to distance
        wheel_circumference = math.pi * self._config.wheel_diameter_mm
        ticks_per_rev = self._config.encoder_ticks_per_revolution

        left_dist = (delta_left / ticks_per_rev) * wheel_circumference
        right_dist = (delta_right / ticks_per_rev) * wheel_circumference

        # Differential drive kinematics
        # Distance traveled by center of robot
        center_dist = (left_dist + right_dist) / 2

        # Change in heading
        delta_heading = (right_dist - left_dist) / self._config.wheel_base_mm

        # Update position (using midpoint heading for arc approximation)
        mid_heading = self._heading + delta_heading / 2
        self._x += center_dist * math.cos(mid_heading)
        self._y += center_dist * math.sin(mid_heading)
        self._heading += delta_heading

        # Normalize heading to [-pi, pi]
        while self._heading > math.pi:
            self._heading -= 2 * math.pi
        while self._heading < -math.pi:
            self._heading += 2 * math.pi

        return (self._x, self._y, math.degrees(self._heading))

    def reset(self) -> None:
        """Reset odometry to origin."""
        self._x = 0.0
        self._y = 0.0
        self._heading = 0.0
        self._last_left_ticks = 0
        self._last_right_ticks = 0

    def get_position(self) -> tuple[float, float, float]:
        """Get current position estimate (x_mm, y_mm, heading_degrees)."""
        return (self._x, self._y, math.degrees(self._heading))
