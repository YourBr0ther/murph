"""
Murph - Motor Controller
DRV8833-based motor driver for differential drive locomotion.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .base import MotorController

logger = logging.getLogger(__name__)


class MockMotorController(MotorController):
    """
    Mock motor controller for testing without hardware.

    Simulates motor behavior by tracking state and logging actions.
    Can be used with the emulator for visual feedback.
    """

    VALID_DIRECTIONS = ("forward", "backward", "left", "right", "stop")

    def __init__(self) -> None:
        self._ready = False
        self._state: dict[str, Any] = {
            "direction": "stop",
            "speed": 0.0,
            "is_moving": False,
            "left_speed": 0.0,
            "right_speed": 0.0,
        }
        self._move_task: asyncio.Task[None] | None = None
        self._state_callback: callable | None = None

    @property
    def name(self) -> str:
        return "MockMotorController"

    async def initialize(self) -> bool:
        """Initialize mock motor controller."""
        self._ready = True
        logger.info("MockMotorController initialized")
        return True

    async def shutdown(self) -> None:
        """Shutdown mock motor controller."""
        await self.stop()
        self._ready = False
        logger.info("MockMotorController shut down")

    def is_ready(self) -> bool:
        return self._ready

    async def move(
        self,
        direction: str,
        speed: float,
        duration_ms: float = 0,
    ) -> None:
        """Simulate motor movement."""
        if not self._ready:
            logger.warning("Motor controller not ready")
            return

        if direction not in self.VALID_DIRECTIONS:
            logger.warning(f"Invalid direction: {direction}")
            return

        # Clamp speed to valid range
        speed = max(0.0, min(1.0, speed))

        # Cancel any existing movement
        if self._move_task and not self._move_task.done():
            self._move_task.cancel()
            try:
                await self._move_task
            except asyncio.CancelledError:
                pass

        # Update state
        self._state["direction"] = direction
        self._state["speed"] = speed
        self._state["is_moving"] = direction != "stop" and speed > 0

        # Calculate wheel speeds for differential drive
        if direction == "forward":
            self._state["left_speed"] = speed
            self._state["right_speed"] = speed
        elif direction == "backward":
            self._state["left_speed"] = -speed
            self._state["right_speed"] = -speed
        elif direction == "left":
            self._state["left_speed"] = -speed
            self._state["right_speed"] = speed
        elif direction == "right":
            self._state["left_speed"] = speed
            self._state["right_speed"] = -speed
        else:
            self._state["left_speed"] = 0.0
            self._state["right_speed"] = 0.0

        logger.debug(
            f"Motor move: {direction} at {speed:.2f} for "
            f"{duration_ms:.0f}ms"
        )

        # Schedule stop if duration specified
        if duration_ms > 0:
            self._move_task = asyncio.create_task(
                self._auto_stop(duration_ms)
            )

    async def _auto_stop(self, duration_ms: float) -> None:
        """Automatically stop after duration."""
        await asyncio.sleep(duration_ms / 1000.0)
        await self.stop()

    async def turn(self, angle: float, speed: float) -> None:
        """Simulate turning by angle."""
        if not self._ready:
            logger.warning("Motor controller not ready")
            return

        # Clamp speed
        speed = max(0.0, min(1.0, speed))

        # Estimate duration based on angle and speed
        # Assume 180 degrees takes 1 second at full speed
        duration_ms = abs(angle) / (180.0 * max(0.1, speed)) * 1000

        direction = "right" if angle > 0 else "left"
        logger.debug(
            f"Motor turn: {angle:.0f}deg at {speed:.2f} "
            f"(~{duration_ms:.0f}ms)"
        )

        await self.move(direction, speed, duration_ms)

    async def stop(self) -> None:
        """Stop all motors immediately."""
        if self._move_task and not self._move_task.done():
            self._move_task.cancel()
            try:
                await self._move_task
            except asyncio.CancelledError:
                pass
            self._move_task = None

        self._state["direction"] = "stop"
        self._state["speed"] = 0.0
        self._state["is_moving"] = False
        self._state["left_speed"] = 0.0
        self._state["right_speed"] = 0.0

        logger.debug("Motor stopped")

    def get_state(self) -> dict[str, Any]:
        """Get current motor state."""
        return self._state.copy()

    def set_state_callback(self, callback: callable) -> None:
        """Set callback for state changes (used by emulator)."""
        self._state_callback = callback


class DRV8833MotorController(MotorController):
    """
    Real hardware implementation using DRV8833 motor driver.

    Requires Raspberry Pi GPIO access.
    """

    # GPIO pin assignments (BCM numbering)
    LEFT_MOTOR_IN1 = 17
    LEFT_MOTOR_IN2 = 27
    RIGHT_MOTOR_IN1 = 22
    RIGHT_MOTOR_IN2 = 23

    PWM_FREQUENCY = 1000  # Hz

    def __init__(self) -> None:
        self._ready = False
        self._gpio = None
        self._left_pwm1 = None
        self._left_pwm2 = None
        self._right_pwm1 = None
        self._right_pwm2 = None
        self._state: dict[str, Any] = {
            "direction": "stop",
            "speed": 0.0,
            "is_moving": False,
            "left_speed": 0.0,
            "right_speed": 0.0,
        }
        self._move_task: asyncio.Task[None] | None = None

    @property
    def name(self) -> str:
        return "DRV8833MotorController"

    async def initialize(self) -> bool:
        """Initialize GPIO and PWM for motor control."""
        try:
            import RPi.GPIO as GPIO

            self._gpio = GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)

            # Setup motor pins
            for pin in [
                self.LEFT_MOTOR_IN1,
                self.LEFT_MOTOR_IN2,
                self.RIGHT_MOTOR_IN1,
                self.RIGHT_MOTOR_IN2,
            ]:
                GPIO.setup(pin, GPIO.OUT)

            # Setup PWM
            self._left_pwm1 = GPIO.PWM(self.LEFT_MOTOR_IN1, self.PWM_FREQUENCY)
            self._left_pwm2 = GPIO.PWM(self.LEFT_MOTOR_IN2, self.PWM_FREQUENCY)
            self._right_pwm1 = GPIO.PWM(self.RIGHT_MOTOR_IN1, self.PWM_FREQUENCY)
            self._right_pwm2 = GPIO.PWM(self.RIGHT_MOTOR_IN2, self.PWM_FREQUENCY)

            # Start with motors stopped
            self._left_pwm1.start(0)
            self._left_pwm2.start(0)
            self._right_pwm1.start(0)
            self._right_pwm2.start(0)

            self._ready = True
            logger.info("DRV8833MotorController initialized")
            return True

        except ImportError:
            logger.error("RPi.GPIO not available - not running on Pi?")
            return False
        except Exception as e:
            logger.error(f"Motor init failed: {e}")
            return False

    async def shutdown(self) -> None:
        """Cleanup GPIO."""
        await self.stop()

        if self._gpio:
            for pwm in [
                self._left_pwm1,
                self._left_pwm2,
                self._right_pwm1,
                self._right_pwm2,
            ]:
                if pwm:
                    pwm.stop()

            self._gpio.cleanup([
                self.LEFT_MOTOR_IN1,
                self.LEFT_MOTOR_IN2,
                self.RIGHT_MOTOR_IN1,
                self.RIGHT_MOTOR_IN2,
            ])

        self._ready = False
        logger.info("DRV8833MotorController shut down")

    def is_ready(self) -> bool:
        return self._ready

    async def move(
        self,
        direction: str,
        speed: float,
        duration_ms: float = 0,
    ) -> None:
        """Control motors for movement."""
        if not self._ready:
            return

        speed = max(0.0, min(1.0, speed))
        duty_cycle = speed * 100  # Convert to percentage

        # Cancel existing movement
        if self._move_task and not self._move_task.done():
            self._move_task.cancel()
            try:
                await self._move_task
            except asyncio.CancelledError:
                pass

        # Set motor speeds based on direction
        if direction == "forward":
            self._set_motor_speeds(duty_cycle, 0, duty_cycle, 0)
            self._state["left_speed"] = speed
            self._state["right_speed"] = speed
        elif direction == "backward":
            self._set_motor_speeds(0, duty_cycle, 0, duty_cycle)
            self._state["left_speed"] = -speed
            self._state["right_speed"] = -speed
        elif direction == "left":
            self._set_motor_speeds(0, duty_cycle, duty_cycle, 0)
            self._state["left_speed"] = -speed
            self._state["right_speed"] = speed
        elif direction == "right":
            self._set_motor_speeds(duty_cycle, 0, 0, duty_cycle)
            self._state["left_speed"] = speed
            self._state["right_speed"] = -speed
        else:
            await self.stop()
            return

        self._state["direction"] = direction
        self._state["speed"] = speed
        self._state["is_moving"] = speed > 0

        if duration_ms > 0:
            self._move_task = asyncio.create_task(
                self._auto_stop(duration_ms)
            )

    def _set_motor_speeds(
        self,
        left1: float,
        left2: float,
        right1: float,
        right2: float,
    ) -> None:
        """Set PWM duty cycles for all motors."""
        if self._left_pwm1:
            self._left_pwm1.ChangeDutyCycle(left1)
        if self._left_pwm2:
            self._left_pwm2.ChangeDutyCycle(left2)
        if self._right_pwm1:
            self._right_pwm1.ChangeDutyCycle(right1)
        if self._right_pwm2:
            self._right_pwm2.ChangeDutyCycle(right2)

    async def _auto_stop(self, duration_ms: float) -> None:
        """Automatically stop after duration."""
        await asyncio.sleep(duration_ms / 1000.0)
        await self.stop()

    async def turn(self, angle: float, speed: float) -> None:
        """Turn by angle using differential drive."""
        speed = max(0.0, min(1.0, speed))
        duration_ms = abs(angle) / (180.0 * max(0.1, speed)) * 1000
        direction = "right" if angle > 0 else "left"
        await self.move(direction, speed, duration_ms)

    async def stop(self) -> None:
        """Stop all motors."""
        if self._move_task and not self._move_task.done():
            self._move_task.cancel()
            try:
                await self._move_task
            except asyncio.CancelledError:
                pass
            self._move_task = None

        self._set_motor_speeds(0, 0, 0, 0)
        self._state["direction"] = "stop"
        self._state["speed"] = 0.0
        self._state["is_moving"] = False
        self._state["left_speed"] = 0.0
        self._state["right_speed"] = 0.0

    def get_state(self) -> dict[str, Any]:
        return self._state.copy()
