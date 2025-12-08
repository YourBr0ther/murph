"""
Murph - Local Behavior Reflexes
Instant reflexive behaviors that run on the Pi without server round-trip.

These handle time-critical responses to physical events like being picked up,
bumped, or shaken. Response time is < 50ms.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable

from shared.constants import (
    BUMP_DECELERATION_THRESHOLD,
    FREEFALL_THRESHOLD,
    PICKUP_ACCELERATION_THRESHOLD,
    SHAKE_FREQUENCY_THRESHOLD,
)
from shared.messages import IMUData

logger = logging.getLogger(__name__)


class ReflexType(Enum):
    """Types of reflexive behaviors."""

    PICKED_UP_FAST = auto()  # Quick pickup (> 1.5g)
    PICKED_UP_GENTLE = auto()  # Slow lift
    BUMP = auto()  # Collision
    SHAKE = auto()  # Being shaken
    FALLING = auto()  # Freefall
    SET_DOWN = auto()  # Placed back down


@dataclass
class ReflexConfig:
    """Configuration for a single reflex."""

    trigger_threshold: float
    expression: str
    sound: str
    cooldown_ms: float = 500.0  # Minimum time between triggers


# Reflex configurations
REFLEX_CONFIGS: dict[ReflexType, ReflexConfig] = {
    ReflexType.PICKED_UP_FAST: ReflexConfig(
        trigger_threshold=PICKUP_ACCELERATION_THRESHOLD + 0.5,  # > 2g
        expression="surprised",
        sound="startled",
        cooldown_ms=1000,
    ),
    ReflexType.PICKED_UP_GENTLE: ReflexConfig(
        trigger_threshold=PICKUP_ACCELERATION_THRESHOLD,  # > 1.5g
        expression="happy",
        sound="chirp",
        cooldown_ms=1000,
    ),
    ReflexType.BUMP: ReflexConfig(
        trigger_threshold=BUMP_DECELERATION_THRESHOLD,  # > 2g decel
        expression="surprised",
        sound="oof",
        cooldown_ms=500,
    ),
    ReflexType.FALLING: ReflexConfig(
        trigger_threshold=FREEFALL_THRESHOLD,  # < 0.3g
        expression="scared",
        sound="scream",
        cooldown_ms=2000,
    ),
    ReflexType.SET_DOWN: ReflexConfig(
        trigger_threshold=1.1,  # Return to ~1g
        expression="neutral",
        sound="sigh",
        cooldown_ms=1000,
    ),
}


class ReflexController:
    """
    Monitors sensors and triggers instant reflexive responses.

    These bypass the server brain for < 50ms response time.
    The server is notified after the reflex executes.
    """

    def __init__(
        self,
        on_expression: Callable[[str], None] | None = None,
        on_sound: Callable[[str], None] | None = None,
        on_trigger: Callable[[str, float], None] | None = None,
    ) -> None:
        """
        Initialize reflex controller.

        Args:
            on_expression: Callback to set expression
            on_sound: Callback to play sound
            on_trigger: Callback to notify server of trigger (name, intensity)
        """
        self._on_expression = on_expression
        self._on_sound = on_sound
        self._on_trigger = on_trigger

        self._running = False
        self._last_trigger_time: dict[ReflexType, float] = {}

        # State tracking
        self._was_held = False
        self._accel_history: list[float] = []
        self._gyro_history: list[float] = []

    async def start(self) -> None:
        """Start reflex monitoring."""
        self._running = True
        logger.info("Reflex controller started")

    async def stop(self) -> None:
        """Stop reflex monitoring."""
        self._running = False
        logger.info("Reflex controller stopped")

    def process_imu(self, imu: IMUData) -> None:
        """
        Process IMU data and trigger reflexes if needed.

        Called by sensor loop at high frequency (50-100Hz).
        """
        if not self._running:
            return

        # Calculate acceleration magnitude
        accel_mag = imu.acceleration_magnitude

        # Track history for pattern detection
        self._accel_history.append(accel_mag)
        if len(self._accel_history) > 10:
            self._accel_history.pop(0)

        gyro_mag = (imu.gyro_x**2 + imu.gyro_y**2 + imu.gyro_z**2) ** 0.5
        self._gyro_history.append(gyro_mag)
        if len(self._gyro_history) > 10:
            self._gyro_history.pop(0)

        # Check for reflexes
        self._check_pickup(accel_mag)
        self._check_falling(accel_mag)
        self._check_bump(accel_mag)
        self._check_shake()
        self._check_set_down(accel_mag)

    def _check_pickup(self, accel_mag: float) -> None:
        """Check for pickup events."""
        if self._was_held:
            return  # Already being held

        fast_config = REFLEX_CONFIGS[ReflexType.PICKED_UP_FAST]
        gentle_config = REFLEX_CONFIGS[ReflexType.PICKED_UP_GENTLE]

        if accel_mag > fast_config.trigger_threshold:
            if self._trigger_reflex(ReflexType.PICKED_UP_FAST, accel_mag):
                self._was_held = True

        elif accel_mag > gentle_config.trigger_threshold:
            if self._trigger_reflex(ReflexType.PICKED_UP_GENTLE, accel_mag):
                self._was_held = True

    def _check_falling(self, accel_mag: float) -> None:
        """Check for freefall (being dropped)."""
        config = REFLEX_CONFIGS[ReflexType.FALLING]

        if accel_mag < config.trigger_threshold:
            self._trigger_reflex(ReflexType.FALLING, accel_mag)

    def _check_bump(self, accel_mag: float) -> None:
        """Check for collision/bump."""
        config = REFLEX_CONFIGS[ReflexType.BUMP]

        # Detect sudden spike in acceleration
        if len(self._accel_history) >= 3:
            prev_avg = sum(self._accel_history[:-1]) / len(self._accel_history[:-1])
            if accel_mag - prev_avg > config.trigger_threshold:
                self._trigger_reflex(ReflexType.BUMP, accel_mag)

    def _check_shake(self) -> None:
        """Check for shaking (high-frequency oscillation)."""
        if len(self._gyro_history) < 5:
            return

        avg_gyro = sum(self._gyro_history) / len(self._gyro_history)

        # High average gyro indicates shaking
        if avg_gyro > 100:  # deg/s threshold
            self._trigger_reflex(ReflexType.SHAKE, avg_gyro / 100)

    def _check_set_down(self, accel_mag: float) -> None:
        """Check for being set down (return to rest after being held)."""
        if not self._was_held:
            return

        config = REFLEX_CONFIGS[ReflexType.SET_DOWN]

        # Check if acceleration returned to ~1g (resting)
        if 0.9 < accel_mag < config.trigger_threshold:
            # Check stability over recent samples
            if len(self._accel_history) >= 5:
                variation = max(self._accel_history) - min(self._accel_history)
                if variation < 0.2:  # Stable
                    if self._trigger_reflex(ReflexType.SET_DOWN, accel_mag):
                        self._was_held = False

    def _trigger_reflex(self, reflex_type: ReflexType, intensity: float) -> bool:
        """
        Trigger a reflexive response.

        Args:
            reflex_type: Type of reflex to trigger
            intensity: Intensity of the triggering event

        Returns:
            True if reflex was triggered, False if on cooldown
        """
        config = REFLEX_CONFIGS.get(reflex_type)
        if not config:
            return False

        # Check cooldown
        now = time.time() * 1000
        last = self._last_trigger_time.get(reflex_type, 0)

        if now - last < config.cooldown_ms:
            return False

        self._last_trigger_time[reflex_type] = now

        # Execute reflex immediately
        logger.info(f"Reflex triggered: {reflex_type.name} (intensity: {intensity:.2f})")

        # Call callbacks directly (they should be fast)
        # In production, these would update hardware immediately
        if self._on_expression:
            try:
                self._on_expression(config.expression)
            except Exception as e:
                logger.warning(f"Expression callback failed: {e}")

        if self._on_sound:
            try:
                self._on_sound(config.sound)
            except Exception as e:
                logger.warning(f"Sound callback failed: {e}")

        # Notify server
        if self._on_trigger:
            try:
                normalized_intensity = min(1.0, intensity / 3.0)
                self._on_trigger(reflex_type.name.lower(), normalized_intensity)
            except Exception as e:
                logger.warning(f"Trigger callback failed: {e}")

        return True

    @property
    def is_being_held(self) -> bool:
        """Check if robot is currently being held."""
        return self._was_held

    def get_state(self) -> dict:
        """Get current reflex state."""
        return {
            "is_being_held": self._was_held,
            "recent_accel": self._accel_history[-1] if self._accel_history else None,
            "recent_gyro": self._gyro_history[-1] if self._gyro_history else None,
            "last_triggers": {
                rt.name: t for rt, t in self._last_trigger_time.items()
            },
        }
