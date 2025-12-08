"""
Murph - Sensor Processor
Processes incoming sensor data and updates WorldContext.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from shared.constants import (
    BUMP_DECELERATION_THRESHOLD,
    FREEFALL_THRESHOLD,
    PICKUP_ACCELERATION_THRESHOLD,
    SHAKE_FREQUENCY_THRESHOLD,
)
from shared.messages import IMUData, LocalTrigger, SensorData, TouchData
from server.cognition.behavior.context import WorldContext

logger = logging.getLogger(__name__)


@dataclass
class IMUState:
    """Processed IMU state with event detection."""

    acceleration_magnitude: float = 1.0
    is_being_held: bool = False
    is_falling: bool = False
    recent_bump: bool = False
    is_shaking: bool = False

    # For shake detection
    gyro_history: list[float] = None

    def __post_init__(self):
        if self.gyro_history is None:
            self.gyro_history = []


class SensorProcessor:
    """
    Processes sensor data from Pi and updates perception state.

    Bridges raw sensor Protocol Buffer messages to the WorldContext
    used by the cognition system.

    Responsibilities:
    - Process IMU data for motion detection (pickup, bump, fall)
    - Process touch data for petting detection
    - Update WorldContext with sensor-derived state
    - Handle local trigger events from Pi
    """

    def __init__(self) -> None:
        """Initialize the sensor processor."""
        # Latest sensor data
        self._last_imu: IMUData | None = None
        self._last_touch: TouchData | None = None

        # Processed state
        self._imu_state = IMUState()
        self._is_being_petted = False

        # Timing
        self._last_imu_time: float = 0
        self._last_touch_time: float = 0
        self._bump_cooldown: float = 0  # Prevent repeated bump detection

        # Local trigger state (from Pi reflexes)
        self._recent_local_trigger: str | None = None
        self._local_trigger_time: float = 0

    def process_sensor_data(self, data: SensorData) -> None:
        """
        Process incoming sensor data.

        Called by WebSocket server when sensor messages arrive.

        Args:
            data: Sensor data message from Pi
        """
        payload = data.payload

        if isinstance(payload, IMUData):
            self._process_imu(payload)
        elif isinstance(payload, TouchData):
            self._process_touch(payload)

    def _process_imu(self, imu: IMUData) -> None:
        """
        Process IMU data for motion detection.

        Detects:
        - Being held (upward acceleration)
        - Falling (low acceleration = freefall)
        - Bumps (sudden high deceleration)
        - Shaking (high-frequency oscillation)
        """
        self._last_imu = imu
        self._last_imu_time = time.time()

        # Calculate acceleration magnitude
        accel_mag = imu.acceleration_magnitude
        self._imu_state.acceleration_magnitude = accel_mag

        # Detect being picked up / held
        # When lifted, total acceleration briefly exceeds 1g
        if accel_mag > PICKUP_ACCELERATION_THRESHOLD:
            self._imu_state.is_being_held = True
        elif accel_mag < 1.1:  # Close to 1g = at rest or held steady
            # Keep held state if recently picked up
            pass

        # Detect falling (freefall = near 0g)
        self._imu_state.is_falling = accel_mag < FREEFALL_THRESHOLD

        # Detect bump (sudden high acceleration)
        if accel_mag > BUMP_DECELERATION_THRESHOLD:
            if time.time() - self._bump_cooldown > 1.0:  # 1 second cooldown
                self._imu_state.recent_bump = True
                self._bump_cooldown = time.time()
        else:
            self._imu_state.recent_bump = False

        # Detect shaking (high gyro activity)
        gyro_mag = (imu.gyro_x**2 + imu.gyro_y**2 + imu.gyro_z**2) ** 0.5
        self._imu_state.gyro_history.append(gyro_mag)

        # Keep last 10 samples for frequency detection
        if len(self._imu_state.gyro_history) > 10:
            self._imu_state.gyro_history.pop(0)

        # Simple shake detection: high average gyro activity
        if len(self._imu_state.gyro_history) >= 5:
            avg_gyro = sum(self._imu_state.gyro_history) / len(self._imu_state.gyro_history)
            self._imu_state.is_shaking = avg_gyro > 50  # threshold in deg/s

    def _process_touch(self, touch: TouchData) -> None:
        """
        Process touch data for petting detection.

        Petting = multiple electrodes touched simultaneously.
        """
        self._last_touch = touch
        self._last_touch_time = time.time()

        # Petting detection: 2+ electrodes touched
        self._is_being_petted = touch.is_touched and len(touch.touched_electrodes) >= 2

    def handle_local_trigger(self, trigger: LocalTrigger) -> None:
        """
        Handle local trigger events from Pi.

        These are fast responses triggered on the Pi (e.g., picked_up, bump)
        that the server should know about.

        Args:
            trigger: Local trigger event from Pi
        """
        self._recent_local_trigger = trigger.trigger_name
        self._local_trigger_time = time.time()

        logger.info(
            f"Local trigger received: {trigger.trigger_name} "
            f"(intensity: {trigger.intensity:.2f})"
        )

        # Update state based on trigger
        if trigger.trigger_name in ("picked_up_fast", "picked_up_gentle"):
            self._imu_state.is_being_held = True
        elif trigger.trigger_name == "bump":
            self._imu_state.recent_bump = True
        elif trigger.trigger_name == "falling":
            self._imu_state.is_falling = True

    def update_world_context(self, context: WorldContext) -> None:
        """
        Update WorldContext with latest sensor state.

        Called each cognition cycle to sync sensor state to context.

        Args:
            context: WorldContext to update
        """
        # Physical state from IMU
        context.is_being_held = self._imu_state.is_being_held
        context.recent_bump = self._imu_state.recent_bump

        # Touch state
        context.is_being_petted = self._is_being_petted

        # Add computed triggers based on sensor state
        if self._imu_state.is_being_held:
            context.add_trigger("being_held")
        else:
            context.remove_trigger("being_held")

        if self._is_being_petted:
            context.add_trigger("being_petted")
        else:
            context.remove_trigger("being_petted")

        if self._imu_state.recent_bump:
            context.add_trigger("recent_bump")

        if self._imu_state.is_shaking:
            context.add_trigger("being_shaken")
        else:
            context.remove_trigger("being_shaken")

    def clear_transient_state(self) -> None:
        """
        Clear one-shot states after processing.

        Call at end of cognition cycle.
        """
        self._imu_state.recent_bump = False

        # Clear local trigger if older than 1 second
        if time.time() - self._local_trigger_time > 1.0:
            self._recent_local_trigger = None

    def get_state(self) -> dict:
        """Get current sensor state for debugging."""
        return {
            "imu": {
                "acceleration": self._imu_state.acceleration_magnitude,
                "is_being_held": self._imu_state.is_being_held,
                "is_falling": self._imu_state.is_falling,
                "recent_bump": self._imu_state.recent_bump,
                "is_shaking": self._imu_state.is_shaking,
            },
            "touch": {
                "is_being_petted": self._is_being_petted,
                "touched": self._last_touch.touched_electrodes if self._last_touch else [],
            },
            "local_trigger": self._recent_local_trigger,
            "data_age": {
                "imu": time.time() - self._last_imu_time if self._last_imu_time else None,
                "touch": time.time() - self._last_touch_time if self._last_touch_time else None,
            },
        }
