"""
Murph - Emulator Configuration
Centralized configuration for tunable emulator parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from shared.constants import (
    ACCELERATION_LIMIT,
    DEFAULT_SERVER_HOST,
    DEFAULT_SERVER_PORT,
)


@dataclass
class EmulatorConfig:
    """Configuration for the emulator with sensible defaults."""

    # Server connection
    server_host: str = DEFAULT_SERVER_HOST
    server_port: int = DEFAULT_SERVER_PORT

    # Motor simulation
    base_speed: float = 50.0  # units per second at speed=1.0
    acceleration_rate: float = ACCELERATION_LIMIT  # speed change per tick
    deceleration_rate: float = ACCELERATION_LIMIT * 1.5  # slightly faster braking
    wheel_base_mm: float = 100.0  # distance between wheels

    # Encoder simulation
    encoder_ticks_per_revolution: int = 1440  # typical encoder resolution
    wheel_diameter_mm: float = 60.0  # wheel diameter
    encoder_noise: float = 0.01  # noise as fraction of tick count

    # Sensor simulation
    sensor_interval_ms: int = 100  # how often to send sensor data
    imu_noise: float = 0.02  # accelerometer/gyro noise level

    # Timing
    physics_tick_ms: int = 50  # physics simulation tick rate

    # Features
    video_enabled: bool = True  # enable webcam video streaming
    audio_enabled: bool = False  # enable microphone audio capture

    # Reflex thresholds (can override shared constants)
    pickup_threshold_g: float = 1.5  # acceleration threshold for pickup detection
    bump_threshold_g: float = 2.0  # acceleration spike for bump detection
    shake_gyro_threshold: float = 100.0  # gyro threshold for shake detection
    reflex_cooldown_s: float = 1.0  # cooldown between reflex triggers

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []

        if self.base_speed <= 0:
            errors.append("base_speed must be positive")
        if self.acceleration_rate <= 0:
            errors.append("acceleration_rate must be positive")
        if self.sensor_interval_ms <= 0:
            errors.append("sensor_interval_ms must be positive")
        if self.physics_tick_ms <= 0:
            errors.append("physics_tick_ms must be positive")
        if self.wheel_diameter_mm <= 0:
            errors.append("wheel_diameter_mm must be positive")
        if self.encoder_ticks_per_revolution <= 0:
            errors.append("encoder_ticks_per_revolution must be positive")

        return errors

    @classmethod
    def from_dict(cls, data: dict) -> EmulatorConfig:
        """Create config from dictionary, ignoring unknown keys."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)
