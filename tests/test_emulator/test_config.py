"""Tests for EmulatorConfig."""

import pytest

from emulator.config import EmulatorConfig


class TestEmulatorConfig:
    """Test EmulatorConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = EmulatorConfig()

        assert config.server_host == "0.0.0.0"
        assert config.server_port == 6765
        assert config.base_speed == 50.0
        assert config.acceleration_rate == 0.1
        assert config.sensor_interval_ms == 100
        assert config.video_enabled is True
        assert config.audio_enabled is False

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = EmulatorConfig(
            server_host="192.168.1.100",
            server_port=9000,
            base_speed=100.0,
            video_enabled=False,
        )

        assert config.server_host == "192.168.1.100"
        assert config.server_port == 9000
        assert config.base_speed == 100.0
        assert config.video_enabled is False

    def test_validate_success(self) -> None:
        """Test validation with valid config."""
        config = EmulatorConfig()
        errors = config.validate()
        assert errors == []

    def test_validate_invalid_base_speed(self) -> None:
        """Test validation catches invalid base_speed."""
        config = EmulatorConfig(base_speed=0)
        errors = config.validate()
        assert "base_speed must be positive" in errors

    def test_validate_invalid_acceleration_rate(self) -> None:
        """Test validation catches invalid acceleration_rate."""
        config = EmulatorConfig(acceleration_rate=-1)
        errors = config.validate()
        assert "acceleration_rate must be positive" in errors

    def test_validate_invalid_sensor_interval(self) -> None:
        """Test validation catches invalid sensor_interval_ms."""
        config = EmulatorConfig(sensor_interval_ms=0)
        errors = config.validate()
        assert "sensor_interval_ms must be positive" in errors

    def test_validate_multiple_errors(self) -> None:
        """Test validation can return multiple errors."""
        config = EmulatorConfig(
            base_speed=0,
            acceleration_rate=0,
            sensor_interval_ms=0,
        )
        errors = config.validate()
        assert len(errors) == 3

    def test_from_dict(self) -> None:
        """Test creating config from dictionary."""
        data = {
            "server_host": "10.0.0.1",
            "base_speed": 75.0,
            "video_enabled": False,
        }
        config = EmulatorConfig.from_dict(data)

        assert config.server_host == "10.0.0.1"
        assert config.base_speed == 75.0
        assert config.video_enabled is False
        # Defaults for unspecified values
        assert config.server_port == 6765

    def test_from_dict_ignores_unknown_keys(self) -> None:
        """Test from_dict ignores unknown keys."""
        data = {
            "server_host": "test",
            "unknown_key": "should be ignored",
            "another_unknown": 123,
        }
        config = EmulatorConfig.from_dict(data)

        assert config.server_host == "test"
        # Should not raise, unknown keys are ignored

    def test_reflex_thresholds(self) -> None:
        """Test reflex threshold defaults."""
        config = EmulatorConfig()

        assert config.pickup_threshold_g == 1.5
        assert config.bump_threshold_g == 2.0
        assert config.shake_gyro_threshold == 100.0
        assert config.reflex_cooldown_s == 1.0

    def test_encoder_config(self) -> None:
        """Test encoder configuration defaults."""
        config = EmulatorConfig()

        assert config.encoder_ticks_per_revolution == 1440
        assert config.wheel_diameter_mm == 60.0
        assert config.encoder_noise == 0.01
