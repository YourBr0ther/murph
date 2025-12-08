"""
Murph - Sensor Base Classes
Abstract interfaces for all robot sensors.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, TypeVar

T = TypeVar("T")


class Sensor(ABC):
    """
    Base class for all sensors.

    Provides common interface for sensor initialization,
    reading, and shutdown.
    """

    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the sensor hardware.

        Returns:
            True if initialization succeeded, False otherwise.
        """
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Safely shut down the sensor."""
        pass

    @abstractmethod
    async def read(self) -> Any:
        """
        Read current sensor value.

        Returns:
            Sensor-specific data type
        """
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        """Check if sensor is ready for reading."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Get sensor name for logging."""
        pass


class StreamingSensor(Sensor):
    """
    Sensor that streams data continuously.

    Use for sensors that should continuously report data
    (e.g., IMU, camera) rather than being polled.
    """

    @abstractmethod
    async def start_streaming(
        self,
        callback: Callable[[Any], None],
        interval_ms: float = 100,
    ) -> None:
        """
        Start streaming sensor data.

        Args:
            callback: Function to call with each reading
            interval_ms: Milliseconds between readings
        """
        pass

    @abstractmethod
    async def stop_streaming(self) -> None:
        """Stop the data stream."""
        pass

    @abstractmethod
    def is_streaming(self) -> bool:
        """Check if sensor is currently streaming."""
        pass
