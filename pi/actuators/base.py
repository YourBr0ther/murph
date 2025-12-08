"""
Murph - Actuator Base Classes
Abstract interfaces for all robot actuators.
"""

from abc import ABC, abstractmethod
from typing import Any


class Actuator(ABC):
    """
    Base class for all actuators.

    Provides common interface for hardware initialization,
    shutdown, and status checking.
    """

    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the hardware.

        Returns:
            True if initialization succeeded, False otherwise.
        """
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Safely shut down the actuator."""
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        """Check if actuator is ready for commands."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Get actuator name for logging."""
        pass


class MotorController(Actuator):
    """
    Abstract motor controller interface.

    Controls robot locomotion via differential drive
    (4x N20 motors via DRV8833 driver).
    """

    @abstractmethod
    async def move(
        self,
        direction: str,
        speed: float,
        duration_ms: float = 0,
    ) -> None:
        """
        Move the robot in a direction.

        Args:
            direction: "forward", "backward", "left", "right"
            speed: 0.0-1.0 normalized speed
            duration_ms: Duration in milliseconds (0 = until stopped)
        """
        pass

    @abstractmethod
    async def turn(self, angle: float, speed: float) -> None:
        """
        Turn the robot by an angle.

        Args:
            angle: Degrees to turn (positive = clockwise)
            speed: 0.0-1.0 normalized speed
        """
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop all motor movement immediately."""
        pass

    @abstractmethod
    def get_state(self) -> dict[str, Any]:
        """Get current motor state for feedback."""
        pass


class DisplayController(Actuator):
    """
    Abstract display controller interface.

    Controls the 1.3" OLED SSD1306 display for facial expressions.
    """

    @abstractmethod
    async def set_expression(self, name: str) -> None:
        """
        Set the facial expression.

        Args:
            name: Expression name (e.g., "happy", "sad", "curious", "neutral")
        """
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear the display."""
        pass

    @abstractmethod
    def get_current_expression(self) -> str:
        """Get the currently displayed expression."""
        pass


class AudioController(Actuator):
    """
    Abstract audio controller interface.

    Controls the speaker (MAX98357A) for sound playback.
    """

    @abstractmethod
    async def play_sound(self, name: str, volume: float = 1.0) -> None:
        """
        Play a sound effect.

        Args:
            name: Sound name (e.g., "greeting", "happy", "curious")
            volume: 0.0-1.0 normalized volume
        """
        pass

    @abstractmethod
    async def stop_sound(self) -> None:
        """Stop any currently playing sound."""
        pass

    @abstractmethod
    def is_playing(self) -> bool:
        """Check if a sound is currently playing."""
        pass

    @abstractmethod
    def get_current_sound(self) -> str | None:
        """Get the name of the currently playing sound, if any."""
        pass
