"""
Murph - Actuator Module
Hardware controllers for robot output (motors, display, speaker).
"""

from .base import Actuator, AudioController, DisplayController, MotorController
from .display import MockDisplayController, SSD1306DisplayController
from .motors import DRV8833MotorController, MockMotorController
from .pygame_display import PygameDisplayController
from .speaker import MAX98357AudioController, MockAudioController

__all__ = [
    # Base classes
    "Actuator",
    "MotorController",
    "DisplayController",
    "AudioController",
    # Mock implementations
    "MockMotorController",
    "MockDisplayController",
    "MockAudioController",
    # Real hardware implementations
    "DRV8833MotorController",
    "SSD1306DisplayController",
    "MAX98357AudioController",
    # Alternative implementations
    "PygameDisplayController",
]
