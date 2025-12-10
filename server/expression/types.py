"""
Murph - Expression Types
Defines expression enums and metadata for the robot's facial expressions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ExpressionType(Enum):
    """
    Available expression types for the robot's display.

    These map directly to the expressions supported by the Pi display controller.
    """

    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    CURIOUS = "curious"
    SURPRISED = "surprised"
    SLEEPY = "sleepy"
    PLAYFUL = "playful"
    LOVE = "love"
    SCARED = "scared"
    ALERT = "alert"


class ExpressionCategory(Enum):
    """Categories for grouping expressions by their emotional domain."""

    NEUTRAL = "neutral"  # Default, calm states
    SOCIAL = "social"  # Interaction with people
    PLAY = "play"  # Playful, curious activities
    COMFORT = "comfort"  # Rest, tired states
    SAFETY = "safety"  # Alert, scared states


@dataclass(frozen=True)
class ExpressionMetadata:
    """
    Metadata describing an expression's emotional properties.

    Attributes:
        name: The expression name (matches ExpressionType value)
        display_name: Human-readable name for the expression
        category: The emotional category this expression belongs to
        valence: Emotional valence from -1.0 (negative) to 1.0 (positive)
        arousal: Arousal level from 0.0 (calm) to 1.0 (excited)
    """

    name: str
    display_name: str
    category: ExpressionCategory
    valence: float  # -1.0 to 1.0 (negative to positive emotion)
    arousal: float  # 0.0 to 1.0 (calm to excited)

    def __post_init__(self) -> None:
        """Validate the metadata values."""
        if not -1.0 <= self.valence <= 1.0:
            raise ValueError(f"valence must be between -1.0 and 1.0, got {self.valence}")
        if not 0.0 <= self.arousal <= 1.0:
            raise ValueError(f"arousal must be between 0.0 and 1.0, got {self.arousal}")
