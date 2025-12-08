"""
Murph - Personality Traits
Defines personality traits that modify behavior preferences and need decay rates.
"""

from dataclasses import dataclass, field
import random
from typing import Any


# Personality presets
PRESETS: dict[str, dict[str, float]] = {
    "playful": {
        "playfulness": 0.8,
        "boldness": 0.3,
        "sociability": 0.5,
        "curiosity": 0.4,
        "affectionate": 0.3,
        "energy_level": 0.7,
    },
    "shy": {
        "playfulness": -0.3,
        "boldness": -0.6,
        "sociability": -0.4,
        "curiosity": 0.2,
        "affectionate": 0.5,
        "energy_level": -0.2,
    },
    "explorer": {
        "playfulness": 0.3,
        "boldness": 0.7,
        "sociability": 0.1,
        "curiosity": 0.9,
        "affectionate": 0.0,
        "energy_level": 0.5,
    },
    "cuddly": {
        "playfulness": 0.2,
        "boldness": -0.2,
        "sociability": 0.7,
        "curiosity": -0.1,
        "affectionate": 0.9,
        "energy_level": -0.3,
    },
    "hyper": {
        "playfulness": 0.9,
        "boldness": 0.5,
        "sociability": 0.6,
        "curiosity": 0.7,
        "affectionate": 0.4,
        "energy_level": 0.9,
    },
    "balanced": {
        "playfulness": 0.0,
        "boldness": 0.0,
        "sociability": 0.0,
        "curiosity": 0.0,
        "affectionate": 0.0,
        "energy_level": 0.0,
    },
}

# Behavior to trait mappings
# Each behavior is influenced by certain traits
BEHAVIOR_TRAIT_MAPPINGS: dict[str, dict[str, float]] = {
    "explore": {"curiosity": 0.4, "boldness": 0.3, "energy_level": 0.2},
    "investigate": {"curiosity": 0.5, "boldness": 0.2},
    "greet": {"sociability": 0.5, "affectionate": 0.2, "boldness": 0.1},
    "follow": {"sociability": 0.4, "affectionate": 0.3},
    "play": {"playfulness": 0.5, "energy_level": 0.3},
    "chase": {"playfulness": 0.4, "energy_level": 0.4, "boldness": 0.1},
    "bounce": {"playfulness": 0.4, "energy_level": 0.5},
    "nuzzle": {"affectionate": 0.6, "sociability": 0.2},
    "request_attention": {"sociability": 0.4, "affectionate": 0.3},
    "rest": {"energy_level": -0.5},  # Low energy = more likely to rest
    "find_cozy_spot": {"energy_level": -0.3},
    "retreat": {"boldness": -0.5},  # Low boldness = more likely to retreat
    "hide": {"boldness": -0.6, "sociability": -0.2},
    "approach_trusted": {"sociability": 0.3, "affectionate": 0.2},
    "wander": {"curiosity": 0.2, "energy_level": 0.2},
}

# Need to trait mappings
# Traits affect how quickly needs decay or how important they feel
NEED_TRAIT_MAPPINGS: dict[str, dict[str, float]] = {
    "energy": {"energy_level": 0.3},  # High energy = faster energy decay
    "curiosity": {"curiosity": 0.4},  # High curiosity = faster curiosity decay
    "play": {"playfulness": 0.4, "energy_level": 0.2},
    "social": {"sociability": 0.5},
    "affection": {"affectionate": 0.5},
    "comfort": {"energy_level": -0.2},  # Low energy = more comfort-seeking
    "safety": {"boldness": -0.3},  # Low boldness = safety is more important
}


@dataclass
class Personality:
    """
    Personality traits that modify Murph's behavior preferences.

    Each trait ranges from -1.0 to +1.0:
    - playfulness: How much Murph enjoys play activities
    - boldness: Willingness to approach new things vs retreat
    - sociability: Desire for social interaction
    - curiosity: Drive to explore and investigate
    - affectionate: Need for physical affection and closeness
    - energy_level: Overall activity level (high = hyperactive, low = lazy)
    """

    playfulness: float = 0.0
    boldness: float = 0.0
    sociability: float = 0.0
    curiosity: float = 0.0
    affectionate: float = 0.0
    energy_level: float = 0.0

    # Valid trait names
    TRAIT_NAMES: list[str] = field(
        default_factory=lambda: [
            "playfulness",
            "boldness",
            "sociability",
            "curiosity",
            "affectionate",
            "energy_level",
        ],
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Clamp all traits to valid range."""
        self.playfulness = self._clamp(self.playfulness)
        self.boldness = self._clamp(self.boldness)
        self.sociability = self._clamp(self.sociability)
        self.curiosity = self._clamp(self.curiosity)
        self.affectionate = self._clamp(self.affectionate)
        self.energy_level = self._clamp(self.energy_level)

    @staticmethod
    def _clamp(value: float, min_val: float = -1.0, max_val: float = 1.0) -> float:
        """Clamp value to range."""
        return max(min_val, min(max_val, value))

    def get_trait(self, name: str) -> float:
        """Get a trait value by name."""
        return getattr(self, name, 0.0)

    def set_trait(self, name: str, value: float) -> None:
        """Set a trait value by name (clamped to valid range)."""
        if hasattr(self, name):
            setattr(self, name, self._clamp(value))

    def get_behavior_modifier(self, behavior_name: str) -> float:
        """
        Calculate a modifier for a specific behavior based on personality.

        Returns a multiplier (typically 0.5 to 1.5) that affects how likely
        Murph is to choose this behavior.
        """
        if behavior_name not in BEHAVIOR_TRAIT_MAPPINGS:
            return 1.0  # No modification for unknown behaviors

        modifier = 1.0
        trait_weights = BEHAVIOR_TRAIT_MAPPINGS[behavior_name]

        for trait_name, weight in trait_weights.items():
            trait_value = self.get_trait(trait_name)
            # Convert trait (-1 to 1) to modifier adjustment
            # weight of 0.5 with trait of 1.0 = +0.5 modifier
            # weight of 0.5 with trait of -1.0 = -0.5 modifier
            adjustment = trait_value * weight
            modifier += adjustment

        # Clamp to reasonable range
        return max(0.3, min(2.0, modifier))

    def get_need_decay_modifier(self, need_name: str) -> float:
        """
        Calculate how personality affects need decay rate.

        Returns a multiplier (typically 0.7 to 1.3) for need decay.
        High curiosity = curiosity need decays faster (you get bored faster).
        """
        if need_name not in NEED_TRAIT_MAPPINGS:
            return 1.0

        modifier = 1.0
        trait_weights = NEED_TRAIT_MAPPINGS[need_name]

        for trait_name, weight in trait_weights.items():
            trait_value = self.get_trait(trait_name)
            adjustment = trait_value * weight
            modifier += adjustment

        # Clamp to reasonable range
        return max(0.5, min(1.5, modifier))

    def get_need_importance_modifier(self, need_name: str) -> float:
        """
        Calculate how personality affects need importance in happiness.

        A very social personality weights social need higher in happiness calc.
        """
        if need_name not in NEED_TRAIT_MAPPINGS:
            return 1.0

        modifier = 1.0
        trait_weights = NEED_TRAIT_MAPPINGS[need_name]

        for trait_name, weight in trait_weights.items():
            trait_value = self.get_trait(trait_name)
            # More subtle effect on importance
            adjustment = trait_value * weight * 0.5
            modifier += adjustment

        return max(0.5, min(1.5, modifier))

    @classmethod
    def random(cls) -> "Personality":
        """Generate a random personality."""
        return cls(
            playfulness=random.uniform(-0.8, 0.8),
            boldness=random.uniform(-0.8, 0.8),
            sociability=random.uniform(-0.8, 0.8),
            curiosity=random.uniform(-0.8, 0.8),
            affectionate=random.uniform(-0.8, 0.8),
            energy_level=random.uniform(-0.8, 0.8),
        )

    @classmethod
    def from_preset(cls, preset_name: str) -> "Personality":
        """Create a personality from a preset name."""
        if preset_name not in PRESETS:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PRESETS.keys())}")
        return cls(**PRESETS[preset_name])

    @classmethod
    def available_presets(cls) -> list[str]:
        """Get list of available preset names."""
        return list(PRESETS.keys())

    def get_state(self) -> dict[str, Any]:
        """Get serializable state for persistence."""
        return {
            "playfulness": self.playfulness,
            "boldness": self.boldness,
            "sociability": self.sociability,
            "curiosity": self.curiosity,
            "affectionate": self.affectionate,
            "energy_level": self.energy_level,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "Personality":
        """Create a Personality from saved state."""
        return cls(
            playfulness=state.get("playfulness", 0.0),
            boldness=state.get("boldness", 0.0),
            sociability=state.get("sociability", 0.0),
            curiosity=state.get("curiosity", 0.0),
            affectionate=state.get("affectionate", 0.0),
            energy_level=state.get("energy_level", 0.0),
        )

    def describe(self) -> str:
        """Get a human-readable description of this personality."""
        descriptions = []

        if self.playfulness > 0.5:
            descriptions.append("very playful")
        elif self.playfulness < -0.5:
            descriptions.append("serious")

        if self.boldness > 0.5:
            descriptions.append("bold and adventurous")
        elif self.boldness < -0.5:
            descriptions.append("timid and cautious")

        if self.sociability > 0.5:
            descriptions.append("highly social")
        elif self.sociability < -0.5:
            descriptions.append("independent")

        if self.curiosity > 0.5:
            descriptions.append("extremely curious")
        elif self.curiosity < -0.5:
            descriptions.append("content with routine")

        if self.affectionate > 0.5:
            descriptions.append("very affectionate")
        elif self.affectionate < -0.5:
            descriptions.append("aloof")

        if self.energy_level > 0.5:
            descriptions.append("high-energy")
        elif self.energy_level < -0.5:
            descriptions.append("laid-back")

        if not descriptions:
            return "balanced personality"

        return ", ".join(descriptions)

    def __str__(self) -> str:
        return f"Personality({self.describe()})"
