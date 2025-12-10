"""
Murph - Behavior Definition
Defines a single behavior that Murph can execute.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Behavior:
    """
    A single behavior that Murph can execute.

    Behaviors have intrinsic properties and can satisfy one or more needs.
    The evaluator scores behaviors based on current context.

    Attributes:
        name: Unique identifier (e.g., "explore", "nuzzle")
        display_name: Human-readable name for logging
        base_value: Intrinsic desirability (0.5-2.0), higher = more desirable
        need_effects: How executing this behavior affects needs
                     Positive values satisfy needs, negative deplete them
        driven_by_needs: Which needs drive this behavior (for need_modifier)
        opportunity_triggers: Context conditions that provide opportunity bonus
        duration_seconds: Expected execution duration
        interruptible: Whether behavior can be interrupted by higher priority
        cooldown_seconds: Minimum time before behavior can repeat
        energy_cost: Energy need cost when executed
        tags: Tags for filtering/grouping behaviors
        time_preferences: Time period preferences (morning/midday/evening/night -> multiplier)
        spontaneous_probability: Chance of spontaneous activation (0.0-1.0)
    """

    name: str
    display_name: str
    base_value: float = 1.0
    need_effects: dict[str, float] = field(default_factory=dict)
    driven_by_needs: list[str] = field(default_factory=list)
    opportunity_triggers: list[str] = field(default_factory=list)
    duration_seconds: float = 5.0
    interruptible: bool = True
    cooldown_seconds: float = 0.0
    energy_cost: float = 0.0
    tags: list[str] = field(default_factory=list)
    # Time-based behavior preferences (time_period -> multiplier)
    # e.g., {"morning": 1.5, "night": 0.5} means preferred in morning
    time_preferences: dict[str, float] = field(default_factory=dict)
    # Probability of spontaneous activation (0.0-1.0)
    # Allows personality expressions to occur randomly
    spontaneous_probability: float = 0.0

    def __post_init__(self) -> None:
        """Validate and clamp values to valid ranges."""
        self.base_value = max(0.1, min(3.0, self.base_value))
        self.duration_seconds = max(0.1, self.duration_seconds)
        self.cooldown_seconds = max(0.0, self.cooldown_seconds)
        self.energy_cost = max(0.0, self.energy_cost)
        self.spontaneous_probability = max(0.0, min(1.0, self.spontaneous_probability))

    def satisfies_need(self, need_name: str) -> bool:
        """Check if this behavior satisfies a specific need."""
        return self.need_effects.get(need_name, 0.0) > 0

    def depletes_need(self, need_name: str) -> bool:
        """Check if this behavior depletes a specific need."""
        return self.need_effects.get(need_name, 0.0) < 0

    def has_tag(self, tag: str) -> bool:
        """Check if behavior has a specific tag."""
        return tag in self.tags

    def get_state(self) -> dict[str, Any]:
        """Get serializable state for persistence."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "base_value": self.base_value,
            "need_effects": self.need_effects.copy(),
            "driven_by_needs": self.driven_by_needs.copy(),
            "opportunity_triggers": self.opportunity_triggers.copy(),
            "duration_seconds": self.duration_seconds,
            "interruptible": self.interruptible,
            "cooldown_seconds": self.cooldown_seconds,
            "energy_cost": self.energy_cost,
            "tags": self.tags.copy(),
            "time_preferences": self.time_preferences.copy(),
            "spontaneous_probability": self.spontaneous_probability,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "Behavior":
        """Create a Behavior from saved state."""
        return cls(
            name=state["name"],
            display_name=state.get("display_name", state["name"]),
            base_value=state.get("base_value", 1.0),
            need_effects=state.get("need_effects", {}),
            driven_by_needs=state.get("driven_by_needs", []),
            opportunity_triggers=state.get("opportunity_triggers", []),
            duration_seconds=state.get("duration_seconds", 5.0),
            interruptible=state.get("interruptible", True),
            cooldown_seconds=state.get("cooldown_seconds", 0.0),
            energy_cost=state.get("energy_cost", 0.0),
            tags=state.get("tags", []),
            time_preferences=state.get("time_preferences", {}),
            spontaneous_probability=state.get("spontaneous_probability", 0.0),
        )

    def __str__(self) -> str:
        effects = ", ".join(f"{k}:{v:+.0f}" for k, v in self.need_effects.items())
        return f"Behavior({self.name}, base={self.base_value}, effects=[{effects}])"

    def summary(self) -> dict[str, Any]:
        """Get a summary for logging/debugging."""
        return {
            "name": self.name,
            "base_value": self.base_value,
            "driven_by": self.driven_by_needs,
            "triggers": self.opportunity_triggers,
            "duration": self.duration_seconds,
            "tags": self.tags,
        }
