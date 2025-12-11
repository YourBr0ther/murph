"""
Murph - Needs System
Manages all of Murph's needs and calculates overall happiness.
"""

from typing import Any
import time

from .need import Need
from .personality import Personality


# Default need configurations
DEFAULT_NEEDS: dict[str, dict[str, Any]] = {
    "energy": {
        "decay_rate": 0.5,  # Slow - lasts a while
        "critical_threshold": 20.0,
        "happiness_weight": 1.5,
        "satisfaction_behaviors": ["rest", "charge", "sleep"],
    },
    "curiosity": {
        "decay_rate": 1.5,  # Moderate - gets bored
        "critical_threshold": 30.0,
        "happiness_weight": 1.0,
        "satisfaction_behaviors": ["explore", "investigate", "observe", "wander"],
    },
    "play": {
        "decay_rate": 1.0,  # Moderate
        "critical_threshold": 25.0,
        "happiness_weight": 1.2,
        "satisfaction_behaviors": ["play", "chase", "bounce", "pounce"],
    },
    "social": {
        "decay_rate": 0.8,  # Moderate
        "critical_threshold": 20.0,
        "happiness_weight": 1.3,
        "satisfaction_behaviors": ["greet", "follow", "interact", "approach"],
    },
    "affection": {
        "decay_rate": 0.3,  # Slow
        "critical_threshold": 15.0,
        "happiness_weight": 1.4,
        "satisfaction_behaviors": ["nuzzle", "be_petted", "cuddle", "request_attention"],
    },
    "comfort": {
        "decay_rate": 0.2,  # Very slow
        "critical_threshold": 30.0,
        "happiness_weight": 0.8,
        "satisfaction_behaviors": ["find_cozy_spot", "settle", "adjust_position"],
    },
    "safety": {
        "decay_rate": 0.0,  # Event-driven, doesn't decay on its own
        "critical_threshold": 40.0,
        "happiness_weight": 2.0,  # Very important when low
        "satisfaction_behaviors": ["retreat", "hide", "approach_trusted", "scan"],
    },
}


class NeedsSystem:
    """
    Manages all of Murph's needs, inspired by The Sims motive system.

    Each need:
    - Ranges from 0 (critical) to 100 (satisfied)
    - Decays over time at a rate modified by personality
    - Can be satisfied by specific behaviors
    - Contributes to overall happiness with a weight
    """

    def __init__(self, personality: Personality | None = None) -> None:
        """
        Initialize the needs system.

        Args:
            personality: Personality traits that modify need behavior.
                        If None, uses balanced personality.
        """
        self.personality = personality or Personality()
        self.needs: dict[str, Need] = {}
        self._last_update_time: float = time.time()

        # Initialize all default needs
        for name, config in DEFAULT_NEEDS.items():
            self.needs[name] = Need(
                name=name,
                value=100.0,  # Start fully satisfied
                decay_rate=config["decay_rate"],
                critical_threshold=config["critical_threshold"],
                happiness_weight=config["happiness_weight"],
                satisfaction_behaviors=config["satisfaction_behaviors"],
            )

    def update(self, delta_seconds: float | None = None) -> None:
        """
        Update all needs (decay over time).

        Args:
            delta_seconds: Time elapsed since last update.
                          If None, calculates from last update time.
        """
        if delta_seconds is None:
            current_time = time.time()
            delta_seconds = current_time - self._last_update_time
            self._last_update_time = current_time

        for need in self.needs.values():
            # Apply personality modifier to decay rate
            decay_modifier = self.personality.get_need_decay_modifier(need.name)
            effective_decay = need.decay_rate * decay_modifier

            # Temporarily modify decay rate for this update
            original_rate = need.decay_rate
            need.decay_rate = effective_decay
            need.decay(delta_seconds)
            need.decay_rate = original_rate

    def get_need(self, name: str) -> Need | None:
        """Get a specific need by name."""
        return self.needs.get(name)

    def satisfy_need(self, name: str, amount: float) -> bool:
        """
        Satisfy a specific need.

        Args:
            name: Name of the need
            amount: How much to satisfy (positive value)

        Returns:
            True if need exists and was satisfied, False otherwise
        """
        need = self.needs.get(name)
        if need:
            need.satisfy(amount)
            return True
        return False

    def deplete_need(self, name: str, amount: float) -> bool:
        """
        Deplete a specific need (for events that drain needs).

        Args:
            name: Name of the need
            amount: How much to deplete (positive value)

        Returns:
            True if need exists, False otherwise
        """
        need = self.needs.get(name)
        if need:
            need.deplete(amount)
            return True
        return False

    def calculate_happiness(self) -> float:
        """
        Calculate overall happiness as a weighted average of all needs.

        Returns:
            Happiness value from 0 (miserable) to 100 (perfectly happy)
        """
        if not self.needs:
            return 100.0

        total_weighted_value = 0.0
        total_weight = 0.0

        for need in self.needs.values():
            # Apply personality importance modifier
            importance_mod = self.personality.get_need_importance_modifier(need.name)
            effective_weight = need.happiness_weight * importance_mod

            total_weighted_value += need.value * effective_weight
            total_weight += effective_weight

        if total_weight == 0:
            return 100.0

        return total_weighted_value / total_weight

    def get_happiness(self) -> float:
        """
        Get the robot's current happiness level.

        Returns:
            Happiness value from 0 (miserable) to 100 (perfectly happy)
        """
        return self.calculate_happiness()

    def get_critical_needs(self) -> list[Need]:
        """Get list of needs that are below their critical threshold."""
        return [need for need in self.needs.values() if need.is_critical()]

    def get_most_urgent_need(self) -> Need | None:
        """Get the need with the highest urgency (lowest value weighted by importance)."""
        if not self.needs:
            return None

        most_urgent = None
        highest_urgency = -1.0

        for need in self.needs.values():
            importance_mod = self.personality.get_need_importance_modifier(need.name)
            urgency = need.weighted_urgency() * importance_mod

            if urgency > highest_urgency:
                highest_urgency = urgency
                most_urgent = need

        return most_urgent

    def get_behaviors_for_need(self, need_name: str) -> list[str]:
        """Get list of behaviors that can satisfy a specific need."""
        need = self.needs.get(need_name)
        if need:
            return need.satisfaction_behaviors
        return []

    def get_suggested_behaviors(self, count: int = 3) -> list[tuple[str, str, float]]:
        """
        Get suggested behaviors based on current need states.

        Returns:
            List of (behavior_name, need_name, urgency_score) tuples
            sorted by urgency (most urgent first)
        """
        suggestions = []

        for need in self.needs.values():
            importance_mod = self.personality.get_need_importance_modifier(need.name)
            urgency = need.weighted_urgency() * importance_mod

            for behavior in need.satisfaction_behaviors:
                # Apply personality modifier to behavior preference
                behavior_mod = self.personality.get_behavior_modifier(behavior)
                final_score = urgency * behavior_mod
                suggestions.append((behavior, need.name, final_score))

        # Sort by urgency score (highest first) and take top N
        suggestions.sort(key=lambda x: x[2], reverse=True)
        return suggestions[:count]

    def get_mood(self) -> str:
        """
        Get a simple mood description based on current state.

        Returns:
            One of: "happy", "content", "neutral", "uneasy", "distressed"
        """
        happiness = self.calculate_happiness()
        critical_count = len(self.get_critical_needs())

        if critical_count >= 3:
            return "distressed"
        elif critical_count >= 1:
            return "uneasy"
        elif happiness >= 80:
            return "happy"
        elif happiness >= 60:
            return "content"
        else:
            return "neutral"

    def get_state(self) -> dict[str, Any]:
        """Get serializable state for persistence."""
        return {
            "needs": {name: need.get_state() for name, need in self.needs.items()},
            "personality": self.personality.get_state(),
            "last_update_time": self._last_update_time,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "NeedsSystem":
        """Create a NeedsSystem from saved state."""
        personality = Personality.from_state(state.get("personality", {}))
        system = cls(personality=personality)

        # Restore need values
        needs_state = state.get("needs", {})
        for name, need_state in needs_state.items():
            if name in system.needs:
                system.needs[name].value = need_state.get("value", 100.0)

        system._last_update_time = state.get("last_update_time", time.time())
        return system

    def __str__(self) -> str:
        lines = [f"NeedsSystem (Happiness: {self.calculate_happiness():.1f}, Mood: {self.get_mood()})"]
        for need in self.needs.values():
            lines.append(f"  {need}")
        return "\n".join(lines)

    def summary(self) -> dict[str, Any]:
        """Get a summary of current state for logging/debugging."""
        return {
            "happiness": round(self.calculate_happiness(), 1),
            "mood": self.get_mood(),
            "critical_needs": [n.name for n in self.get_critical_needs()],
            "most_urgent": self.get_most_urgent_need().name if self.get_most_urgent_need() else None,
            "needs": {name: round(need.value, 1) for name, need in self.needs.items()},
        }
