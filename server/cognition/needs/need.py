"""
Murph - Need Base Class
Represents a single need (like energy, play, social) that decays over time.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Need:
    """
    A single need that decays over time and can be satisfied by behaviors.

    Inspired by The Sims motive system - needs range from 0 (critical) to 100 (satisfied)
    and decay at different rates. When a need drops below its critical threshold,
    it becomes urgent and should drive behavior selection.
    """

    name: str
    value: float = 100.0  # Current value (0-100)
    decay_rate: float = 1.0  # Points lost per minute
    critical_threshold: float = 20.0  # Below this = urgent
    happiness_weight: float = 1.0  # Weight in happiness calculation
    satisfaction_behaviors: list[str] = field(default_factory=list)

    # Bounds
    MIN_VALUE: float = field(default=0.0, init=False, repr=False)
    MAX_VALUE: float = field(default=100.0, init=False, repr=False)

    def decay(self, delta_seconds: float) -> None:
        """
        Decay the need over time.

        Args:
            delta_seconds: Time elapsed in seconds
        """
        if self.decay_rate <= 0:
            return  # Safety need doesn't decay on its own

        # Convert decay rate from per-minute to per-second
        decay_amount = (self.decay_rate / 60.0) * delta_seconds
        self.value = max(self.MIN_VALUE, self.value - decay_amount)

    def satisfy(self, amount: float) -> None:
        """
        Increase the need value (satisfy the need).

        Args:
            amount: How much to increase the need (positive value)
        """
        self.value = min(self.MAX_VALUE, self.value + abs(amount))

    def deplete(self, amount: float) -> None:
        """
        Decrease the need value (deplete the need).

        Args:
            amount: How much to decrease the need (positive value)
        """
        self.value = max(self.MIN_VALUE, self.value - abs(amount))

    def is_critical(self) -> bool:
        """Check if the need is below its critical threshold."""
        return self.value < self.critical_threshold

    def is_satisfied(self, threshold: float = 80.0) -> bool:
        """Check if the need is well satisfied (above threshold)."""
        return self.value >= threshold

    def urgency(self) -> float:
        """
        Calculate how urgent this need is (0-1).

        Returns higher values when the need is lower.
        Used by the utility AI to prioritize behaviors.
        """
        # Linear urgency: 0 at max, 1 at min
        return 1.0 - (self.value / self.MAX_VALUE)

    def weighted_urgency(self) -> float:
        """
        Calculate urgency weighted by happiness weight.

        Needs with higher happiness_weight are more important to satisfy.
        """
        return self.urgency() * self.happiness_weight

    def get_state(self) -> dict[str, Any]:
        """Get serializable state for persistence."""
        return {
            "name": self.name,
            "value": self.value,
            "decay_rate": self.decay_rate,
            "critical_threshold": self.critical_threshold,
            "happiness_weight": self.happiness_weight,
            "satisfaction_behaviors": self.satisfaction_behaviors,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "Need":
        """Create a Need from saved state."""
        return cls(
            name=state["name"],
            value=state.get("value", 100.0),
            decay_rate=state.get("decay_rate", 1.0),
            critical_threshold=state.get("critical_threshold", 20.0),
            happiness_weight=state.get("happiness_weight", 1.0),
            satisfaction_behaviors=state.get("satisfaction_behaviors", []),
        )

    def __str__(self) -> str:
        status = "CRITICAL" if self.is_critical() else "ok"
        return f"{self.name}: {self.value:.1f}/100 [{status}]"
