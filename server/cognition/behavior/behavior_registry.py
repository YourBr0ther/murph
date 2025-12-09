"""
Murph - Behavior Registry
Central registry for all available behaviors.
"""

from typing import Any

from .behavior import Behavior


# Default behaviors that match the satisfaction_behaviors in needs_system.py
DEFAULT_BEHAVIORS: list[Behavior] = [
    # Energy behaviors
    Behavior(
        name="rest",
        display_name="Rest",
        base_value=1.0,
        need_effects={"energy": 20.0},
        driven_by_needs=["energy"],
        duration_seconds=30.0,
        interruptible=True,
        energy_cost=0.0,
        tags=["passive", "recovery"],
    ),
    Behavior(
        name="sleep",
        display_name="Sleep",
        base_value=0.9,
        need_effects={"energy": 50.0, "comfort": 10.0},
        driven_by_needs=["energy"],
        duration_seconds=120.0,
        interruptible=False,
        energy_cost=0.0,
        tags=["passive", "recovery"],
    ),
    # Curiosity behaviors
    Behavior(
        name="explore",
        display_name="Explore",
        base_value=1.1,
        need_effects={"curiosity": 15.0, "energy": -10.0},
        driven_by_needs=["curiosity"],
        opportunity_triggers=["unknown_object"],
        duration_seconds=20.0,
        energy_cost=10.0,
        tags=["active", "movement"],
    ),
    Behavior(
        name="investigate",
        display_name="Investigate Object",
        base_value=1.2,
        need_effects={"curiosity": 20.0, "energy": -5.0},
        driven_by_needs=["curiosity"],
        opportunity_triggers=["unknown_object"],
        duration_seconds=10.0,
        energy_cost=5.0,
        tags=["active", "focused"],
    ),
    Behavior(
        name="observe",
        display_name="Observe",
        base_value=0.9,
        need_effects={"curiosity": 10.0, "energy": -2.0},
        driven_by_needs=["curiosity"],
        opportunity_triggers=["has_objects"],
        duration_seconds=8.0,
        energy_cost=2.0,
        tags=["passive", "focused"],
    ),
    Behavior(
        name="wander",
        display_name="Wander",
        base_value=0.7,
        need_effects={"curiosity": 5.0, "energy": -3.0},
        driven_by_needs=["curiosity"],
        duration_seconds=10.0,
        energy_cost=3.0,
        tags=["idle", "movement"],
    ),
    # Play behaviors
    Behavior(
        name="play",
        display_name="Play",
        base_value=1.2,
        need_effects={"play": 20.0, "energy": -15.0},
        driven_by_needs=["play"],
        duration_seconds=15.0,
        energy_cost=15.0,
        tags=["active", "fun"],
    ),
    Behavior(
        name="chase",
        display_name="Chase",
        base_value=1.1,
        need_effects={"play": 18.0, "energy": -20.0, "curiosity": 5.0},
        driven_by_needs=["play"],
        opportunity_triggers=["has_objects"],
        duration_seconds=12.0,
        energy_cost=20.0,
        tags=["active", "fun", "movement"],
    ),
    Behavior(
        name="bounce",
        display_name="Bounce Around",
        base_value=1.0,
        need_effects={"play": 15.0, "energy": -20.0},
        driven_by_needs=["play"],
        duration_seconds=10.0,
        energy_cost=20.0,
        tags=["active", "fun", "movement"],
    ),
    Behavior(
        name="pounce",
        display_name="Pounce",
        base_value=1.0,
        need_effects={"play": 12.0, "energy": -10.0},
        driven_by_needs=["play"],
        opportunity_triggers=["has_objects"],
        duration_seconds=5.0,
        cooldown_seconds=10.0,
        energy_cost=10.0,
        tags=["active", "fun"],
    ),
    # Social behaviors
    Behavior(
        name="greet",
        display_name="Greet",
        base_value=1.3,
        need_effects={"social": 25.0, "affection": 10.0},
        driven_by_needs=["social"],
        opportunity_triggers=["person_nearby", "familiar_person"],
        duration_seconds=5.0,
        energy_cost=5.0,
        tags=["social", "active"],
    ),
    Behavior(
        name="follow",
        display_name="Follow Person",
        base_value=1.0,
        need_effects={"social": 15.0, "curiosity": 5.0, "energy": -15.0},
        driven_by_needs=["social"],
        opportunity_triggers=["person_nearby"],
        duration_seconds=30.0,
        interruptible=True,
        energy_cost=15.0,
        tags=["social", "movement"],
    ),
    Behavior(
        name="interact",
        display_name="Interact",
        base_value=1.1,
        need_effects={"social": 20.0, "play": 10.0},
        driven_by_needs=["social"],
        opportunity_triggers=["person_nearby"],
        duration_seconds=15.0,
        energy_cost=8.0,
        tags=["social", "active"],
    ),
    Behavior(
        name="approach",
        display_name="Approach Person",
        base_value=1.0,
        need_effects={"social": 10.0},
        driven_by_needs=["social"],
        opportunity_triggers=["person_detected", "person_far"],
        duration_seconds=8.0,
        energy_cost=5.0,
        tags=["social", "movement"],
    ),
    # Affection behaviors
    Behavior(
        name="nuzzle",
        display_name="Nuzzle",
        base_value=1.4,
        need_effects={"affection": 30.0, "social": 15.0},
        driven_by_needs=["affection"],
        opportunity_triggers=["person_nearby", "familiar_person"],
        duration_seconds=8.0,
        energy_cost=3.0,
        tags=["social", "affection"],
    ),
    Behavior(
        name="be_petted",
        display_name="Be Petted",
        base_value=1.5,
        need_effects={"affection": 35.0, "comfort": 15.0, "social": 10.0},
        driven_by_needs=["affection"],
        opportunity_triggers=["being_petted"],
        duration_seconds=15.0,
        interruptible=False,
        energy_cost=0.0,
        tags=["passive", "affection"],
    ),
    Behavior(
        name="cuddle",
        display_name="Cuddle",
        base_value=1.3,
        need_effects={"affection": 25.0, "comfort": 20.0, "social": 15.0},
        driven_by_needs=["affection"],
        opportunity_triggers=["being_held", "familiar_person"],
        duration_seconds=30.0,
        interruptible=True,
        energy_cost=0.0,
        tags=["passive", "affection"],
    ),
    Behavior(
        name="request_attention",
        display_name="Request Attention",
        base_value=1.1,
        need_effects={"affection": 10.0, "social": 10.0},
        driven_by_needs=["affection", "social"],
        opportunity_triggers=["person_nearby"],
        duration_seconds=5.0,
        cooldown_seconds=30.0,
        energy_cost=5.0,
        tags=["social", "affection"],
    ),
    # Comfort behaviors
    Behavior(
        name="find_cozy_spot",
        display_name="Find Cozy Spot",
        base_value=0.9,
        need_effects={"comfort": 20.0, "energy": -5.0},
        driven_by_needs=["comfort"],
        duration_seconds=15.0,
        energy_cost=5.0,
        tags=["passive", "movement"],
    ),
    Behavior(
        name="settle",
        display_name="Settle Down",
        base_value=0.8,
        need_effects={"comfort": 15.0, "energy": 5.0},
        driven_by_needs=["comfort"],
        duration_seconds=10.0,
        energy_cost=0.0,
        tags=["passive"],
    ),
    Behavior(
        name="adjust_position",
        display_name="Adjust Position",
        base_value=0.7,
        need_effects={"comfort": 8.0},
        driven_by_needs=["comfort"],
        duration_seconds=3.0,
        energy_cost=1.0,
        tags=["passive"],
    ),
    # Safety behaviors (high priority when triggered)
    Behavior(
        name="retreat",
        display_name="Retreat",
        base_value=2.0,
        need_effects={"safety": 30.0},
        driven_by_needs=["safety"],
        opportunity_triggers=["near_edge", "stranger"],
        duration_seconds=3.0,
        interruptible=False,
        energy_cost=5.0,
        tags=["safety", "movement"],
    ),
    Behavior(
        name="hide",
        display_name="Hide",
        base_value=1.8,
        need_effects={"safety": 25.0, "comfort": 5.0},
        driven_by_needs=["safety"],
        opportunity_triggers=["stranger", "loud_environment"],
        duration_seconds=20.0,
        interruptible=True,
        energy_cost=3.0,
        tags=["safety", "passive"],
    ),
    Behavior(
        name="approach_trusted",
        display_name="Approach Trusted Person",
        base_value=1.5,
        need_effects={"safety": 20.0, "social": 10.0},
        driven_by_needs=["safety"],
        opportunity_triggers=["familiar_person"],
        duration_seconds=8.0,
        energy_cost=5.0,
        tags=["safety", "social", "movement"],
    ),
    Behavior(
        name="scan",
        display_name="Scan Environment",
        base_value=1.5,
        need_effects={"safety": 15.0, "curiosity": 5.0},
        driven_by_needs=["safety"],
        duration_seconds=5.0,
        energy_cost=2.0,
        tags=["safety", "passive"],
    ),
    # Loneliness behaviors
    Behavior(
        name="sigh",
        display_name="Sigh",
        base_value=0.8,
        need_effects={"social": -5.0, "affection": -3.0},
        driven_by_needs=["social", "affection"],
        opportunity_triggers=["lonely"],
        duration_seconds=4.0,
        interruptible=True,
        cooldown_seconds=60.0,
        energy_cost=1.0,
        tags=["loneliness", "passive", "expressive"],
    ),
    Behavior(
        name="mope",
        display_name="Mope Around",
        base_value=0.7,
        need_effects={"social": -2.0, "energy": -3.0},
        driven_by_needs=["social", "affection"],
        opportunity_triggers=["very_lonely"],
        duration_seconds=10.0,
        interruptible=True,
        cooldown_seconds=120.0,
        energy_cost=3.0,
        tags=["loneliness", "passive", "movement"],
    ),
    Behavior(
        name="perk_up_hopeful",
        display_name="Perk Up Hopeful",
        base_value=0.9,
        need_effects={"curiosity": 5.0},
        driven_by_needs=["social", "affection"],
        opportunity_triggers=["lonely"],
        duration_seconds=6.0,
        interruptible=True,
        cooldown_seconds=90.0,
        energy_cost=2.0,
        tags=["loneliness", "active", "expressive"],
    ),
    Behavior(
        name="seek_company",
        display_name="Seek Company",
        base_value=1.0,
        need_effects={"social": 5.0, "energy": -8.0},
        driven_by_needs=["social", "affection"],
        opportunity_triggers=["lonely", "very_lonely"],
        duration_seconds=15.0,
        interruptible=True,
        cooldown_seconds=180.0,
        energy_cost=8.0,
        tags=["loneliness", "active", "movement"],
    ),
    # Idle/fallback behaviors
    Behavior(
        name="idle",
        display_name="Idle",
        base_value=0.5,
        need_effects={"energy": 2.0},
        driven_by_needs=[],
        duration_seconds=5.0,
        energy_cost=0.0,
        tags=["idle", "passive"],
    ),
]


class BehaviorRegistry:
    """
    Central registry for all available behaviors.

    Provides behavior lookup, filtering, and registration.
    """

    def __init__(self, load_defaults: bool = True) -> None:
        """
        Initialize the behavior registry.

        Args:
            load_defaults: Whether to load default behaviors on init
        """
        self._behaviors: dict[str, Behavior] = {}
        self._behaviors_by_tag: dict[str, list[str]] = {}

        if load_defaults:
            self._load_default_behaviors()

    def _load_default_behaviors(self) -> None:
        """Load the core behavior set."""
        for behavior in DEFAULT_BEHAVIORS:
            self.register(behavior)

    def register(self, behavior: Behavior) -> None:
        """
        Register a behavior.

        Args:
            behavior: The behavior to register
        """
        self._behaviors[behavior.name] = behavior

        # Index by tags
        for tag in behavior.tags:
            if tag not in self._behaviors_by_tag:
                self._behaviors_by_tag[tag] = []
            if behavior.name not in self._behaviors_by_tag[tag]:
                self._behaviors_by_tag[tag].append(behavior.name)

    def unregister(self, name: str) -> bool:
        """
        Unregister a behavior.

        Args:
            name: The behavior name to remove

        Returns:
            True if behavior was removed, False if not found
        """
        if name not in self._behaviors:
            return False

        behavior = self._behaviors.pop(name)

        # Remove from tag index
        for tag in behavior.tags:
            if tag in self._behaviors_by_tag:
                if name in self._behaviors_by_tag[tag]:
                    self._behaviors_by_tag[tag].remove(name)

        return True

    def get(self, name: str) -> Behavior | None:
        """Get behavior by name."""
        return self._behaviors.get(name)

    def get_all(self) -> list[Behavior]:
        """Get all registered behaviors."""
        return list(self._behaviors.values())

    def get_names(self) -> list[str]:
        """Get all registered behavior names."""
        return list(self._behaviors.keys())

    def get_by_tag(self, tag: str) -> list[Behavior]:
        """Get behaviors with a specific tag."""
        names = self._behaviors_by_tag.get(tag, [])
        return [self._behaviors[name] for name in names if name in self._behaviors]

    def get_by_tags(self, tags: list[str], match_all: bool = False) -> list[Behavior]:
        """
        Get behaviors matching tags.

        Args:
            tags: Tags to match
            match_all: If True, behavior must have all tags. If False, any tag.

        Returns:
            List of matching behaviors
        """
        if not tags:
            return []

        if match_all:
            result = []
            for behavior in self._behaviors.values():
                if all(tag in behavior.tags for tag in tags):
                    result.append(behavior)
            return result
        else:
            seen = set()
            result = []
            for tag in tags:
                for behavior in self.get_by_tag(tag):
                    if behavior.name not in seen:
                        seen.add(behavior.name)
                        result.append(behavior)
            return result

    def get_for_need(self, need_name: str) -> list[Behavior]:
        """Get behaviors that satisfy a specific need."""
        return [
            behavior for behavior in self._behaviors.values()
            if behavior.satisfies_need(need_name)
        ]

    def get_driven_by_need(self, need_name: str) -> list[Behavior]:
        """Get behaviors driven by a specific need."""
        return [
            behavior for behavior in self._behaviors.values()
            if need_name in behavior.driven_by_needs
        ]

    def __len__(self) -> int:
        return len(self._behaviors)

    def __contains__(self, name: str) -> bool:
        return name in self._behaviors

    def get_state(self) -> dict[str, Any]:
        """Get serializable state (custom behaviors only)."""
        # Only save non-default behaviors
        default_names = {b.name for b in DEFAULT_BEHAVIORS}
        custom = [
            b.get_state() for name, b in self._behaviors.items()
            if name not in default_names
        ]
        return {"custom_behaviors": custom}

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "BehaviorRegistry":
        """Create a BehaviorRegistry from saved state."""
        registry = cls(load_defaults=True)

        # Add any custom behaviors
        for behavior_state in state.get("custom_behaviors", []):
            behavior = Behavior.from_state(behavior_state)
            registry.register(behavior)

        return registry

    def summary(self) -> dict[str, Any]:
        """Get a summary for logging/debugging."""
        return {
            "total_behaviors": len(self._behaviors),
            "tags": list(self._behaviors_by_tag.keys()),
            "behavior_names": list(self._behaviors.keys()),
        }
