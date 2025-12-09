"""
Murph - World Context
Current world state used for opportunity bonus calculation.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WorldContext:
    """
    Current world state used for opportunity bonus calculation.

    Updated each perception cycle with sensor data and scene analysis.
    The evaluator uses this to determine context-appropriate behaviors.

    Attributes:
        person_detected: Whether a person is currently in view
        person_is_familiar: Whether the detected person is recognized
        person_familiarity_score: Recognition confidence (0-100)
        person_distance: Distance to detected person in cm, None if not detected
        objects_in_view: List of recognized objects
        unknown_object_detected: Whether an unrecognized object is in view
        is_dark: Whether lighting is low
        is_loud: Whether ambient noise is high
        near_edge: Whether robot is near a table/surface edge
        near_charging_station: Whether charging station is nearby
        is_being_held: Whether robot is being held
        is_being_petted: Whether robot is being petted
        recent_bump: Whether a recent collision occurred
        time_since_last_interaction: Seconds since last human interaction
        current_behavior: Currently executing behavior name
        active_triggers: Explicitly set trigger conditions
    """

    # Person detection
    person_detected: bool = False
    person_is_familiar: bool = False
    person_familiarity_score: float = 0.0
    person_distance: float | None = None

    # Object detection
    objects_in_view: list[str] = field(default_factory=list)
    unknown_object_detected: bool = False

    # Environment
    is_dark: bool = False
    is_loud: bool = False
    near_edge: bool = False
    near_charging_station: bool = False

    # Physical state
    is_being_held: bool = False
    is_being_petted: bool = False
    recent_bump: bool = False

    # Time context
    time_since_last_interaction: float = 0.0
    current_behavior: str | None = None

    # Memory-derived state (populated by MemorySystem)
    remembered_person_name: str | None = None
    person_interaction_count: int = 0
    person_sentiment: float = 0.0
    recent_event_types: list[str] = field(default_factory=list)

    # Spatial awareness (populated by MemorySystem.spatial_map)
    current_zone_type: str | None = None
    current_zone_safety: float = 0.5
    near_known_landmark: bool = False
    landmark_type_nearby: str | None = None
    at_home_base: bool = False
    at_charging_station: bool = False
    position_confidence: float = 0.0  # 0 = lost, 1 = certain

    # Navigation state (populated by MemorySystem.spatial_map)
    has_unfamiliar_zones: bool = False
    has_path_to_home: bool = False
    has_path_to_charger: bool = False
    zone_familiarity: float = 1.0  # 0 = unfamiliar, 1 = well-explored

    # Custom triggers (extensible)
    active_triggers: set[str] = field(default_factory=set)

    def has_trigger(self, trigger: str) -> bool:
        """
        Check if a trigger is currently active.

        Supports both explicit triggers (in active_triggers set)
        and computed triggers based on context state.

        Args:
            trigger: The trigger name to check

        Returns:
            True if the trigger is active
        """
        # Check explicit triggers first
        if trigger in self.active_triggers:
            return True

        # Check computed triggers based on context state
        trigger_checks: dict[str, bool] = {
            # Person-related triggers
            "person_nearby": self.person_detected and (self.person_distance or 100) < 50,
            "person_far": self.person_detected and (self.person_distance or 0) > 100,
            "familiar_person": self.person_detected and self.person_is_familiar,
            "stranger": self.person_detected and not self.person_is_familiar,
            "person_detected": self.person_detected,
            # Object triggers
            "unknown_object": self.unknown_object_detected,
            "has_objects": len(self.objects_in_view) > 0,
            # Physical state triggers
            "being_held": self.is_being_held,
            "being_petted": self.is_being_petted,
            "recent_bump": self.recent_bump,
            # Environment triggers
            "low_light": self.is_dark,
            "loud_environment": self.is_loud,
            "near_edge": self.near_edge,
            "near_charger": self.near_charging_station,
            # Time-based triggers
            "lonely": self.time_since_last_interaction > 300,  # 5 minutes
            "very_lonely": self.time_since_last_interaction > 600,  # 10 minutes
            # Memory-derived triggers
            "familiar_person_remembered": (
                self.person_detected and self.remembered_person_name is not None
            ),
            "positive_history": self.person_interaction_count >= 5,
            "negative_sentiment": self.person_sentiment < -0.3,
            "positive_sentiment": self.person_sentiment > 0.3,
            "recently_greeted": "greeting" in self.recent_event_types,
            "recently_played": "play" in self.recent_event_types,
            "recently_petted": "petting" in self.recent_event_types,
            # Spatial awareness triggers
            "at_home": self.at_home_base,
            "at_charger": self.at_charging_station,
            "in_safe_zone": self.current_zone_safety > 0.7,
            "in_danger_zone": self.current_zone_safety < 0.3,
            "position_known": self.position_confidence > 0.5,
            "position_lost": self.position_confidence < 0.2,
            "near_landmark": self.near_known_landmark,
            # Navigation triggers
            "has_unfamiliar_zones": self.has_unfamiliar_zones,
            "has_path_home": self.has_path_to_home,
            "has_path_charger": self.has_path_to_charger,
            "in_unfamiliar_zone": self.zone_familiarity < 0.5,
            "in_familiar_zone": self.zone_familiarity >= 0.5,
        }

        return trigger_checks.get(trigger, False)

    def add_trigger(self, trigger: str) -> None:
        """Add an explicit trigger."""
        self.active_triggers.add(trigger)

    def remove_trigger(self, trigger: str) -> None:
        """Remove an explicit trigger."""
        self.active_triggers.discard(trigger)

    def clear_triggers(self) -> None:
        """Clear all explicit triggers."""
        self.active_triggers.clear()

    def get_active_triggers(self) -> list[str]:
        """Get list of all currently active triggers (explicit and computed)."""
        active = list(self.active_triggers)

        # Check all computed triggers
        computed_triggers = [
            "person_nearby", "person_far", "familiar_person", "stranger",
            "person_detected", "unknown_object", "has_objects", "being_held",
            "being_petted", "recent_bump", "low_light", "loud_environment",
            "near_edge", "near_charger", "lonely", "very_lonely",
            # Memory-derived triggers
            "familiar_person_remembered", "positive_history", "negative_sentiment",
            "positive_sentiment", "recently_greeted", "recently_played", "recently_petted",
            # Spatial awareness triggers
            "at_home", "at_charger", "in_safe_zone", "in_danger_zone",
            "position_known", "position_lost", "near_landmark",
            # Navigation triggers
            "has_unfamiliar_zones", "has_path_home", "has_path_charger",
            "in_unfamiliar_zone", "in_familiar_zone",
        ]

        for trigger in computed_triggers:
            if trigger not in active and self.has_trigger(trigger):
                active.append(trigger)

        return active

    def get_state(self) -> dict[str, Any]:
        """Get serializable state."""
        return {
            "person_detected": self.person_detected,
            "person_is_familiar": self.person_is_familiar,
            "person_familiarity_score": self.person_familiarity_score,
            "person_distance": self.person_distance,
            "objects_in_view": self.objects_in_view.copy(),
            "unknown_object_detected": self.unknown_object_detected,
            "is_dark": self.is_dark,
            "is_loud": self.is_loud,
            "near_edge": self.near_edge,
            "near_charging_station": self.near_charging_station,
            "is_being_held": self.is_being_held,
            "is_being_petted": self.is_being_petted,
            "recent_bump": self.recent_bump,
            "time_since_last_interaction": self.time_since_last_interaction,
            "current_behavior": self.current_behavior,
            "remembered_person_name": self.remembered_person_name,
            "person_interaction_count": self.person_interaction_count,
            "person_sentiment": self.person_sentiment,
            "recent_event_types": self.recent_event_types.copy(),
            "current_zone_type": self.current_zone_type,
            "current_zone_safety": self.current_zone_safety,
            "near_known_landmark": self.near_known_landmark,
            "landmark_type_nearby": self.landmark_type_nearby,
            "at_home_base": self.at_home_base,
            "at_charging_station": self.at_charging_station,
            "position_confidence": self.position_confidence,
            "has_unfamiliar_zones": self.has_unfamiliar_zones,
            "has_path_to_home": self.has_path_to_home,
            "has_path_to_charger": self.has_path_to_charger,
            "zone_familiarity": self.zone_familiarity,
            "active_triggers": list(self.active_triggers),
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "WorldContext":
        """Create a WorldContext from saved state."""
        context = cls(
            person_detected=state.get("person_detected", False),
            person_is_familiar=state.get("person_is_familiar", False),
            person_familiarity_score=state.get("person_familiarity_score", 0.0),
            person_distance=state.get("person_distance"),
            objects_in_view=state.get("objects_in_view", []),
            unknown_object_detected=state.get("unknown_object_detected", False),
            is_dark=state.get("is_dark", False),
            is_loud=state.get("is_loud", False),
            near_edge=state.get("near_edge", False),
            near_charging_station=state.get("near_charging_station", False),
            is_being_held=state.get("is_being_held", False),
            is_being_petted=state.get("is_being_petted", False),
            recent_bump=state.get("recent_bump", False),
            time_since_last_interaction=state.get("time_since_last_interaction", 0.0),
            current_behavior=state.get("current_behavior"),
            remembered_person_name=state.get("remembered_person_name"),
            person_interaction_count=state.get("person_interaction_count", 0),
            person_sentiment=state.get("person_sentiment", 0.0),
            recent_event_types=state.get("recent_event_types", []),
            current_zone_type=state.get("current_zone_type"),
            current_zone_safety=state.get("current_zone_safety", 0.5),
            near_known_landmark=state.get("near_known_landmark", False),
            landmark_type_nearby=state.get("landmark_type_nearby"),
            at_home_base=state.get("at_home_base", False),
            at_charging_station=state.get("at_charging_station", False),
            position_confidence=state.get("position_confidence", 0.0),
            has_unfamiliar_zones=state.get("has_unfamiliar_zones", False),
            has_path_to_home=state.get("has_path_to_home", False),
            has_path_to_charger=state.get("has_path_to_charger", False),
            zone_familiarity=state.get("zone_familiarity", 1.0),
        )
        context.active_triggers = set(state.get("active_triggers", []))
        return context

    def __str__(self) -> str:
        triggers = self.get_active_triggers()
        return f"WorldContext(triggers={triggers})"

    def summary(self) -> dict[str, Any]:
        """Get a summary for logging/debugging."""
        return {
            "person": {
                "detected": self.person_detected,
                "familiar": self.person_is_familiar,
                "distance": self.person_distance,
                "remembered_name": self.remembered_person_name,
                "interaction_count": self.person_interaction_count,
                "sentiment": self.person_sentiment,
            },
            "environment": {
                "dark": self.is_dark,
                "loud": self.is_loud,
                "near_edge": self.near_edge,
            },
            "physical": {
                "held": self.is_being_held,
                "petted": self.is_being_petted,
                "bump": self.recent_bump,
            },
            "memory": {
                "recent_events": self.recent_event_types,
            },
            "spatial": {
                "zone_type": self.current_zone_type,
                "zone_safety": self.current_zone_safety,
                "zone_familiarity": self.zone_familiarity,
                "at_home": self.at_home_base,
                "near_landmark": self.near_known_landmark,
                "landmark_type": self.landmark_type_nearby,
                "position_confidence": self.position_confidence,
                "has_unfamiliar_zones": self.has_unfamiliar_zones,
                "has_path_home": self.has_path_to_home,
                "has_path_charger": self.has_path_to_charger,
            },
            "active_triggers": self.get_active_triggers(),
        }
