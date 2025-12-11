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
    # Navigation behaviors (use spatial map for intelligent movement)
    Behavior(
        name="go_home",
        display_name="Go Home",
        base_value=1.3,
        need_effects={"comfort": 15.0, "safety": 10.0},
        driven_by_needs=["comfort", "safety"],
        opportunity_triggers=["position_known", "has_path_home"],
        duration_seconds=60.0,
        interruptible=True,
        energy_cost=15.0,
        tags=["navigation", "goal", "movement"],
    ),
    Behavior(
        name="go_to_charger",
        display_name="Go To Charger",
        base_value=1.8,
        need_effects={"energy": 50.0},
        driven_by_needs=["energy"],
        opportunity_triggers=["position_known", "has_path_charger"],
        duration_seconds=60.0,
        interruptible=False,
        energy_cost=10.0,
        tags=["navigation", "goal", "movement", "charging"],
    ),
    Behavior(
        name="go_to_landmark",
        display_name="Go To Landmark",
        base_value=1.0,
        need_effects={"curiosity": 10.0},
        driven_by_needs=["curiosity"],
        opportunity_triggers=["position_known", "near_landmark"],
        duration_seconds=45.0,
        interruptible=True,
        energy_cost=12.0,
        tags=["navigation", "goal", "movement"],
    ),
    Behavior(
        name="explore_unfamiliar",
        display_name="Explore Unfamiliar Area",
        base_value=1.2,
        need_effects={"curiosity": 25.0, "energy": -15.0},
        driven_by_needs=["curiosity"],
        opportunity_triggers=["position_known", "has_unfamiliar_zones"],
        duration_seconds=45.0,
        interruptible=True,
        energy_cost=15.0,
        tags=["navigation", "exploration", "movement"],
    ),
    Behavior(
        name="patrol",
        display_name="Patrol",
        base_value=0.9,
        need_effects={"curiosity": 10.0, "safety": 5.0, "energy": -10.0},
        driven_by_needs=["curiosity", "safety"],
        opportunity_triggers=["position_known", "in_safe_zone"],
        duration_seconds=60.0,
        interruptible=True,
        energy_cost=10.0,
        tags=["navigation", "exploration", "movement"],
    ),
    Behavior(
        name="flee_danger",
        display_name="Flee Danger",
        base_value=2.5,
        need_effects={"safety": 40.0},
        driven_by_needs=["safety"],
        opportunity_triggers=["in_danger_zone"],
        duration_seconds=10.0,
        interruptible=False,
        cooldown_seconds=60.0,
        energy_cost=20.0,
        tags=["navigation", "safety", "movement", "urgent"],
    ),
    Behavior(
        name="retreat_to_safe",
        display_name="Retreat To Safe Zone",
        base_value=2.0,
        need_effects={"safety": 30.0, "comfort": 10.0},
        driven_by_needs=["safety"],
        opportunity_triggers=["in_danger_zone", "near_edge"],
        duration_seconds=20.0,
        interruptible=True,
        energy_cost=15.0,
        tags=["navigation", "safety", "movement"],
    ),
    Behavior(
        name="reorient",
        display_name="Reorient",
        base_value=1.5,
        need_effects={"safety": 15.0},
        driven_by_needs=["safety"],
        opportunity_triggers=["position_lost"],
        duration_seconds=30.0,
        interruptible=True,
        energy_cost=8.0,
        tags=["navigation", "recovery", "scanning"],
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
    # ========================================
    # Time-Based Routine Behaviors
    # ========================================
    Behavior(
        name="wake_up",
        display_name="Wake Up",
        base_value=1.0,
        need_effects={"energy": 5.0, "curiosity": 10.0},
        driven_by_needs=["energy"],
        opportunity_triggers=["is_morning"],
        duration_seconds=8.0,
        cooldown_seconds=3600.0,  # Once per hour max
        energy_cost=0.0,
        tags=["routine", "morning", "active"],
        time_preferences={"morning": 1.5, "midday": 0.5, "evening": 0.5, "night": 0.3},
    ),
    Behavior(
        name="morning_stretch",
        display_name="Morning Stretch",
        base_value=1.1,
        need_effects={"comfort": 15.0, "energy": 5.0},
        driven_by_needs=["comfort"],
        opportunity_triggers=["is_morning"],
        duration_seconds=6.0,
        cooldown_seconds=1800.0,  # 30 min cooldown
        energy_cost=2.0,
        tags=["routine", "morning", "expressive"],
        time_preferences={"morning": 1.5, "midday": 0.8, "evening": 0.6, "night": 0.4},
    ),
    Behavior(
        name="energetic_start",
        display_name="Energetic Start",
        base_value=1.2,
        need_effects={"play": 10.0, "curiosity": 10.0, "energy": -10.0},
        driven_by_needs=["play", "curiosity"],
        opportunity_triggers=["is_morning"],
        duration_seconds=12.0,
        cooldown_seconds=3600.0,
        energy_cost=10.0,
        tags=["routine", "morning", "active", "movement"],
        time_preferences={"morning": 1.5, "midday": 1.0, "evening": 0.5, "night": 0.3},
    ),
    Behavior(
        name="midday_activity",
        display_name="Midday Activity",
        base_value=1.1,
        need_effects={"play": 15.0, "curiosity": 10.0, "energy": -12.0},
        driven_by_needs=["play", "curiosity"],
        opportunity_triggers=["is_midday"],
        duration_seconds=15.0,
        cooldown_seconds=1800.0,
        energy_cost=12.0,
        tags=["routine", "midday", "active", "movement"],
        time_preferences={"morning": 0.8, "midday": 1.4, "evening": 0.7, "night": 0.4},
    ),
    Behavior(
        name="afternoon_rest",
        display_name="Afternoon Rest",
        base_value=0.9,
        need_effects={"energy": 10.0, "comfort": 5.0},
        driven_by_needs=["energy", "comfort"],
        opportunity_triggers=["is_midday", "is_evening"],
        duration_seconds=20.0,
        cooldown_seconds=1800.0,
        energy_cost=0.0,
        tags=["routine", "midday", "evening", "passive"],
        time_preferences={"morning": 0.6, "midday": 1.2, "evening": 1.3, "night": 0.8},
    ),
    Behavior(
        name="evening_settle",
        display_name="Evening Settle",
        base_value=1.0,
        need_effects={"comfort": 20.0, "energy": 5.0},
        driven_by_needs=["comfort"],
        opportunity_triggers=["is_evening"],
        duration_seconds=15.0,
        cooldown_seconds=3600.0,
        energy_cost=0.0,
        tags=["routine", "evening", "passive"],
        time_preferences={"morning": 0.4, "midday": 0.6, "evening": 1.5, "night": 1.2},
    ),
    Behavior(
        name="pre_sleep_yawn",
        display_name="Pre-Sleep Yawn",
        base_value=0.9,
        need_effects={"energy": 3.0, "comfort": 5.0},
        driven_by_needs=["energy"],
        opportunity_triggers=["is_evening", "is_night"],
        duration_seconds=4.0,
        cooldown_seconds=300.0,  # 5 min cooldown
        energy_cost=0.0,
        tags=["routine", "evening", "night", "expressive"],
        time_preferences={"morning": 0.3, "midday": 0.5, "evening": 1.3, "night": 1.5},
    ),
    Behavior(
        name="night_stir",
        display_name="Night Stir",
        base_value=0.6,
        need_effects={"comfort": 5.0},
        driven_by_needs=["comfort"],
        opportunity_triggers=["is_night"],
        duration_seconds=5.0,
        cooldown_seconds=600.0,  # 10 min cooldown
        energy_cost=1.0,
        tags=["routine", "night", "passive"],
        time_preferences={"morning": 0.3, "midday": 0.3, "evening": 0.5, "night": 1.5},
    ),
    # ========================================
    # Personality Expression Behaviors
    # ========================================
    Behavior(
        name="stretch",
        display_name="Stretch",
        base_value=0.8,
        need_effects={"comfort": 10.0, "energy": 3.0},
        driven_by_needs=["comfort"],
        duration_seconds=5.0,
        cooldown_seconds=120.0,  # 2 min cooldown
        energy_cost=1.0,
        tags=["expressive", "passive"],
        spontaneous_probability=0.05,
    ),
    Behavior(
        name="yawn",
        display_name="Yawn",
        base_value=0.7,
        need_effects={"energy": 2.0},
        driven_by_needs=["energy"],
        duration_seconds=3.0,
        cooldown_seconds=180.0,  # 3 min cooldown
        energy_cost=0.0,
        tags=["expressive", "passive"],
        spontaneous_probability=0.04,
    ),
    Behavior(
        name="daydream",
        display_name="Daydream",
        base_value=0.6,
        need_effects={"curiosity": 5.0},
        driven_by_needs=["curiosity"],
        duration_seconds=8.0,
        cooldown_seconds=300.0,  # 5 min cooldown
        energy_cost=0.0,
        tags=["expressive", "idle", "passive"],
        spontaneous_probability=0.03,
    ),
    Behavior(
        name="shake_off",
        display_name="Shake Off",
        base_value=0.9,
        need_effects={"comfort": 8.0},
        driven_by_needs=["comfort"],
        duration_seconds=3.0,
        cooldown_seconds=240.0,  # 4 min cooldown
        energy_cost=3.0,
        tags=["expressive", "active"],
        spontaneous_probability=0.02,
    ),
    Behavior(
        name="sneeze",
        display_name="Sneeze",
        base_value=0.5,
        need_effects={},  # No effect, just cute
        driven_by_needs=[],
        duration_seconds=2.0,
        cooldown_seconds=600.0,  # 10 min cooldown
        energy_cost=0.0,
        tags=["expressive", "passive"],
        spontaneous_probability=0.01,
    ),
    Behavior(
        name="happy_wiggle",
        display_name="Happy Wiggle",
        base_value=1.0,
        need_effects={"play": 8.0, "social": 5.0},
        driven_by_needs=["play", "social"],
        opportunity_triggers=["person_nearby", "familiar_person"],
        duration_seconds=4.0,
        cooldown_seconds=60.0,
        energy_cost=3.0,
        tags=["expressive", "active", "social"],
        spontaneous_probability=0.03,
    ),
    Behavior(
        name="curious_tilt",
        display_name="Curious Tilt",
        base_value=0.8,
        need_effects={"curiosity": 5.0},
        driven_by_needs=["curiosity"],
        opportunity_triggers=["unknown_object", "heard_speech"],
        duration_seconds=3.0,
        cooldown_seconds=30.0,
        energy_cost=1.0,
        tags=["expressive", "passive"],
        spontaneous_probability=0.04,
    ),
    Behavior(
        name="contented_sigh",
        display_name="Contented Sigh",
        base_value=0.7,
        need_effects={"comfort": 5.0, "affection": 3.0},
        driven_by_needs=["comfort", "affection"],
        opportunity_triggers=["being_petted", "familiar_person"],
        duration_seconds=3.0,
        cooldown_seconds=120.0,
        energy_cost=0.0,
        tags=["expressive", "passive", "affection"],
        spontaneous_probability=0.03,
    ),
    # ========================================
    # Reactive Behaviors
    # ========================================
    Behavior(
        name="dropped_recovery",
        display_name="Dropped Recovery",
        base_value=2.0,
        need_effects={"safety": 20.0, "comfort": -10.0},
        driven_by_needs=["safety"],
        opportunity_triggers=["dropped", "recent_bump"],
        duration_seconds=5.0,
        interruptible=False,
        cooldown_seconds=30.0,
        energy_cost=5.0,
        tags=["reactive", "safety"],
    ),
    Behavior(
        name="loud_noise_reaction",
        display_name="Loud Noise Reaction",
        base_value=1.8,
        need_effects={"safety": 15.0},
        driven_by_needs=["safety"],
        opportunity_triggers=["loud_environment"],
        duration_seconds=4.0,
        interruptible=False,
        cooldown_seconds=20.0,
        energy_cost=3.0,
        tags=["reactive", "safety"],
    ),
    Behavior(
        name="new_object_investigation",
        display_name="New Object Investigation",
        base_value=1.5,
        need_effects={"curiosity": 15.0, "safety": -5.0},
        driven_by_needs=["curiosity"],
        opportunity_triggers=["unknown_object"],
        duration_seconds=8.0,
        interruptible=True,
        cooldown_seconds=60.0,
        energy_cost=5.0,
        tags=["reactive", "curiosity", "active"],
    ),
    Behavior(
        name="person_left_sad",
        display_name="Person Left Sad",
        base_value=1.3,
        need_effects={"social": -10.0, "affection": -5.0},
        driven_by_needs=["social", "affection"],
        opportunity_triggers=["person_left"],
        duration_seconds=6.0,
        interruptible=True,
        cooldown_seconds=120.0,
        energy_cost=0.0,
        tags=["reactive", "loneliness", "expressive"],
    ),
    Behavior(
        name="touched_unexpectedly",
        display_name="Touched Unexpectedly",
        base_value=1.6,
        need_effects={"safety": 10.0},
        driven_by_needs=["safety"],
        opportunity_triggers=["touched_unexpected"],
        duration_seconds=3.0,
        interruptible=False,
        cooldown_seconds=15.0,
        energy_cost=2.0,
        tags=["reactive", "safety"],
    ),
    Behavior(
        name="picked_up_happy",
        display_name="Picked Up Happy",
        base_value=1.4,
        need_effects={"social": 10.0, "safety": 5.0},
        driven_by_needs=["social", "safety"],
        opportunity_triggers=["being_held", "familiar_person"],
        duration_seconds=5.0,
        interruptible=True,
        cooldown_seconds=60.0,
        energy_cost=2.0,
        tags=["reactive", "social", "affection"],
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
