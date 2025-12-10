"""
Murph - Behavior Tree Factory
Creates behavior trees for each behavior type.
"""

from typing import Callable

import py_trees
from py_trees.composites import Selector, Sequence

from .actions import (
    MoveAction,
    TurnAction,
    PlaySoundAction,
    SetExpressionAction,
    WaitAction,
    ScanAction,
    StopAction,
    NavigateToLandmarkAction,
    ReorientAction,
    MoveTowardSafetyAction,
)
from .conditions import (
    PersonDetectedCondition,
    TriggerActiveCondition,
    AtLandmarkCondition,
    ZoneSafetyCondition,
    PositionKnownCondition,
)


class BehaviorTreeFactory:
    """
    Factory for creating behavior trees.

    Each behavior defined in BehaviorRegistry has a corresponding tree
    that implements the actual execution sequence.

    Trees are intentionally simple (3-5 nodes) for the initial implementation.
    Complex trees will be added later as behaviors are refined.
    """

    _TREE_CREATORS: dict[str, Callable[[], py_trees.behaviour.Behaviour]] = {}

    @classmethod
    def register_tree(cls, behavior_name: str) -> Callable:
        """Decorator to register a tree creator method."""
        def decorator(func: Callable[[], py_trees.behaviour.Behaviour]) -> Callable:
            cls._TREE_CREATORS[behavior_name] = func
            return func
        return decorator

    @classmethod
    def create_tree(cls, behavior_name: str) -> py_trees.behaviour.Behaviour:
        """
        Create a behavior tree for the given behavior name.

        Args:
            behavior_name: Name of the behavior (e.g., "explore", "greet")

        Returns:
            Root node of the behavior tree, or fallback if no tree defined
        """
        creator = cls._TREE_CREATORS.get(behavior_name)
        if creator is None:
            return cls._create_fallback_tree(behavior_name)
        return creator()

    @classmethod
    def has_tree(cls, behavior_name: str) -> bool:
        """Check if a tree is defined for this behavior."""
        return behavior_name in cls._TREE_CREATORS

    @classmethod
    def available_trees(cls) -> list[str]:
        """Get list of behaviors with defined trees."""
        return list(cls._TREE_CREATORS.keys())

    @staticmethod
    def _create_fallback_tree(behavior_name: str) -> py_trees.behaviour.Behaviour:
        """
        Create a simple fallback tree for behaviors without specific trees.

        Just shows an expression and waits.
        """
        return Sequence(
            name=f"{behavior_name}_fallback",
            memory=True,
            children=[
                SetExpressionAction("neutral"),
                WaitAction(3.0),
            ],
        )


# --- EXPLORATION BEHAVIORS ---

@BehaviorTreeFactory.register_tree("explore")
def create_explore_tree() -> py_trees.behaviour.Behaviour:
    """Exploration tree with person detection interrupt."""
    return Selector(
        name="explore",
        memory=False,
        children=[
            # Branch 1: Person detected - express happiness and stop
            Sequence(
                name="explore_person_detected",
                memory=True,
                children=[
                    PersonDetectedCondition(familiar_only=False, max_distance=150.0),
                    SetExpressionAction("happy"),
                    PlaySoundAction("greeting"),
                    StopAction(),
                ],
            ),
            # Branch 2: Normal exploration
            Sequence(
                name="explore_default",
                memory=True,
                children=[
                    SetExpressionAction("curious"),
                    ScanAction("partial"),
                    MoveAction("forward", speed=0.4, duration=2.0),
                    TurnAction(angle=45.0, speed=0.3),
                    MoveAction("forward", speed=0.4, duration=1.5),
                ],
            ),
        ],
    )


@BehaviorTreeFactory.register_tree("investigate")
def create_investigate_tree() -> py_trees.behaviour.Behaviour:
    """Investigation tree: Approach cautiously and examine."""
    return Sequence(
        name="investigate",
        memory=True,
        children=[
            SetExpressionAction("curious"),
            PlaySoundAction("curious"),
            MoveAction("forward", speed=0.2, duration=1.5),
            ScanAction("quick"),
            WaitAction(2.0),
        ],
    )


@BehaviorTreeFactory.register_tree("observe")
def create_observe_tree() -> py_trees.behaviour.Behaviour:
    """Observe tree: Stay still and watch."""
    return Sequence(
        name="observe",
        memory=True,
        children=[
            SetExpressionAction("curious"),
            ScanAction("partial"),
            WaitAction(5.0),
        ],
    )


@BehaviorTreeFactory.register_tree("wander")
def create_wander_tree() -> py_trees.behaviour.Behaviour:
    """Wander tree with edge safety check."""
    return Selector(
        name="wander",
        memory=False,
        children=[
            # Branch 1: Near edge - stop and turn away
            Sequence(
                name="wander_edge_detected",
                memory=True,
                children=[
                    TriggerActiveCondition("near_edge"),
                    SetExpressionAction("alert"),
                    StopAction(),
                    MoveAction("backward", speed=0.3, duration=0.5),
                    TurnAction(angle=180.0, speed=0.4),
                ],
            ),
            # Branch 2: Normal wandering
            Sequence(
                name="wander_default",
                memory=True,
                children=[
                    SetExpressionAction("neutral"),
                    MoveAction("forward", speed=0.3, duration=1.0),
                    TurnAction(angle=90.0, speed=0.4),
                    MoveAction("forward", speed=0.3, duration=1.0),
                    TurnAction(angle=-45.0, speed=0.4),
                ],
            ),
        ],
    )


# --- SOCIAL BEHAVIORS ---

@BehaviorTreeFactory.register_tree("greet")
def create_greet_tree() -> py_trees.behaviour.Behaviour:
    """Greeting tree: Express happiness, make sound, approach."""
    return Sequence(
        name="greet",
        memory=True,
        children=[
            SetExpressionAction("happy"),
            PlaySoundAction("greeting"),
            MoveAction("forward", speed=0.3, duration=1.0),
            WaitAction(0.5),
        ],
    )


@BehaviorTreeFactory.register_tree("follow")
def create_follow_tree() -> py_trees.behaviour.Behaviour:
    """Follow tree: Keep moving toward person."""
    return Sequence(
        name="follow",
        memory=True,
        children=[
            SetExpressionAction("happy"),
            MoveAction("forward", speed=0.3, duration=2.0),
            ScanAction("quick"),
        ],
    )


@BehaviorTreeFactory.register_tree("interact")
def create_interact_tree() -> py_trees.behaviour.Behaviour:
    """Interact tree: Playful engagement with person."""
    return Sequence(
        name="interact",
        memory=True,
        children=[
            SetExpressionAction("playful"),
            PlaySoundAction("playful"),
            MoveAction("forward", speed=0.2, duration=0.5),
            MoveAction("backward", speed=0.2, duration=0.5),
            WaitAction(1.0),
        ],
    )


@BehaviorTreeFactory.register_tree("approach")
def create_approach_tree() -> py_trees.behaviour.Behaviour:
    """Approach tree: Move toward detected person."""
    return Sequence(
        name="approach",
        memory=True,
        children=[
            SetExpressionAction("curious"),
            MoveAction("forward", speed=0.3, duration=3.0),
        ],
    )


# --- AFFECTION BEHAVIORS ---

@BehaviorTreeFactory.register_tree("nuzzle")
def create_nuzzle_tree() -> py_trees.behaviour.Behaviour:
    """Nuzzle tree: Express love with gentle rhythmic nuzzling movements."""
    return Sequence(
        name="nuzzle",
        memory=True,
        children=[
            # Approach with love
            SetExpressionAction("love"),
            PlaySoundAction("affection"),
            MoveAction("forward", speed=0.15, duration=1.0),
            # Rhythmic nuzzling motions (gentle forward-back)
            MoveAction("forward", speed=0.1, duration=0.3),
            MoveAction("backward", speed=0.1, duration=0.2),
            PlaySoundAction("happy"),
            MoveAction("forward", speed=0.1, duration=0.3),
            MoveAction("backward", speed=0.1, duration=0.2),
            # Side nuzzle (slight turns)
            TurnAction(angle=15.0, speed=0.2),
            MoveAction("forward", speed=0.1, duration=0.2),
            TurnAction(angle=-30.0, speed=0.2),
            MoveAction("forward", speed=0.1, duration=0.2),
            TurnAction(angle=15.0, speed=0.2),
            # Final affection
            PlaySoundAction("affection"),
            SetExpressionAction("happy"),
            WaitAction(1.0),
        ],
    )


@BehaviorTreeFactory.register_tree("be_petted")
def create_be_petted_tree() -> py_trees.behaviour.Behaviour:
    """Be petted tree: Show contentment with leaning and periodic happy sounds."""
    return Sequence(
        name="be_petted",
        memory=True,
        children=[
            # Initial happy response
            SetExpressionAction("happy"),
            PlaySoundAction("happy"),
            # Lean into the petting (slight forward movement)
            MoveAction("forward", speed=0.1, duration=0.3),
            WaitAction(2.0),
            # Blissful expression
            SetExpressionAction("love"),
            PlaySoundAction("affection"),
            WaitAction(2.5),
            # Another lean
            MoveAction("forward", speed=0.08, duration=0.2),
            PlaySoundAction("happy"),
            WaitAction(2.5),
            # Contentment
            SetExpressionAction("happy"),
            PlaySoundAction("affection"),
            WaitAction(2.0),
        ],
    )


@BehaviorTreeFactory.register_tree("cuddle")
def create_cuddle_tree() -> py_trees.behaviour.Behaviour:
    """Cuddle tree: Settle in and enjoy extended affection with gentle movements."""
    return Sequence(
        name="cuddle",
        memory=True,
        children=[
            # Initial settling
            SetExpressionAction("love"),
            PlaySoundAction("affection"),
            # Gentle settling movements (like getting comfortable)
            TurnAction(angle=5.0, speed=0.1),
            TurnAction(angle=-5.0, speed=0.1),
            WaitAction(3.0),
            # Periodic contentment sounds
            PlaySoundAction("happy"),
            WaitAction(4.0),
            # Slight nuzzle motion
            MoveAction("forward", speed=0.1, duration=0.2),
            MoveAction("backward", speed=0.1, duration=0.2),
            PlaySoundAction("affection"),
            WaitAction(4.0),
            # Final contentment
            SetExpressionAction("happy"),
            PlaySoundAction("happy"),
            WaitAction(3.0),
        ],
    )


@BehaviorTreeFactory.register_tree("request_attention")
def create_request_attention_tree() -> py_trees.behaviour.Behaviour:
    """Request attention tree: Get noticed with conditional excitement if person responds."""
    return Selector(
        name="request_attention",
        memory=False,
        children=[
            # Branch 1: Person detected - express excitement!
            Sequence(
                name="request_attention_noticed",
                memory=True,
                children=[
                    PersonDetectedCondition(familiar_only=False),
                    SetExpressionAction("happy"),
                    PlaySoundAction("happy"),
                    MoveAction("forward", speed=0.3, duration=0.5),
                    PlaySoundAction("greeting"),
                ],
            ),
            # Branch 2: No person - try harder to get attention
            Sequence(
                name="request_attention_seeking",
                memory=True,
                children=[
                    SetExpressionAction("curious"),
                    PlaySoundAction("curious"),
                    # Attention-getting movements
                    MoveAction("forward", speed=0.25, duration=0.4),
                    MoveAction("backward", speed=0.25, duration=0.4),
                    PlaySoundAction("playful"),
                    # Look around hoping to be noticed
                    TurnAction(angle=30.0, speed=0.4),
                    TurnAction(angle=-60.0, speed=0.4),
                    TurnAction(angle=30.0, speed=0.4),
                    PlaySoundAction("curious"),
                ],
            ),
        ],
    )


# --- PLAY BEHAVIORS ---

@BehaviorTreeFactory.register_tree("play")
def create_play_tree() -> py_trees.behaviour.Behaviour:
    """Play tree: Energetic movements."""
    return Sequence(
        name="play",
        memory=True,
        children=[
            SetExpressionAction("playful"),
            PlaySoundAction("playful"),
            MoveAction("forward", speed=0.5, duration=1.0),
            TurnAction(angle=180.0, speed=0.6),
            MoveAction("forward", speed=0.5, duration=1.0),
        ],
    )


@BehaviorTreeFactory.register_tree("chase")
def create_chase_tree() -> py_trees.behaviour.Behaviour:
    """Chase tree: Fast pursuit movement."""
    return Sequence(
        name="chase",
        memory=True,
        children=[
            SetExpressionAction("playful"),
            PlaySoundAction("playful"),
            MoveAction("forward", speed=0.7, duration=2.0),
            TurnAction(angle=30.0, speed=0.5),
            MoveAction("forward", speed=0.7, duration=1.5),
        ],
    )


@BehaviorTreeFactory.register_tree("bounce")
def create_bounce_tree() -> py_trees.behaviour.Behaviour:
    """Bounce tree: Energetic bouncing with directional variations."""
    return Sequence(
        name="bounce",
        memory=True,
        children=[
            SetExpressionAction("playful"),
            PlaySoundAction("happy"),
            # Bounce 1: Forward-back
            MoveAction("forward", speed=0.5, duration=0.25),
            MoveAction("backward", speed=0.5, duration=0.25),
            # Bounce 2: With slight turn
            TurnAction(angle=30.0, speed=0.6),
            MoveAction("forward", speed=0.5, duration=0.25),
            MoveAction("backward", speed=0.5, duration=0.25),
            # Bounce 3: Other direction
            TurnAction(angle=-60.0, speed=0.6),
            MoveAction("forward", speed=0.5, duration=0.25),
            MoveAction("backward", speed=0.5, duration=0.25),
            PlaySoundAction("playful"),
            # Final spin for excitement
            TurnAction(angle=180.0, speed=0.8),
            SetExpressionAction("happy"),
        ],
    )


@BehaviorTreeFactory.register_tree("pounce")
def create_pounce_tree() -> py_trees.behaviour.Behaviour:
    """Pounce tree: Stalk, wiggle, then burst forward like a cat."""
    return Sequence(
        name="pounce",
        memory=True,
        children=[
            # Stalking phase - alert and focused
            SetExpressionAction("alert"),
            WaitAction(0.5),
            # Wiggle side to side (like a cat preparing to pounce)
            TurnAction(angle=10.0, speed=0.3),
            TurnAction(angle=-20.0, speed=0.3),
            TurnAction(angle=10.0, speed=0.3),
            WaitAction(0.3),
            # The pounce!
            SetExpressionAction("playful"),
            PlaySoundAction("playful"),
            MoveAction("forward", speed=0.9, duration=0.4),
            # Landing
            StopAction(),
            PlaySoundAction("happy"),
        ],
    )


# --- REST/RECOVERY BEHAVIORS ---

@BehaviorTreeFactory.register_tree("rest")
def create_rest_tree() -> py_trees.behaviour.Behaviour:
    """Rest tree: Stop and recover energy."""
    return Sequence(
        name="rest",
        memory=True,
        children=[
            StopAction(),
            SetExpressionAction("sleepy"),
            WaitAction(20.0),
        ],
    )


@BehaviorTreeFactory.register_tree("sleep")
def create_sleep_tree() -> py_trees.behaviour.Behaviour:
    """Sleep tree: Deep rest state."""
    return Sequence(
        name="sleep",
        memory=True,
        children=[
            StopAction(),
            SetExpressionAction("sleepy"),
            PlaySoundAction("sleepy"),
            WaitAction(60.0),
        ],
    )


@BehaviorTreeFactory.register_tree("find_cozy_spot")
def create_find_cozy_spot_tree() -> py_trees.behaviour.Behaviour:
    """Find cozy spot tree: Move around looking for comfort."""
    return Sequence(
        name="find_cozy_spot",
        memory=True,
        children=[
            SetExpressionAction("neutral"),
            ScanAction("partial"),
            MoveAction("forward", speed=0.2, duration=2.0),
            TurnAction(angle=45.0, speed=0.3),
            StopAction(),
        ],
    )


@BehaviorTreeFactory.register_tree("settle")
def create_settle_tree() -> py_trees.behaviour.Behaviour:
    """Settle tree: Circle like a dog before lying down, then settle with a sigh."""
    return Sequence(
        name="settle",
        memory=True,
        children=[
            SetExpressionAction("neutral"),
            # Circling behavior (like dogs do before lying down)
            TurnAction(angle=90.0, speed=0.25),
            MoveAction("forward", speed=0.15, duration=0.3),
            TurnAction(angle=90.0, speed=0.25),
            MoveAction("forward", speed=0.15, duration=0.3),
            TurnAction(angle=90.0, speed=0.25),
            MoveAction("forward", speed=0.15, duration=0.3),
            TurnAction(angle=90.0, speed=0.25),
            # Settling adjustments
            TurnAction(angle=10.0, speed=0.15),
            TurnAction(angle=-10.0, speed=0.15),
            # Final settle with sigh
            SetExpressionAction("sleepy"),
            PlaySoundAction("sigh"),
            WaitAction(3.0),
        ],
    )


@BehaviorTreeFactory.register_tree("adjust_position")
def create_adjust_position_tree() -> py_trees.behaviour.Behaviour:
    """Adjust position tree: Naturalistic fidgeting with small movements and turns."""
    return Sequence(
        name="adjust_position",
        memory=True,
        children=[
            # Small shuffle forward-back
            MoveAction("forward", speed=0.1, duration=0.2),
            MoveAction("backward", speed=0.1, duration=0.15),
            # Slight turn adjustment
            TurnAction(angle=8.0, speed=0.15),
            WaitAction(0.3),
            # Another small shuffle
            MoveAction("left", speed=0.1, duration=0.15),
            MoveAction("right", speed=0.1, duration=0.15),
            # Settle
            TurnAction(angle=-8.0, speed=0.15),
        ],
    )


# --- SAFETY BEHAVIORS ---

@BehaviorTreeFactory.register_tree("retreat")
def create_retreat_tree() -> py_trees.behaviour.Behaviour:
    """Retreat tree: Back away quickly."""
    return Sequence(
        name="retreat",
        memory=True,
        children=[
            SetExpressionAction("scared"),
            PlaySoundAction("alert"),
            MoveAction("backward", speed=0.6, duration=1.5),
            StopAction(),
        ],
    )


@BehaviorTreeFactory.register_tree("hide")
def create_hide_tree() -> py_trees.behaviour.Behaviour:
    """Hide tree: Find cover, peek cautiously, then stay hidden with relief."""
    return Sequence(
        name="hide",
        memory=True,
        children=[
            # Initial scare and retreat
            SetExpressionAction("scared"),
            PlaySoundAction("alert"),
            MoveAction("backward", speed=0.4, duration=1.0),
            TurnAction(angle=180.0, speed=0.5),
            MoveAction("forward", speed=0.5, duration=2.0),
            StopAction(),
            # Hiding - cautious peek behavior
            WaitAction(2.0),
            SetExpressionAction("alert"),
            ScanAction("quick"),
            # Still scared, hunker down
            SetExpressionAction("scared"),
            WaitAction(3.0),
            # Another cautious peek
            TurnAction(angle=30.0, speed=0.2),
            ScanAction("quick"),
            TurnAction(angle=-30.0, speed=0.2),
            WaitAction(2.0),
            # Danger seems to have passed - relief
            SetExpressionAction("neutral"),
            PlaySoundAction("sigh"),
            WaitAction(2.0),
        ],
    )


@BehaviorTreeFactory.register_tree("approach_trusted")
def create_approach_trusted_tree() -> py_trees.behaviour.Behaviour:
    """Approach trusted tree: Move toward familiar person for safety."""
    return Sequence(
        name="approach_trusted",
        memory=True,
        children=[
            SetExpressionAction("scared"),
            MoveAction("forward", speed=0.4, duration=3.0),
            SetExpressionAction("happy"),
            PlaySoundAction("happy"),
        ],
    )


@BehaviorTreeFactory.register_tree("scan")
def create_scan_tree() -> py_trees.behaviour.Behaviour:
    """Scan environment tree: Multi-directional scanning with threat assessment."""
    return Sequence(
        name="scan",
        memory=True,
        children=[
            # Initial alert
            SetExpressionAction("alert"),
            StopAction(),
            # Scan left
            TurnAction(angle=-45.0, speed=0.3),
            ScanAction("quick"),
            WaitAction(0.5),
            # Scan right
            TurnAction(angle=90.0, speed=0.3),
            ScanAction("quick"),
            WaitAction(0.5),
            # Scan center
            TurnAction(angle=-45.0, speed=0.3),
            ScanAction("partial"),
            # Assessment complete - relax if nothing found
            SetExpressionAction("curious"),
            WaitAction(0.5),
            SetExpressionAction("neutral"),
        ],
    )


# --- IDLE BEHAVIOR ---

@BehaviorTreeFactory.register_tree("idle")
def create_idle_tree() -> py_trees.behaviour.Behaviour:
    """Idle tree: Do nothing in particular."""
    return Sequence(
        name="idle",
        memory=True,
        children=[
            SetExpressionAction("neutral"),
            WaitAction(3.0),
        ],
    )


# --- LONELINESS BEHAVIORS ---

@BehaviorTreeFactory.register_tree("sigh")
def create_sigh_tree() -> py_trees.behaviour.Behaviour:
    """Sigh tree: Express loneliness with a sad sound and expression."""
    return Sequence(
        name="sigh",
        memory=True,
        children=[
            SetExpressionAction("sad"),
            PlaySoundAction("sigh"),
            WaitAction(2.0),
            SetExpressionAction("neutral"),
        ],
    )


@BehaviorTreeFactory.register_tree("mope")
def create_mope_tree() -> py_trees.behaviour.Behaviour:
    """Mope tree: Slow, sad movement expressing loneliness."""
    return Sequence(
        name="mope",
        memory=True,
        children=[
            SetExpressionAction("sad"),
            MoveAction("forward", speed=0.15, duration=1.5),
            WaitAction(1.0),
            TurnAction(angle=30.0, speed=0.2),
            WaitAction(1.5),
            MoveAction("forward", speed=0.1, duration=1.0),
            PlaySoundAction("sad"),
            WaitAction(2.0),
        ],
    )


@BehaviorTreeFactory.register_tree("perk_up_hopeful")
def create_perk_up_hopeful_tree() -> py_trees.behaviour.Behaviour:
    """Perk up hopeful tree: Quick look around hoping to find someone."""
    return Sequence(
        name="perk_up_hopeful",
        memory=True,
        children=[
            SetExpressionAction("alert"),
            ScanAction("quick"),
            SetExpressionAction("curious"),
            PlaySoundAction("curious"),
            WaitAction(2.0),
            SetExpressionAction("sad"),
        ],
    )


@BehaviorTreeFactory.register_tree("seek_company")
def create_seek_company_tree() -> py_trees.behaviour.Behaviour:
    """Seek company tree with person detection success."""
    return Selector(
        name="seek_company",
        memory=False,
        children=[
            # Branch 1: Found someone! Express joy
            Sequence(
                name="seek_company_found",
                memory=True,
                children=[
                    PersonDetectedCondition(familiar_only=False),
                    SetExpressionAction("happy"),
                    PlaySoundAction("happy"),
                    MoveAction("forward", speed=0.3, duration=1.0),
                ],
            ),
            # Branch 2: Keep searching
            Sequence(
                name="seek_company_searching",
                memory=True,
                children=[
                    SetExpressionAction("curious"),
                    PlaySoundAction("curious"),
                    ScanAction("partial"),
                    MoveAction("forward", speed=0.3, duration=2.0),
                    TurnAction(angle=90.0, speed=0.4),
                    ScanAction("quick"),
                    MoveAction("forward", speed=0.3, duration=2.0),
                    SetExpressionAction("sad"),
                ],
            ),
        ],
    )


# --- NAVIGATION BEHAVIORS ---


@BehaviorTreeFactory.register_tree("go_home")
def create_go_home_tree() -> py_trees.behaviour.Behaviour:
    """Navigate to home base with position-lost fallback."""
    return Selector(
        name="go_home",
        memory=False,
        children=[
            # Branch 1: Already at home - success
            Sequence(
                name="go_home_arrived",
                memory=True,
                children=[
                    TriggerActiveCondition("at_home"),
                    SetExpressionAction("happy"),
                    PlaySoundAction("happy"),
                ],
            ),
            # Branch 2: Position lost - scan first
            Sequence(
                name="go_home_lost",
                memory=True,
                children=[
                    TriggerActiveCondition("position_lost"),
                    SetExpressionAction("alert"),
                    ScanAction("full"),
                    WaitAction(1.0),
                ],
            ),
            # Branch 3: Navigate to home
            Sequence(
                name="go_home_navigate",
                memory=True,
                children=[
                    SetExpressionAction("neutral"),
                    NavigateToLandmarkAction(target_type="home_base", timeout=45.0),
                    SetExpressionAction("happy"),
                ],
            ),
        ],
    )


@BehaviorTreeFactory.register_tree("go_to_charger")
def create_go_to_charger_tree() -> py_trees.behaviour.Behaviour:
    """Navigate to charging station - high priority when energy low."""
    return Selector(
        name="go_to_charger",
        memory=False,
        children=[
            # Branch 1: Already at charger
            Sequence(
                name="go_to_charger_arrived",
                memory=True,
                children=[
                    TriggerActiveCondition("at_charger"),
                    SetExpressionAction("sleepy"),
                    StopAction(),
                ],
            ),
            # Branch 2: Position lost - scan desperately
            Sequence(
                name="go_to_charger_lost",
                memory=True,
                children=[
                    TriggerActiveCondition("position_lost"),
                    SetExpressionAction("scared"),
                    ReorientAction(max_attempts=5),
                ],
            ),
            # Branch 3: Navigate to charger
            Sequence(
                name="go_to_charger_navigate",
                memory=True,
                children=[
                    SetExpressionAction("neutral"),
                    NavigateToLandmarkAction(target_type="charging_station", timeout=45.0),
                    SetExpressionAction("sleepy"),
                ],
            ),
        ],
    )


@BehaviorTreeFactory.register_tree("go_to_landmark")
def create_go_to_landmark_tree() -> py_trees.behaviour.Behaviour:
    """Navigate to a specific landmark (generic)."""
    return Selector(
        name="go_to_landmark",
        memory=False,
        children=[
            # Branch 1: Position lost - scan first
            Sequence(
                name="go_to_landmark_lost",
                memory=True,
                children=[
                    TriggerActiveCondition("position_lost"),
                    SetExpressionAction("alert"),
                    ScanAction("full"),
                ],
            ),
            # Branch 2: Navigate
            Sequence(
                name="go_to_landmark_navigate",
                memory=True,
                children=[
                    SetExpressionAction("curious"),
                    NavigateToLandmarkAction(target_type="landmark", timeout=30.0),
                    SetExpressionAction("happy"),
                ],
            ),
        ],
    )


@BehaviorTreeFactory.register_tree("explore_unfamiliar")
def create_explore_unfamiliar_tree() -> py_trees.behaviour.Behaviour:
    """Explore areas with low familiarity."""
    return Selector(
        name="explore_unfamiliar",
        memory=False,
        children=[
            # Branch 1: Found something interesting (person)
            Sequence(
                name="explore_unfamiliar_discovery",
                memory=True,
                children=[
                    PersonDetectedCondition(familiar_only=False),
                    SetExpressionAction("happy"),
                    PlaySoundAction("greeting"),
                    StopAction(),
                ],
            ),
            # Branch 2: Position lost - reorient
            Sequence(
                name="explore_unfamiliar_lost",
                memory=True,
                children=[
                    TriggerActiveCondition("position_lost"),
                    SetExpressionAction("curious"),
                    ScanAction("partial"),
                ],
            ),
            # Branch 3: Navigate toward unfamiliar area
            Sequence(
                name="explore_unfamiliar_navigate",
                memory=True,
                children=[
                    SetExpressionAction("curious"),
                    PlaySoundAction("curious"),
                    NavigateToLandmarkAction(target_type="unfamiliar_zone", timeout=30.0),
                    ScanAction("partial"),
                ],
            ),
        ],
    )


@BehaviorTreeFactory.register_tree("patrol")
def create_patrol_tree() -> py_trees.behaviour.Behaviour:
    """Patrol between known safe landmarks."""
    return Selector(
        name="patrol",
        memory=False,
        children=[
            # Branch 1: Danger detected - abort
            Sequence(
                name="patrol_danger",
                memory=True,
                children=[
                    TriggerActiveCondition("in_danger_zone"),
                    SetExpressionAction("alert"),
                    StopAction(),
                ],
            ),
            # Branch 2: Normal patrol
            Sequence(
                name="patrol_default",
                memory=True,
                children=[
                    SetExpressionAction("neutral"),
                    MoveAction("forward", speed=0.3, duration=3.0),
                    ScanAction("quick"),
                    TurnAction(angle=90.0, speed=0.3),
                    MoveAction("forward", speed=0.3, duration=3.0),
                    ScanAction("quick"),
                    TurnAction(angle=90.0, speed=0.3),
                ],
            ),
        ],
    )


@BehaviorTreeFactory.register_tree("flee_danger")
def create_flee_danger_tree() -> py_trees.behaviour.Behaviour:
    """Emergency escape from dangerous zone."""
    return Sequence(
        name="flee_danger",
        memory=True,
        children=[
            SetExpressionAction("scared"),
            PlaySoundAction("alert"),
            MoveTowardSafetyAction(retreat_duration=1.5, retreat_speed=0.6),
            ScanAction("quick"),
            SetExpressionAction("alert"),
        ],
    )


@BehaviorTreeFactory.register_tree("retreat_to_safe")
def create_retreat_to_safe_tree() -> py_trees.behaviour.Behaviour:
    """Retreat to a known safe zone."""
    return Selector(
        name="retreat_to_safe",
        memory=False,
        children=[
            # Branch 1: Already in safe zone
            Sequence(
                name="retreat_to_safe_arrived",
                memory=True,
                children=[
                    TriggerActiveCondition("in_safe_zone"),
                    SetExpressionAction("neutral"),
                    StopAction(),
                ],
            ),
            # Branch 2: Navigate to safety
            Sequence(
                name="retreat_to_safe_navigate",
                memory=True,
                children=[
                    SetExpressionAction("scared"),
                    NavigateToLandmarkAction(target_type="safe_zone", timeout=20.0),
                    SetExpressionAction("neutral"),
                    PlaySoundAction("happy"),
                ],
            ),
        ],
    )


@BehaviorTreeFactory.register_tree("reorient")
def create_reorient_tree() -> py_trees.behaviour.Behaviour:
    """Attempt to regain position awareness."""
    return Sequence(
        name="reorient",
        memory=True,
        children=[
            SetExpressionAction("curious"),
            StopAction(),
            ReorientAction(max_attempts=4, scan_duration=2.0),
            SetExpressionAction("neutral"),
        ],
    )
