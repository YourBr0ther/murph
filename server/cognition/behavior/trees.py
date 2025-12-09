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
)
from .conditions import PersonDetectedCondition, TriggerActiveCondition


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
    """Nuzzle tree: Express love and get close."""
    return Sequence(
        name="nuzzle",
        memory=True,
        children=[
            SetExpressionAction("love"),
            PlaySoundAction("affection"),
            MoveAction("forward", speed=0.15, duration=1.5),
            WaitAction(3.0),
        ],
    )


@BehaviorTreeFactory.register_tree("be_petted")
def create_be_petted_tree() -> py_trees.behaviour.Behaviour:
    """Be petted tree: Show contentment while being petted."""
    return Sequence(
        name="be_petted",
        memory=True,
        children=[
            SetExpressionAction("happy"),
            PlaySoundAction("happy"),
            WaitAction(10.0),
        ],
    )


@BehaviorTreeFactory.register_tree("cuddle")
def create_cuddle_tree() -> py_trees.behaviour.Behaviour:
    """Cuddle tree: Stay still and enjoy affection."""
    return Sequence(
        name="cuddle",
        memory=True,
        children=[
            SetExpressionAction("love"),
            PlaySoundAction("affection"),
            WaitAction(15.0),
        ],
    )


@BehaviorTreeFactory.register_tree("request_attention")
def create_request_attention_tree() -> py_trees.behaviour.Behaviour:
    """Request attention tree: Make sounds to get noticed."""
    return Sequence(
        name="request_attention",
        memory=True,
        children=[
            SetExpressionAction("curious"),
            PlaySoundAction("curious"),
            MoveAction("forward", speed=0.2, duration=0.5),
            MoveAction("backward", speed=0.2, duration=0.5),
            PlaySoundAction("playful"),
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
    """Bounce tree: Quick back-and-forth movements."""
    return Sequence(
        name="bounce",
        memory=True,
        children=[
            SetExpressionAction("playful"),
            PlaySoundAction("happy"),
            MoveAction("forward", speed=0.4, duration=0.3),
            MoveAction("backward", speed=0.4, duration=0.3),
            MoveAction("forward", speed=0.4, duration=0.3),
            MoveAction("backward", speed=0.4, duration=0.3),
        ],
    )


@BehaviorTreeFactory.register_tree("pounce")
def create_pounce_tree() -> py_trees.behaviour.Behaviour:
    """Pounce tree: Pause, then quick forward burst."""
    return Sequence(
        name="pounce",
        memory=True,
        children=[
            SetExpressionAction("alert"),
            WaitAction(1.0),
            SetExpressionAction("playful"),
            PlaySoundAction("playful"),
            MoveAction("forward", speed=0.8, duration=0.5),
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
    """Settle tree: Get comfortable in place."""
    return Sequence(
        name="settle",
        memory=True,
        children=[
            SetExpressionAction("neutral"),
            TurnAction(angle=15.0, speed=0.2),
            TurnAction(angle=-15.0, speed=0.2),
            WaitAction(5.0),
        ],
    )


@BehaviorTreeFactory.register_tree("adjust_position")
def create_adjust_position_tree() -> py_trees.behaviour.Behaviour:
    """Adjust position tree: Minor movement adjustment."""
    return Sequence(
        name="adjust_position",
        memory=True,
        children=[
            MoveAction("forward", speed=0.1, duration=0.3),
            MoveAction("backward", speed=0.1, duration=0.2),
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
    """Hide tree: Find cover and stay still."""
    return Sequence(
        name="hide",
        memory=True,
        children=[
            SetExpressionAction("scared"),
            MoveAction("backward", speed=0.3, duration=1.0),
            TurnAction(angle=180.0, speed=0.5),
            MoveAction("forward", speed=0.4, duration=2.0),
            StopAction(),
            WaitAction(10.0),
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
    """Scan environment tree: Look around for threats."""
    return Sequence(
        name="scan",
        memory=True,
        children=[
            SetExpressionAction("alert"),
            StopAction(),
            ScanAction("full"),
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
