"""
Murph - Behavior System
Utility AI behavior selection, evaluation, and execution.
"""

from .behavior import Behavior
from .behavior_registry import BehaviorRegistry, DEFAULT_BEHAVIORS
from .context import WorldContext
from .evaluator import BehaviorEvaluator, ScoredBehavior
from .actions import (
    ActionNode,
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
    ConditionNode,
    PersonDetectedCondition,
    NeedCriticalCondition,
    TimeElapsedCondition,
    TriggerActiveCondition,
    NotCondition,
    AtLandmarkCondition,
    ZoneSafetyCondition,
    HasPathCondition,
    HasUnexploredZonesCondition,
    PositionKnownCondition,
)
from .trees import BehaviorTreeFactory
from .tree_executor import BehaviorTreeExecutor, ExecutionState, ExecutionResult

__all__ = [
    # Core behavior types
    "Behavior",
    "BehaviorRegistry",
    "DEFAULT_BEHAVIORS",
    "WorldContext",
    # Evaluator
    "BehaviorEvaluator",
    "ScoredBehavior",
    # Action nodes
    "ActionNode",
    "MoveAction",
    "TurnAction",
    "PlaySoundAction",
    "SetExpressionAction",
    "WaitAction",
    "ScanAction",
    "StopAction",
    "NavigateToLandmarkAction",
    "ReorientAction",
    "MoveTowardSafetyAction",
    # Condition nodes
    "ConditionNode",
    "PersonDetectedCondition",
    "NeedCriticalCondition",
    "TimeElapsedCondition",
    "TriggerActiveCondition",
    "NotCondition",
    "AtLandmarkCondition",
    "ZoneSafetyCondition",
    "HasPathCondition",
    "HasUnexploredZonesCondition",
    "PositionKnownCondition",
    # Factory and executor
    "BehaviorTreeFactory",
    "BehaviorTreeExecutor",
    "ExecutionState",
    "ExecutionResult",
]
