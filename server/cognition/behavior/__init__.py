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
)
from .conditions import (
    ConditionNode,
    PersonDetectedCondition,
    NeedCriticalCondition,
    TimeElapsedCondition,
    TriggerActiveCondition,
    NotCondition,
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
    # Condition nodes
    "ConditionNode",
    "PersonDetectedCondition",
    "NeedCriticalCondition",
    "TimeElapsedCondition",
    "TriggerActiveCondition",
    "NotCondition",
    # Factory and executor
    "BehaviorTreeFactory",
    "BehaviorTreeExecutor",
    "ExecutionState",
    "ExecutionResult",
]
