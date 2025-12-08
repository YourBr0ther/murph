"""
Murph - Behavior Tree Condition Nodes
World state checks for behavior tree execution.
"""

import time
from typing import Any

import py_trees
from py_trees.common import Status

from .context import WorldContext


class ConditionNode(py_trees.behaviour.Behaviour):
    """
    Base class for condition nodes.

    Conditions check world state each tick and return SUCCESS or FAILURE.
    They never return RUNNING.

    Conditions receive the current WorldContext via the blackboard.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._context: WorldContext | None = None

    def setup(self, **kwargs: Any) -> None:
        """Setup the blackboard access."""
        self.blackboard = self.attach_blackboard_client(name=self.name)
        self.blackboard.register_key(
            key="world_context",
            access=py_trees.common.Access.READ
        )

    def update(self) -> Status:
        """Read context from blackboard and evaluate condition."""
        try:
            self._context = self.blackboard.world_context
        except KeyError:
            # No context available - condition fails
            return Status.FAILURE

        return self._evaluate()

    def _evaluate(self) -> Status:
        """Override this to implement condition logic."""
        raise NotImplementedError


class PersonDetectedCondition(ConditionNode):
    """
    Check if a person is detected.

    Args:
        familiar_only: If True, only succeeds for familiar people
        max_distance: Maximum distance (cm) to consider person detected
    """

    def __init__(
        self,
        familiar_only: bool = False,
        max_distance: float | None = None,
    ) -> None:
        name = f"PersonDetected(familiar={familiar_only})"
        super().__init__(name)
        self._familiar_only = familiar_only
        self._max_distance = max_distance

    def _evaluate(self) -> Status:
        if self._context is None or not self._context.person_detected:
            return Status.FAILURE

        if self._familiar_only and not self._context.person_is_familiar:
            return Status.FAILURE

        if self._max_distance is not None:
            distance = self._context.person_distance
            if distance is None or distance > self._max_distance:
                return Status.FAILURE

        return Status.SUCCESS


class NeedCriticalCondition(ConditionNode):
    """
    Check if a specific need is critical.

    This requires access to the NeedsSystem via blackboard.

    Args:
        need_name: Name of the need to check
    """

    def __init__(self, need_name: str) -> None:
        name = f"NeedCritical({need_name})"
        super().__init__(name)
        self._need_name = need_name

    def setup(self, **kwargs: Any) -> None:
        """Setup blackboard access for both context and needs."""
        super().setup(**kwargs)
        self.blackboard.register_key(
            key="needs_system",
            access=py_trees.common.Access.READ
        )

    def _evaluate(self) -> Status:
        try:
            needs_system = self.blackboard.needs_system
        except KeyError:
            return Status.FAILURE

        need = needs_system.get_need(self._need_name)
        if need is None:
            return Status.FAILURE

        return Status.SUCCESS if need.is_critical() else Status.FAILURE


class TimeElapsedCondition(ConditionNode):
    """
    Check if specified time has elapsed since condition was first checked.

    Useful for time-limited behaviors.

    Args:
        seconds: Seconds that must elapse
    """

    def __init__(self, seconds: float) -> None:
        name = f"TimeElapsed({seconds:.1f}s)"
        super().__init__(name)
        self._required_seconds = seconds
        self._start_time: float | None = None

    def initialise(self) -> None:
        """Record start time on first activation."""
        if self._start_time is None:
            self._start_time = time.time()

    def _evaluate(self) -> Status:
        if self._start_time is None:
            return Status.FAILURE

        elapsed = time.time() - self._start_time
        return Status.SUCCESS if elapsed >= self._required_seconds else Status.FAILURE

    def terminate(self, new_status: Status) -> None:
        """Reset timer when terminated."""
        self._start_time = None


class TriggerActiveCondition(ConditionNode):
    """
    Check if a specific trigger is active in the world context.

    Args:
        trigger_name: Name of the trigger to check
    """

    def __init__(self, trigger_name: str) -> None:
        name = f"TriggerActive({trigger_name})"
        super().__init__(name)
        self._trigger_name = trigger_name

    def _evaluate(self) -> Status:
        if self._context is None:
            return Status.FAILURE

        return Status.SUCCESS if self._context.has_trigger(self._trigger_name) else Status.FAILURE


class NotCondition(py_trees.decorators.Decorator):
    """
    Invert a condition result.

    Wraps another condition and inverts SUCCESS/FAILURE.
    """

    def __init__(
        self,
        child: py_trees.behaviour.Behaviour,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name or f"Not({child.name})", child=child)

    def update(self) -> Status:
        """Invert child status."""
        if self.decorated.status == Status.SUCCESS:
            return Status.FAILURE
        elif self.decorated.status == Status.FAILURE:
            return Status.SUCCESS
        return Status.RUNNING
