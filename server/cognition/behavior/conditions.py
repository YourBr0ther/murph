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


# --- Spatial Navigation Conditions ---


class AtLandmarkCondition(ConditionNode):
    """
    Check if robot is at a specific landmark type or ID.

    Args:
        landmark_type: Type of landmark to check for (e.g., "home_base", "charging_station")
        landmark_id: Specific landmark ID (overrides type check)
    """

    def __init__(
        self,
        landmark_type: str | None = None,
        landmark_id: str | None = None,
    ) -> None:
        name = f"AtLandmark({landmark_type or landmark_id or 'any'})"
        super().__init__(name)
        self._landmark_type = landmark_type
        self._landmark_id = landmark_id

    def setup(self, **kwargs: Any) -> None:
        """Setup blackboard access for spatial map."""
        super().setup(**kwargs)
        self.blackboard.register_key(
            key="spatial_map",
            access=py_trees.common.Access.READ
        )

    def _evaluate(self) -> Status:
        try:
            spatial_map = self.blackboard.spatial_map
            if spatial_map is None or not spatial_map.is_position_known:
                return Status.FAILURE

            current_id = spatial_map.current_landmark_id

            # Check specific landmark ID
            if self._landmark_id is not None:
                return Status.SUCCESS if current_id == self._landmark_id else Status.FAILURE

            # Check landmark type
            if self._landmark_type is not None:
                current_landmark = spatial_map.current_landmark
                if current_landmark is None:
                    return Status.FAILURE
                if self._landmark_type == "home_base":
                    return Status.SUCCESS if spatial_map.is_at_home else Status.FAILURE
                return (
                    Status.SUCCESS
                    if current_landmark.landmark_type == self._landmark_type
                    else Status.FAILURE
                )

            # If no type or ID specified, just check if at any landmark
            return Status.SUCCESS if current_id is not None else Status.FAILURE

        except (KeyError, AttributeError):
            return Status.FAILURE


class ZoneSafetyCondition(ConditionNode):
    """
    Check if current zone meets a safety threshold.

    Args:
        min_safety: Minimum safety score required (0.0-1.0, default 0.7)
        check_dangerous: If True, succeeds when zone IS dangerous (safety < 0.3)
    """

    def __init__(
        self,
        min_safety: float = 0.7,
        check_dangerous: bool = False,
    ) -> None:
        if check_dangerous:
            name = "ZoneIsDangerous"
        else:
            name = f"ZoneSafety(>={min_safety:.1f})"
        super().__init__(name)
        self._min_safety = min_safety
        self._check_dangerous = check_dangerous

    def setup(self, **kwargs: Any) -> None:
        """Setup blackboard access for spatial map."""
        super().setup(**kwargs)
        self.blackboard.register_key(
            key="spatial_map",
            access=py_trees.common.Access.READ
        )

    def _evaluate(self) -> Status:
        try:
            spatial_map = self.blackboard.spatial_map
            if spatial_map is None:
                return Status.FAILURE

            current_zone = spatial_map.current_zone
            if current_zone is None:
                # No zone info - assume neutral
                return Status.FAILURE

            if self._check_dangerous:
                return Status.SUCCESS if current_zone.is_dangerous else Status.FAILURE

            return Status.SUCCESS if current_zone.safety_score >= self._min_safety else Status.FAILURE

        except (KeyError, AttributeError):
            return Status.FAILURE


class HasPathCondition(ConditionNode):
    """
    Check if a path exists to a target landmark type.

    Args:
        target_type: Type of landmark to find path to
                    ("home_base", "charging_station", "safe_zone", etc.)
    """

    def __init__(self, target_type: str) -> None:
        name = f"HasPathTo({target_type})"
        super().__init__(name)
        self._target_type = target_type

    def setup(self, **kwargs: Any) -> None:
        """Setup blackboard access for spatial map."""
        super().setup(**kwargs)
        self.blackboard.register_key(
            key="spatial_map",
            access=py_trees.common.Access.READ
        )

    def _evaluate(self) -> Status:
        try:
            spatial_map = self.blackboard.spatial_map
            if spatial_map is None or not spatial_map.is_position_known:
                return Status.FAILURE

            has_path = spatial_map.has_path_to(self._target_type)
            return Status.SUCCESS if has_path else Status.FAILURE

        except (KeyError, AttributeError):
            return Status.FAILURE


class HasUnexploredZonesCondition(ConditionNode):
    """
    Check if there are unexplored/unfamiliar zones in the map.

    Args:
        familiarity_threshold: Maximum familiarity to consider unexplored (default 0.5)
    """

    def __init__(self, familiarity_threshold: float = 0.5) -> None:
        name = f"HasUnexploredZones(<{familiarity_threshold:.1f})"
        super().__init__(name)
        self._threshold = familiarity_threshold

    def setup(self, **kwargs: Any) -> None:
        """Setup blackboard access for spatial map."""
        super().setup(**kwargs)
        self.blackboard.register_key(
            key="spatial_map",
            access=py_trees.common.Access.READ
        )

    def _evaluate(self) -> Status:
        try:
            spatial_map = self.blackboard.spatial_map
            if spatial_map is None:
                return Status.FAILURE

            has_unfamiliar = spatial_map.has_unfamiliar_zones(self._threshold)
            return Status.SUCCESS if has_unfamiliar else Status.FAILURE

        except (KeyError, AttributeError):
            return Status.FAILURE


class PositionKnownCondition(ConditionNode):
    """
    Check if the robot's position is known (at a recognized landmark).
    """

    def __init__(self) -> None:
        super().__init__("PositionKnown")

    def setup(self, **kwargs: Any) -> None:
        """Setup blackboard access for spatial map."""
        super().setup(**kwargs)
        self.blackboard.register_key(
            key="spatial_map",
            access=py_trees.common.Access.READ
        )

    def _evaluate(self) -> Status:
        try:
            spatial_map = self.blackboard.spatial_map
            if spatial_map is None:
                return Status.FAILURE

            return Status.SUCCESS if spatial_map.is_position_known else Status.FAILURE

        except (KeyError, AttributeError):
            return Status.FAILURE
