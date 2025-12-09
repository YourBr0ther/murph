"""
Murph - Behavior Tree Action Nodes
Low-level robot actions for behavior tree execution.
"""

import time
from typing import Any, Callable

import py_trees
from py_trees.common import Status


class ActionNode(py_trees.behaviour.Behaviour):
    """
    Base class for Murph action nodes.

    Provides common functionality:
    - Duration-based execution tracking
    - Callback-based action dispatch
    - State serialization for debugging

    Attributes:
        action_callback: Class-level callback called when actions execute.
                        Signature: callback(action_name: str, params: dict) -> bool
    """

    action_callback: Callable[[str, dict[str, Any]], bool] | None = None

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._start_time: float | None = None
        self._params: dict[str, Any] = {}

    def initialise(self) -> None:
        """Called when node transitions to RUNNING."""
        self._start_time = time.time()
        self._dispatch_action()

    def _dispatch_action(self) -> None:
        """Dispatch the action to the robot via callback."""
        if ActionNode.action_callback:
            ActionNode.action_callback(self.name, self._params)

    def terminate(self, new_status: Status) -> None:
        """Called when node transitions away from RUNNING."""
        self._start_time = None

    def elapsed_time(self) -> float:
        """Get elapsed time since action started."""
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    def get_params(self) -> dict[str, Any]:
        """Get action parameters for debugging."""
        return self._params.copy()


class MoveAction(ActionNode):
    """
    Move the robot in a direction for a duration.

    Args:
        direction: "forward", "backward", "left", "right"
        speed: 0.0-1.0 normalized speed
        duration: seconds to move (if 0, moves until interrupted)
    """

    VALID_DIRECTIONS = ("forward", "backward", "left", "right")

    def __init__(
        self,
        direction: str = "forward",
        speed: float = 0.5,
        duration: float = 1.0,
    ) -> None:
        name = f"Move({direction}, {speed:.1f}, {duration:.1f}s)"
        super().__init__(name)

        if direction not in self.VALID_DIRECTIONS:
            raise ValueError(f"Invalid direction: {direction}")

        self._direction = direction
        self._speed = max(0.0, min(1.0, speed))
        self._duration = max(0.0, duration)

        self._params = {
            "action": "move",
            "direction": self._direction,
            "speed": self._speed,
            "duration": self._duration,
        }

    def update(self) -> Status:
        """Execute one tick of the move action."""
        if self._duration > 0 and self.elapsed_time() >= self._duration:
            return Status.SUCCESS
        return Status.RUNNING


class TurnAction(ActionNode):
    """
    Turn the robot by an angle.

    Args:
        angle: degrees to turn (positive = clockwise, negative = counter-clockwise)
        speed: 0.0-1.0 normalized rotation speed
    """

    def __init__(self, angle: float = 90.0, speed: float = 0.5) -> None:
        name = f"Turn({angle:.0f}deg, {speed:.1f})"
        super().__init__(name)

        self._angle = angle
        self._speed = max(0.0, min(1.0, speed))
        # Estimate duration based on angle and speed
        self._estimated_duration = abs(angle) / (180.0 * max(0.1, speed))

        self._params = {
            "action": "turn",
            "angle": self._angle,
            "speed": self._speed,
        }

    def update(self) -> Status:
        """Execute one tick of the turn action."""
        if self.elapsed_time() >= self._estimated_duration:
            return Status.SUCCESS
        return Status.RUNNING


class PlaySoundAction(ActionNode):
    """
    Play a sound effect.

    Args:
        sound_name: name of the sound to play (e.g., "greeting", "happy", "curious")
    """

    SOUND_DURATIONS: dict[str, float] = {
        "greeting": 1.0,
        "happy": 0.5,
        "sad": 0.8,
        "curious": 0.6,
        "surprised": 0.4,
        "sleepy": 1.2,
        "playful": 0.7,
        "affection": 0.8,
        "alert": 0.3,
        "sigh": 1.5,
    }
    DEFAULT_DURATION = 0.5

    def __init__(self, sound_name: str) -> None:
        name = f"PlaySound({sound_name})"
        super().__init__(name)

        self._sound_name = sound_name
        self._duration = self.SOUND_DURATIONS.get(sound_name, self.DEFAULT_DURATION)

        self._params = {
            "action": "play_sound",
            "sound_name": self._sound_name,
        }

    def update(self) -> Status:
        """Execute one tick - sounds complete after their duration."""
        if self.elapsed_time() >= self._duration:
            return Status.SUCCESS
        return Status.RUNNING


class SetExpressionAction(ActionNode):
    """
    Set the robot's facial expression.

    Args:
        expression_name: name of expression (e.g., "happy", "sad", "curious", "neutral")
    """

    VALID_EXPRESSIONS = (
        "neutral", "happy", "sad", "curious", "surprised",
        "sleepy", "playful", "love", "scared", "alert"
    )

    def __init__(self, expression_name: str) -> None:
        name = f"SetExpression({expression_name})"
        super().__init__(name)

        self._expression_name = expression_name

        self._params = {
            "action": "set_expression",
            "expression_name": self._expression_name,
        }

    def update(self) -> Status:
        """Expressions are instant - immediately SUCCESS."""
        return Status.SUCCESS


class WaitAction(ActionNode):
    """
    Wait for a duration without doing anything.

    Args:
        duration: seconds to wait
    """

    def __init__(self, duration: float) -> None:
        name = f"Wait({duration:.1f}s)"
        super().__init__(name)

        self._duration = max(0.0, duration)

        self._params = {
            "action": "wait",
            "duration": self._duration,
        }

    def update(self) -> Status:
        """Wait until duration elapsed."""
        if self.elapsed_time() >= self._duration:
            return Status.SUCCESS
        return Status.RUNNING


class ScanAction(ActionNode):
    """
    Perform a scanning motion to look around the environment.

    Args:
        scan_type: "full" (360), "partial" (180), "quick" (90)
    """

    SCAN_DURATIONS: dict[str, float] = {
        "full": 4.0,
        "partial": 2.0,
        "quick": 1.0,
    }

    def __init__(self, scan_type: str = "partial") -> None:
        name = f"Scan({scan_type})"
        super().__init__(name)

        self._scan_type = scan_type
        self._duration = self.SCAN_DURATIONS.get(scan_type, 2.0)

        self._params = {
            "action": "scan",
            "scan_type": self._scan_type,
        }

    def update(self) -> Status:
        """Scan until duration elapsed."""
        if self.elapsed_time() >= self._duration:
            return Status.SUCCESS
        return Status.RUNNING


class StopAction(ActionNode):
    """
    Stop all current motion. Instant action.
    """

    def __init__(self) -> None:
        super().__init__("Stop")
        self._params = {"action": "stop"}

    def update(self) -> Status:
        """Stop is instant."""
        return Status.SUCCESS
