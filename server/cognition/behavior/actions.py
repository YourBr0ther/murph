"""
Murph - Behavior Tree Action Nodes
Low-level robot actions for behavior tree execution.
"""

import time
from typing import Any, Callable

import py_trees
import py_trees.blackboard
from py_trees.common import Access, Status


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


class SpeakAction(ActionNode):
    """
    Speak a phrase using TTS.

    Unlike PlaySoundAction which plays pre-recorded files, SpeakAction
    synthesizes speech dynamically using the TTS API.

    Voice personality: Wall-E/BMO/BB-8 style - mostly beeps and boops
    with occasional simple words.

    Args:
        text: Text or phrase key to speak (e.g., "greeting", "happy", or raw text)
        emotion: Emotional tone (uses current mood if None)
        wait_for_completion: Block until speech estimated complete
    """

    # Phrase mapping for common robot utterances (Wall-E/BMO style)
    PHRASES: dict[str, str] = {
        "greeting": "beep boop, hello!",
        "happy": "wheee!",
        "sad": "aww...",
        "curious": "hmm?",
        "scared": "eep!",
        "affection": "I like you",
        "playful": "boop boop boop!",
        "sleepy": "zzz...",
        "alert": "oh!",
        "confused": "buh?",
        "excited": "ooh ooh!",
        "goodbye": "bye bye!",
        "yes": "uh huh!",
        "no": "nuh uh",
        "thanks": "beep!",
    }

    # Estimated durations for phrases (seconds)
    PHRASE_DURATIONS: dict[str, float] = {
        "greeting": 1.5,
        "happy": 0.8,
        "sad": 1.0,
        "curious": 0.6,
        "scared": 0.5,
        "affection": 1.2,
        "playful": 1.0,
        "sleepy": 1.5,
        "alert": 0.4,
        "confused": 0.5,
        "excited": 0.8,
        "goodbye": 1.0,
        "yes": 0.6,
        "no": 0.5,
        "thanks": 0.4,
    }
    DEFAULT_DURATION = 1.0

    def __init__(
        self,
        text: str,
        emotion: str | None = None,
        wait_for_completion: bool = True,
    ) -> None:
        # Truncate long text for node name
        display_text = text[:20] + "..." if len(text) > 20 else text
        name = f"Speak({display_text})"
        super().__init__(name)

        # Resolve phrase key to text if it's a known phrase
        self._text = self.PHRASES.get(text, text)
        self._phrase_key = text if text in self.PHRASES else None
        self._emotion = emotion
        self._wait = wait_for_completion

        # Calculate duration: use phrase duration or estimate from text length
        if self._phrase_key:
            self._duration = self.PHRASE_DURATIONS.get(
                self._phrase_key, self.DEFAULT_DURATION
            )
        else:
            # Estimate ~100ms per character
            self._duration = max(0.5, len(self._text) * 0.1)

        self._params = {
            "action": "speak",
            "text": self._text,
            "emotion": self._emotion,
            "phrase_key": self._phrase_key,
        }

    def update(self) -> Status:
        """Wait for estimated speech duration to complete."""
        if self._wait and self.elapsed_time() < self._duration:
            return Status.RUNNING
        return Status.SUCCESS


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


class NavigateToLandmarkAction(ActionNode):
    """
    Navigate toward a target landmark using the spatial map.

    This action reads the spatial map from the blackboard to find a path
    to the target. It issues movement commands based on the estimated
    distance from the landmark connection graph.

    Since the robot has no precise positioning, navigation is approximate:
    - Finds path via BFS on landmark graph
    - Estimates duration from connection distances
    - Moves forward for that duration
    - Perception/recognition handles arrival detection

    Args:
        target_type: Type of landmark ("home_base", "charging_station",
                    "safe_zone", "unfamiliar_zone") or specific landmark type
        target_id: Specific landmark ID (overrides type)
        timeout: Maximum duration before failing (seconds)
        speed: Movement speed (0.0-1.0)
    """

    # Estimated speed in cm/s at speed=1.0
    MAX_SPEED_CM_PER_SEC = 20.0

    def __init__(
        self,
        target_type: str = "home_base",
        target_id: str | None = None,
        timeout: float = 30.0,
        speed: float = 0.4,
    ) -> None:
        name = f"NavigateTo({target_type})"
        super().__init__(name)

        self._target_type = target_type
        self._target_id = target_id
        self._timeout = max(1.0, timeout)
        self._speed = max(0.1, min(1.0, speed))
        self._estimated_duration: float = 0.0
        self._path: list[str] | None = None

        self._params = {
            "action": "navigate",
            "target_type": self._target_type,
            "target_id": self._target_id,
            "speed": self._speed,
        }

    def setup(self, **kwargs: Any) -> None:
        """Setup blackboard access for spatial map."""
        self.blackboard = py_trees.blackboard.Client(name=self.name)
        self.blackboard.register_key(
            key="spatial_map",
            access=Access.READ
        )

    def initialise(self) -> None:
        """Plan path and calculate duration on first activation."""
        super().initialise()

        self._path = None
        self._estimated_duration = self._timeout  # Default to timeout

        # Try to get spatial map and plan path
        try:
            spatial_map = self.blackboard.spatial_map
            if spatial_map is None:
                return

            self._path = spatial_map.find_path_to(
                self._target_type, self._target_id
            )

            if self._path is not None and len(self._path) > 1:
                # Calculate total distance from path
                total_distance = 0.0
                for i in range(len(self._path) - 1):
                    current_lm = spatial_map.get_landmark(self._path[i])
                    if current_lm and self._path[i + 1] in current_lm.connections:
                        total_distance += current_lm.connections[self._path[i + 1]]

                # Estimate duration based on speed
                actual_speed = self._speed * self.MAX_SPEED_CM_PER_SEC
                if total_distance > 0 and actual_speed > 0:
                    self._estimated_duration = min(
                        total_distance / actual_speed,
                        self._timeout
                    )

                # Update params for callback
                self._params["estimated_distance"] = total_distance
                self._params["path_length"] = len(self._path)

        except (KeyError, AttributeError):
            # No spatial map available - will use timeout
            pass

        # Dispatch movement command
        self._params["action"] = "move"
        self._params["direction"] = "forward"
        self._params["duration"] = self._estimated_duration
        self._dispatch_action()

    def update(self) -> Status:
        """Continue navigation until estimated duration or timeout."""
        elapsed = self.elapsed_time()

        if elapsed >= self._timeout:
            return Status.FAILURE  # Took too long

        if self._path is None:
            # No path found - fail
            return Status.FAILURE

        if len(self._path) == 0:
            # Already at target
            return Status.SUCCESS

        if elapsed >= self._estimated_duration:
            return Status.SUCCESS

        return Status.RUNNING


class ReorientAction(ActionNode):
    """
    Attempt to regain position awareness by scanning for landmarks.

    Performs a series of scans in different directions, pausing between
    each to allow perception to process and potentially recognize landmarks.

    Args:
        max_attempts: Number of scan directions to try (default 4 for 360°)
        scan_duration: How long each scan takes (seconds)
    """

    def __init__(
        self,
        max_attempts: int = 4,
        scan_duration: float = 2.0,
    ) -> None:
        name = f"Reorient(attempts={max_attempts})"
        super().__init__(name)

        self._max_attempts = max(1, max_attempts)
        self._scan_duration = max(0.5, scan_duration)
        self._current_attempt = 0
        self._total_duration = self._max_attempts * (self._scan_duration + 1.0)

        self._params = {
            "action": "reorient",
            "scan_type": "full",
            "max_attempts": self._max_attempts,
        }

    def initialise(self) -> None:
        """Start reorientation sequence."""
        super().initialise()
        self._current_attempt = 0
        # Start with a scan
        self._params["action"] = "scan"
        self._params["scan_type"] = "partial"
        self._dispatch_action()

    def update(self) -> Status:
        """Progress through scan attempts."""
        elapsed = self.elapsed_time()

        if elapsed >= self._total_duration:
            return Status.SUCCESS  # Done all attempts

        # Calculate which attempt we're on
        time_per_attempt = self._scan_duration + 1.0  # scan + turn
        current = int(elapsed / time_per_attempt)

        if current > self._current_attempt and current < self._max_attempts:
            self._current_attempt = current
            # Turn 90 degrees between scans
            self._params["action"] = "turn"
            self._params["angle"] = 90.0
            self._params["speed"] = 0.4
            self._dispatch_action()

        return Status.RUNNING


class MoveTowardSafetyAction(ActionNode):
    """
    Emergency movement away from perceived danger.

    Moves backward quickly, then turns to face away from the danger.
    Used for flee/retreat behaviors when in dangerous zones or near edges.

    Args:
        retreat_duration: How long to back up (seconds)
        retreat_speed: Speed of backward movement (0.0-1.0)
    """

    def __init__(
        self,
        retreat_duration: float = 1.5,
        retreat_speed: float = 0.5,
    ) -> None:
        name = "MoveTowardSafety"
        super().__init__(name)

        self._retreat_duration = max(0.5, retreat_duration)
        self._retreat_speed = max(0.2, min(1.0, retreat_speed))
        self._turn_duration = 1.0  # Time to turn 180 degrees
        self._total_duration = self._retreat_duration + self._turn_duration

        self._params = {
            "action": "move",
            "direction": "backward",
            "speed": self._retreat_speed,
            "duration": self._retreat_duration,
        }

    def initialise(self) -> None:
        """Start retreat sequence."""
        super().initialise()
        self._dispatch_action()

    def update(self) -> Status:
        """Execute retreat then turn."""
        elapsed = self.elapsed_time()

        if elapsed >= self._total_duration:
            return Status.SUCCESS

        # Switch to turn after retreat
        if elapsed >= self._retreat_duration:
            if self._params["action"] != "turn":
                self._params = {
                    "action": "turn",
                    "angle": 180.0,
                    "speed": 0.5,
                }
                self._dispatch_action()

        return Status.RUNNING
