"""
Murph - Working Memory
Holds immediate cognitive context, updated each cognition cycle.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Any
import time


@dataclass
class WorkingMemory:
    """
    Working memory for immediate cognitive context.

    Tracks what Murph is currently doing, what just happened,
    and what's in focus. Updated synchronously with each cognition cycle.

    Attributes:
        current_behavior: Name of currently executing behavior
        current_goal: The need being satisfied by current behavior
        active_person_id: Person we're currently interacting with
        active_objects: Objects currently in focus
        previous_behavior: What we just finished doing
        behavior_start_time: When current behavior started
        behavior_history: Recent behavior names (ring buffer)
        attention_target: What we're currently looking at
        attention_start_time: When we started paying attention
        was_interrupted: Whether last behavior was interrupted
        interruption_reason: Why we were interrupted
    """

    # Current focus
    current_behavior: str | None = None
    current_goal: str | None = None

    # Active entities
    active_person_id: str | None = None
    active_objects: list[str] = field(default_factory=list)

    # Context continuity
    previous_behavior: str | None = None
    behavior_start_time: float | None = None
    behavior_history: deque[str] = field(default_factory=lambda: deque(maxlen=5))

    # Attention state
    attention_target: str | None = None
    attention_start_time: float | None = None

    # Interruption tracking
    was_interrupted: bool = False
    interruption_reason: str | None = None

    def start_behavior(self, name: str, goal: str | None = None) -> None:
        """
        Record starting a new behavior.

        Args:
            name: Name of the behavior starting
            goal: The need/goal this behavior addresses
        """
        # Save previous if we had one
        if self.current_behavior:
            self.previous_behavior = self.current_behavior
            self.behavior_history.append(self.current_behavior)

        self.current_behavior = name
        self.current_goal = goal
        self.behavior_start_time = time.time()
        self.was_interrupted = False
        self.interruption_reason = None

    def end_behavior(self, result: str = "completed") -> None:
        """
        Record ending the current behavior.

        Args:
            result: How it ended ("completed", "interrupted", "failed")
        """
        if self.current_behavior:
            self.previous_behavior = self.current_behavior
            self.behavior_history.append(self.current_behavior)

        was_interrupted = result == "interrupted"
        self.was_interrupted = was_interrupted
        # Only set interruption_reason if not already set (interrupt_behavior sets it)
        if was_interrupted and self.interruption_reason is None:
            self.interruption_reason = result

        self.current_behavior = None
        self.current_goal = None
        self.behavior_start_time = None

    def interrupt_behavior(self, reason: str) -> None:
        """
        Record behavior interruption.

        Args:
            reason: Why the behavior was interrupted
        """
        self.was_interrupted = True
        self.interruption_reason = reason
        self.end_behavior(result="interrupted")

    def set_attention(self, target: str | None) -> None:
        """
        Set what we're paying attention to.

        Args:
            target: Target identifier (person_id, object_id, or descriptive)
        """
        if target != self.attention_target:
            self.attention_target = target
            self.attention_start_time = time.time() if target else None

    def clear_attention(self) -> None:
        """Clear attention state."""
        self.attention_target = None
        self.attention_start_time = None

    def set_active_person(self, person_id: str | None) -> None:
        """
        Set the person we're currently interacting with.

        Args:
            person_id: The person's identifier, or None to clear
        """
        self.active_person_id = person_id

    def add_active_object(self, object_id: str) -> None:
        """Add an object to our current focus."""
        if object_id not in self.active_objects:
            self.active_objects.append(object_id)

    def remove_active_object(self, object_id: str) -> None:
        """Remove an object from our current focus."""
        if object_id in self.active_objects:
            self.active_objects.remove(object_id)

    def clear_active_objects(self) -> None:
        """Clear all active objects."""
        self.active_objects.clear()

    @property
    def attention_duration(self) -> float:
        """How long we've been paying attention to current target."""
        if self.attention_start_time is None:
            return 0.0
        return time.time() - self.attention_start_time

    @property
    def behavior_duration(self) -> float:
        """How long current behavior has been running."""
        if self.behavior_start_time is None:
            return 0.0
        return time.time() - self.behavior_start_time

    def get_recent_behaviors(self, count: int = 5) -> list[str]:
        """
        Get list of recent behaviors.

        Args:
            count: Maximum number to return

        Returns:
            List of behavior names, most recent first
        """
        recent = list(self.behavior_history)
        recent.reverse()
        return recent[:count]

    def was_doing(self, behavior: str) -> bool:
        """Check if we were recently doing a specific behavior."""
        return behavior in self.behavior_history or behavior == self.previous_behavior

    def get_context_summary(self) -> dict[str, Any]:
        """
        Get a summary of current context for behavior evaluation.

        Returns:
            Dictionary with relevant context information
        """
        return {
            "current_behavior": self.current_behavior,
            "current_goal": self.current_goal,
            "behavior_duration": self.behavior_duration,
            "active_person_id": self.active_person_id,
            "active_objects": self.active_objects.copy(),
            "attention_target": self.attention_target,
            "attention_duration": self.attention_duration,
            "previous_behavior": self.previous_behavior,
            "was_interrupted": self.was_interrupted,
            "recent_behaviors": self.get_recent_behaviors(),
        }

    def clear(self) -> None:
        """Clear all working memory state."""
        self.current_behavior = None
        self.current_goal = None
        self.active_person_id = None
        self.active_objects.clear()
        self.previous_behavior = None
        self.behavior_start_time = None
        self.behavior_history.clear()
        self.attention_target = None
        self.attention_start_time = None
        self.was_interrupted = False
        self.interruption_reason = None

    def get_state(self) -> dict[str, Any]:
        """Get serializable state for persistence."""
        return {
            "current_behavior": self.current_behavior,
            "current_goal": self.current_goal,
            "active_person_id": self.active_person_id,
            "active_objects": self.active_objects.copy(),
            "previous_behavior": self.previous_behavior,
            "behavior_start_time": self.behavior_start_time,
            "behavior_history": list(self.behavior_history),
            "attention_target": self.attention_target,
            "attention_start_time": self.attention_start_time,
            "was_interrupted": self.was_interrupted,
            "interruption_reason": self.interruption_reason,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "WorkingMemory":
        """Create a WorkingMemory from saved state."""
        memory = cls(
            current_behavior=state.get("current_behavior"),
            current_goal=state.get("current_goal"),
            active_person_id=state.get("active_person_id"),
            active_objects=state.get("active_objects", []),
            previous_behavior=state.get("previous_behavior"),
            behavior_start_time=state.get("behavior_start_time"),
            attention_target=state.get("attention_target"),
            attention_start_time=state.get("attention_start_time"),
            was_interrupted=state.get("was_interrupted", False),
            interruption_reason=state.get("interruption_reason"),
        )
        memory.behavior_history = deque(
            state.get("behavior_history", []), maxlen=5
        )
        return memory

    def __str__(self) -> str:
        if self.current_behavior:
            return (
                f"WorkingMemory: doing '{self.current_behavior}' "
                f"({self.behavior_duration:.1f}s), "
                f"attention={self.attention_target}"
            )
        return "WorkingMemory: idle"
