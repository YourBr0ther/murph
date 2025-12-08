"""
Murph - Memory System
Unified facade combining working and short-term memory.
"""

from typing import Any
import time

from .working_memory import WorkingMemory
from .short_term_memory import ShortTermMemory
from .memory_types import PersonMemory, ObjectMemory, EventMemory


class MemorySystem:
    """
    Unified memory system for Murph.

    Combines working memory (immediate context) and short-term memory
    (decaying memories of people, objects, events) into a single interface.

    Usage:
        memory = MemorySystem()

        # Process perception each cycle
        memory.process_perception(context, person_id="person_1")

        # Update decay
        memory.update(delta_seconds)

        # Query for behavior evaluation
        context_data = memory.get_behavior_context()
    """

    def __init__(
        self,
        event_decay_rate: float = 0.5,
        event_threshold: float = 0.1,
        max_events: int = 50,
        familiarity_growth: float = 1.0,
        familiarity_decay: float = 0.1,
    ) -> None:
        """
        Initialize the memory system.

        Args:
            event_decay_rate: How fast events decay (points/minute)
            event_threshold: Below this strength, events are forgotten
            max_events: Maximum events to keep in memory
            familiarity_growth: How much familiarity grows per interaction
            familiarity_decay: How fast familiarity decays (points/hour)
        """
        self.working = WorkingMemory()
        self.short_term = ShortTermMemory(
            event_decay_rate=event_decay_rate,
            event_threshold=event_threshold,
            max_events=max_events,
            familiarity_growth=familiarity_growth,
            familiarity_decay=familiarity_decay,
        )
        self._last_update_time: float = time.time()

    def update(self, delta_seconds: float | None = None) -> None:
        """
        Update memory systems (decay, pruning).

        Args:
            delta_seconds: Time elapsed. If None, calculates from last update.
        """
        if delta_seconds is None:
            current_time = time.time()
            delta_seconds = current_time - self._last_update_time
            self._last_update_time = current_time
        else:
            self._last_update_time = time.time()

        self.short_term.update(delta_seconds)

    def process_perception(
        self,
        person_detected: bool = False,
        person_id: str | None = None,
        person_is_familiar: bool = False,
        person_distance: float | None = None,
        objects_in_view: list[str] | None = None,
        is_being_petted: bool = False,
        is_being_held: bool = False,
    ) -> None:
        """
        Process perception data and update memories.

        Called each perception/cognition cycle to update memory state
        based on what Murph perceives.

        Args:
            person_detected: Whether a person is in view
            person_id: Identifier for detected person
            person_is_familiar: Whether perception thinks person is familiar
            person_distance: Distance to person in cm
            objects_in_view: List of object IDs/types visible
            is_being_petted: Whether being petted right now
            is_being_held: Whether being held right now
        """
        # Process person perception
        if person_detected and person_id:
            # Determine if this is an interaction
            is_interacting = (
                is_being_petted
                or is_being_held
                or (person_distance is not None and person_distance < 30)
            )

            # Record the sighting
            self.short_term.record_person_seen(
                person_id=person_id,
                is_familiar=person_is_familiar,
                distance=person_distance,
                interaction=is_interacting,
            )

            # Update working memory
            if is_interacting:
                self.working.set_active_person(person_id)
                self.working.set_attention(person_id)
        else:
            # No person detected, clear active person
            self.working.set_active_person(None)

        # Process object perception
        if objects_in_view:
            for obj_id in objects_in_view:
                # For now, use the ID as the type if it looks like a type
                self.short_term.record_object_seen(
                    object_id=obj_id,
                    object_type=obj_id,
                )
            self.working.active_objects = objects_in_view.copy()
        else:
            self.working.clear_active_objects()

        # Record significant events
        if is_being_petted:
            # Only record if not recently recorded
            if not self.short_term.was_event_recent("petting", within_seconds=5.0):
                participants = [person_id] if person_id else []
                self.short_term.record_event(
                    event_type="petting",
                    participants=participants,
                    outcome="positive",
                )

        if is_being_held:
            if not self.short_term.was_event_recent("held", within_seconds=10.0):
                participants = [person_id] if person_id else []
                self.short_term.record_event(
                    event_type="held",
                    participants=participants,
                    outcome="positive",
                )

    def record_behavior_start(self, behavior: str, goal: str | None = None) -> None:
        """
        Record starting a behavior.

        Args:
            behavior: Name of the behavior
            goal: The need/goal being addressed
        """
        self.working.start_behavior(behavior, goal)

    def record_behavior_end(self, result: str = "completed") -> None:
        """
        Record ending a behavior.

        Args:
            result: How it ended ("completed", "interrupted", "failed")
        """
        self.working.end_behavior(result)

    def record_event(
        self,
        event_type: str,
        participants: list[str] | None = None,
        objects: list[str] | None = None,
        outcome: str = "neutral",
    ) -> EventMemory:
        """
        Manually record an event.

        Args:
            event_type: Type of event
            participants: Person IDs involved
            objects: Object IDs involved
            outcome: "positive", "negative", or "neutral"

        Returns:
            The created event memory
        """
        return self.short_term.record_event(
            event_type=event_type,
            participants=participants,
            objects=objects,
            outcome=outcome,
        )

    def get_active_person(self) -> PersonMemory | None:
        """Get memory of the currently active person."""
        if self.working.active_person_id:
            return self.short_term.get_person(self.working.active_person_id)
        return None

    def get_active_person_name(self) -> str | None:
        """Get name of currently active person if known."""
        person = self.get_active_person()
        return person.name if person else None

    def is_active_person_familiar(self) -> bool:
        """Check if the currently active person is familiar."""
        person = self.get_active_person()
        return person.is_familiar if person else False

    def get_behavior_context(self) -> dict[str, Any]:
        """
        Get memory context for behavior evaluation.

        Returns:
            Dictionary with memory-derived context information
        """
        active_person = self.get_active_person()

        return {
            # Working memory context
            **self.working.get_context_summary(),
            # Person context
            "active_person_name": active_person.name if active_person else None,
            "active_person_familiar": active_person.is_familiar if active_person else False,
            "active_person_sentiment": active_person.sentiment if active_person else 0.0,
            "active_person_interactions": (
                active_person.interaction_count if active_person else 0
            ),
            # Memory summary
            "familiar_people_count": len(self.short_term.get_familiar_people()),
            "recent_event_types": self.short_term.get_recent_event_types(60.0),
        }

    def get_memory_triggers(self) -> dict[str, bool]:
        """
        Get memory-derived triggers for WorldContext.

        Returns:
            Dictionary of trigger name -> bool
        """
        active_person = self.get_active_person()
        recent_events = self.short_term.get_recent_event_types(60.0)

        return {
            "familiar_person_remembered": (
                active_person.is_familiar if active_person else False
            ),
            "positive_history": (
                active_person is not None and active_person.interaction_count >= 5
            ),
            "negative_sentiment": (
                active_person is not None and active_person.sentiment < -0.3
            ),
            "positive_sentiment": (
                active_person is not None and active_person.sentiment > 0.3
            ),
            "recently_greeted": "greeting" in recent_events,
            "recently_played": "play" in recent_events,
            "recently_petted": "petting" in recent_events,
            "was_interrupted": self.working.was_interrupted,
        }

    def clear(self) -> None:
        """Clear all memory state."""
        self.working.clear()
        self.short_term.clear()

    def get_state(self) -> dict[str, Any]:
        """Get serializable state for persistence."""
        return {
            "working": self.working.get_state(),
            "short_term": self.short_term.get_state(),
            "last_update_time": self._last_update_time,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "MemorySystem":
        """Create a MemorySystem from saved state."""
        # Get config from short_term state if available
        st_config = state.get("short_term", {}).get("config", {})

        system = cls(
            event_decay_rate=st_config.get("event_decay_rate", 0.5),
            event_threshold=st_config.get("event_threshold", 0.1),
            max_events=st_config.get("max_events", 50),
            familiarity_growth=st_config.get("familiarity_growth", 1.0),
            familiarity_decay=st_config.get("familiarity_decay", 0.1),
        )

        # Restore subsystem states
        if "working" in state:
            system.working = WorkingMemory.from_state(state["working"])
        if "short_term" in state:
            system.short_term = ShortTermMemory.from_state(state["short_term"])

        system._last_update_time = state.get("last_update_time", time.time())
        return system

    def summary(self) -> dict[str, Any]:
        """Get a summary for logging/debugging."""
        return {
            "working": {
                "current_behavior": self.working.current_behavior,
                "active_person": self.working.active_person_id,
                "attention": self.working.attention_target,
            },
            "short_term": self.short_term.summary(),
        }

    def __str__(self) -> str:
        return f"MemorySystem({self.working}, {self.short_term})"
