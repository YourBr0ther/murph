"""
Murph - Short-Term Memory
Stores recent memories of people, objects, and events with time-based decay.
"""

from collections import deque
from typing import Any
import time

from .memory_types import PersonMemory, ObjectMemory, EventMemory


class ShortTermMemory:
    """
    Short-term memory with time-based decay.

    Stores memories of people, objects, and events. Event memories
    decay over time and are pruned when strength falls below threshold.
    Person familiarity grows with interactions and decays slowly.

    Attributes:
        event_decay_rate: Points per minute for event decay
        event_threshold: Minimum strength before event is forgotten
        max_events: Maximum events to store (ring buffer)
        familiarity_growth: Points added per interaction
        familiarity_decay: Points lost per hour (for non-familiar)
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
        Initialize short-term memory.

        Args:
            event_decay_rate: How fast events decay (points/minute)
            event_threshold: Below this strength, events are forgotten
            max_events: Maximum events to keep in memory
            familiarity_growth: How much familiarity grows per interaction
            familiarity_decay: How fast familiarity decays (points/hour)
        """
        self._event_decay_rate = event_decay_rate
        self._event_threshold = event_threshold
        self._max_events = max_events
        self._familiarity_growth = familiarity_growth
        self._familiarity_decay = familiarity_decay

        self._people: dict[str, PersonMemory] = {}
        self._objects: dict[str, ObjectMemory] = {}
        self._events: deque[EventMemory] = deque(maxlen=max_events)

        self._last_update_time: float = time.time()

    def update(self, delta_seconds: float | None = None) -> None:
        """
        Update memory decay.

        Args:
            delta_seconds: Time elapsed. If None, calculates from last update.
        """
        if delta_seconds is None:
            current_time = time.time()
            delta_seconds = current_time - self._last_update_time
            self._last_update_time = current_time
        else:
            self._last_update_time = time.time()

        # Decay event strengths
        decay_amount = (self._event_decay_rate / 60.0) * delta_seconds
        for event in self._events:
            event.decay(decay_amount)

        # Prune weak events
        surviving_events = [e for e in self._events if e.strength > self._event_threshold]
        self._events = deque(surviving_events, maxlen=self._max_events)

        # Decay person familiarity (very slow, only for non-familiar people)
        # Only apply decay for significant time intervals (> 60 seconds)
        if delta_seconds > 60:
            familiarity_decay = (self._familiarity_decay / 3600.0) * delta_seconds
            for person in self._people.values():
                # Don't decay people who are already familiar (family members)
                if not person.is_familiar:
                    person.familiarity_score = max(
                        0.0, person.familiarity_score - familiarity_decay
                    )

    def record_person_seen(
        self,
        person_id: str,
        is_familiar: bool = False,
        distance: float | None = None,
        interaction: bool = False,
    ) -> PersonMemory:
        """
        Record seeing a person.

        Args:
            person_id: Unique identifier for the person
            is_familiar: Whether perception says this person is familiar
            distance: Distance to person in cm
            interaction: Whether this is an interaction (vs just seeing)

        Returns:
            The person's memory record
        """
        if person_id in self._people:
            person = self._people[person_id]
            person.record_sighting(interaction=interaction)
        else:
            person = PersonMemory(person_id=person_id)
            person.record_sighting(interaction=interaction)
            self._people[person_id] = person

        # Increase familiarity based on proximity and interaction
        familiarity_increase = self._familiarity_growth
        if distance is not None and distance < 50:
            familiarity_increase *= 1.5  # Bonus for being close
        if interaction:
            familiarity_increase *= 2.0  # Double for actual interaction

        person.familiarity_score = min(100.0, person.familiarity_score + familiarity_increase)

        return person

    def record_object_seen(
        self,
        object_id: str,
        object_type: str = "unknown",
        position: tuple[float, float] | None = None,
    ) -> ObjectMemory:
        """
        Record seeing an object.

        Args:
            object_id: Unique identifier for the object
            object_type: Type/category of the object
            position: Position (x, y) if known

        Returns:
            The object's memory record
        """
        if object_id in self._objects:
            obj = self._objects[object_id]
            obj.record_sighting(position=position)
        else:
            obj = ObjectMemory(
                object_id=object_id,
                object_type=object_type,
                last_position=position,
            )
            self._objects[object_id] = obj

        return obj

    def record_event(
        self,
        event_type: str,
        participants: list[str] | None = None,
        objects: list[str] | None = None,
        outcome: str = "neutral",
    ) -> EventMemory:
        """
        Record an event.

        Args:
            event_type: Type of event (greeting, petting, play, etc.)
            participants: List of person_ids involved
            objects: List of object_ids involved
            outcome: "positive", "negative", or "neutral"

        Returns:
            The created event memory
        """
        event = EventMemory(
            event_type=event_type,
            participants=participants or [],
            objects=objects or [],
            outcome=outcome,
        )
        self._events.append(event)

        # Update sentiment for participants based on outcome
        sentiment_delta = {"positive": 0.1, "negative": -0.1, "neutral": 0.0}.get(
            outcome, 0.0
        )
        for person_id in event.participants:
            if person_id in self._people:
                self._people[person_id].adjust_sentiment(sentiment_delta)

        return event

    def get_person(self, person_id: str) -> PersonMemory | None:
        """Get memory of a specific person."""
        return self._people.get(person_id)

    def get_object(self, object_id: str) -> ObjectMemory | None:
        """Get memory of a specific object."""
        return self._objects.get(object_id)

    def get_all_people(self) -> list[PersonMemory]:
        """Get all remembered people."""
        return list(self._people.values())

    def get_familiar_people(self) -> list[PersonMemory]:
        """Get all people with familiarity >= 50."""
        return [p for p in self._people.values() if p.is_familiar]

    def is_person_familiar(self, person_id: str) -> bool:
        """Check if a person is familiar (from memory)."""
        person = self._people.get(person_id)
        return person.is_familiar if person else False

    def get_person_sentiment(self, person_id: str) -> float:
        """Get sentiment towards a person (0 if unknown)."""
        person = self._people.get(person_id)
        return person.sentiment if person else 0.0

    def get_recent_events(self, limit: int = 10) -> list[EventMemory]:
        """
        Get recent events, sorted by recency.

        Args:
            limit: Maximum number of events to return

        Returns:
            List of recent events, newest first
        """
        events = sorted(self._events, key=lambda e: e.timestamp, reverse=True)
        return events[:limit]

    def get_events_by_type(self, event_type: str) -> list[EventMemory]:
        """Get all events of a specific type."""
        return [e for e in self._events if e.event_type == event_type]

    def get_events_with_participant(
        self, person_id: str, limit: int = 20
    ) -> list[EventMemory]:
        """
        Get events involving a specific participant.

        Args:
            person_id: The person's ID
            limit: Maximum number of events to return

        Returns:
            List of events involving this person, newest first
        """
        events = [e for e in self._events if person_id in e.participants]
        events.sort(key=lambda e: e.timestamp, reverse=True)
        return events[:limit]

    def was_event_recent(self, event_type: str, within_seconds: float = 60.0) -> bool:
        """
        Check if an event of this type happened recently.

        Args:
            event_type: Type of event to check
            within_seconds: How far back to look

        Returns:
            True if matching event found within timeframe
        """
        cutoff = time.time() - within_seconds
        for event in self._events:
            if event.event_type == event_type and event.timestamp >= cutoff:
                return True
        return False

    def get_recent_event_types(self, within_seconds: float = 60.0) -> list[str]:
        """
        Get types of events that happened recently.

        Args:
            within_seconds: How far back to look

        Returns:
            List of unique event types
        """
        cutoff = time.time() - within_seconds
        types = set()
        for event in self._events:
            if event.timestamp >= cutoff:
                types.add(event.event_type)
        return list(types)

    def clear(self) -> None:
        """Clear all memories."""
        self._people.clear()
        self._objects.clear()
        self._events.clear()

    def get_state(self) -> dict[str, Any]:
        """Get serializable state for persistence."""
        return {
            "people": {pid: p.get_state() for pid, p in self._people.items()},
            "objects": {oid: o.get_state() for oid, o in self._objects.items()},
            "events": [e.get_state() for e in self._events],
            "config": {
                "event_decay_rate": self._event_decay_rate,
                "event_threshold": self._event_threshold,
                "max_events": self._max_events,
                "familiarity_growth": self._familiarity_growth,
                "familiarity_decay": self._familiarity_decay,
            },
            "last_update_time": self._last_update_time,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "ShortTermMemory":
        """Create a ShortTermMemory from saved state."""
        config = state.get("config", {})
        memory = cls(
            event_decay_rate=config.get("event_decay_rate", 0.5),
            event_threshold=config.get("event_threshold", 0.1),
            max_events=config.get("max_events", 50),
            familiarity_growth=config.get("familiarity_growth", 1.0),
            familiarity_decay=config.get("familiarity_decay", 0.1),
        )

        # Restore people
        for pid, pstate in state.get("people", {}).items():
            memory._people[pid] = PersonMemory.from_state(pstate)

        # Restore objects
        for oid, ostate in state.get("objects", {}).items():
            memory._objects[oid] = ObjectMemory.from_state(ostate)

        # Restore events
        for estate in state.get("events", []):
            memory._events.append(EventMemory.from_state(estate))

        memory._last_update_time = state.get("last_update_time", time.time())
        return memory

    def summary(self) -> dict[str, Any]:
        """Get a summary for logging/debugging."""
        return {
            "people_count": len(self._people),
            "familiar_count": len(self.get_familiar_people()),
            "objects_count": len(self._objects),
            "events_count": len(self._events),
            "recent_events": [e.event_type for e in self.get_recent_events(5)],
        }

    def __str__(self) -> str:
        summary = self.summary()
        return (
            f"ShortTermMemory: {summary['people_count']} people "
            f"({summary['familiar_count']} familiar), "
            f"{summary['objects_count']} objects, "
            f"{summary['events_count']} events"
        )
