"""
Murph - Memory Types
Data models for different types of memories.
"""

from dataclasses import dataclass, field
from typing import Any
import time
import uuid


@dataclass
class PersonMemory:
    """
    Memory of a person Murph has encountered.

    Tracks familiarity, interactions, and sentiment over time.
    Familiarity grows with repeated sightings and interactions.

    Attributes:
        person_id: Unique identifier from perception/face recognition
        name: Known name if familiar, None otherwise
        is_familiar: True when familiarity_score >= 50
        familiarity_score: 0-100, grows with interactions
        first_seen: Timestamp of first encounter
        last_seen: Timestamp of most recent encounter
        interaction_count: Number of distinct interactions
        sentiment: -1.0 (negative) to 1.0 (positive) relationship quality
        tags: Descriptive tags like "plays_with_me", "pets_me"
    """

    person_id: str
    name: str | None = None
    familiarity_score: float = 0.0
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    interaction_count: int = 0
    sentiment: float = 0.0
    tags: set[str] = field(default_factory=set)

    @property
    def is_familiar(self) -> bool:
        """Person is considered familiar when familiarity >= 50."""
        return self.familiarity_score >= 50.0

    def record_sighting(self, interaction: bool = False) -> None:
        """
        Record seeing this person.

        Args:
            interaction: Whether this was an interaction (vs just seeing them)
        """
        self.last_seen = time.time()
        if interaction:
            self.interaction_count += 1

    def adjust_sentiment(self, delta: float) -> None:
        """
        Adjust sentiment towards this person.

        Args:
            delta: Amount to adjust (-1.0 to 1.0 range maintained)
        """
        self.sentiment = max(-1.0, min(1.0, self.sentiment + delta))

    def add_tag(self, tag: str) -> None:
        """Add a descriptive tag."""
        self.tags.add(tag)

    def get_state(self) -> dict[str, Any]:
        """Get serializable state for persistence."""
        return {
            "person_id": self.person_id,
            "name": self.name,
            "familiarity_score": self.familiarity_score,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "interaction_count": self.interaction_count,
            "sentiment": self.sentiment,
            "tags": list(self.tags),
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "PersonMemory":
        """Create a PersonMemory from saved state."""
        memory = cls(
            person_id=state["person_id"],
            name=state.get("name"),
            familiarity_score=state.get("familiarity_score", 0.0),
            first_seen=state.get("first_seen", time.time()),
            last_seen=state.get("last_seen", time.time()),
            interaction_count=state.get("interaction_count", 0),
            sentiment=state.get("sentiment", 0.0),
        )
        memory.tags = set(state.get("tags", []))
        return memory

    def __str__(self) -> str:
        familiar_str = "familiar" if self.is_familiar else "stranger"
        name_str = f" ({self.name})" if self.name else ""
        return (
            f"Person[{self.person_id}{name_str}]: {familiar_str}, "
            f"score={self.familiarity_score:.1f}, interactions={self.interaction_count}"
        )


@dataclass
class ObjectMemory:
    """
    Memory of an object Murph has encountered.

    Tracks when and where objects were seen.

    Attributes:
        object_id: Unique identifier
        object_type: Category like "ball", "cup", "unknown"
        first_seen: Timestamp of first encounter
        last_seen: Timestamp of most recent encounter
        times_seen: Number of times this object was seen
        last_position: Relative position (x, y) if known
        interesting: Whether this object has been investigated
    """

    object_id: str
    object_type: str
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    times_seen: int = 1
    last_position: tuple[float, float] | None = None
    interesting: bool = False

    def record_sighting(self, position: tuple[float, float] | None = None) -> None:
        """
        Record seeing this object.

        Args:
            position: Current position if known
        """
        self.last_seen = time.time()
        self.times_seen += 1
        if position is not None:
            self.last_position = position

    def mark_interesting(self) -> None:
        """Mark this object as having been investigated."""
        self.interesting = True

    def get_state(self) -> dict[str, Any]:
        """Get serializable state for persistence."""
        return {
            "object_id": self.object_id,
            "object_type": self.object_type,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "times_seen": self.times_seen,
            "last_position": self.last_position,
            "interesting": self.interesting,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "ObjectMemory":
        """Create an ObjectMemory from saved state."""
        pos = state.get("last_position")
        return cls(
            object_id=state["object_id"],
            object_type=state.get("object_type", "unknown"),
            first_seen=state.get("first_seen", time.time()),
            last_seen=state.get("last_seen", time.time()),
            times_seen=state.get("times_seen", 1),
            last_position=tuple(pos) if pos else None,
            interesting=state.get("interesting", False),
        )

    def __str__(self) -> str:
        pos_str = f" at {self.last_position}" if self.last_position else ""
        return f"Object[{self.object_id}]: {self.object_type}, seen={self.times_seen}{pos_str}"


@dataclass
class EventMemory:
    """
    Memory of a discrete event or interaction.

    Events have a strength that decays over time, eventually being forgotten.

    Attributes:
        event_id: Unique identifier (UUID)
        event_type: Category like "greeting", "petting", "play", "bump"
        timestamp: When the event occurred
        participants: List of person_ids involved
        objects: List of object_ids involved
        outcome: "positive", "negative", or "neutral"
        strength: 0-1, starts at 1.0 and decays over time
    """

    event_type: str
    timestamp: float = field(default_factory=time.time)
    participants: list[str] = field(default_factory=list)
    objects: list[str] = field(default_factory=list)
    outcome: str = "neutral"
    strength: float = 1.0
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def decay(self, amount: float) -> None:
        """
        Decay the memory strength.

        Args:
            amount: Amount to decay (will be clamped to 0)
        """
        self.strength = max(0.0, self.strength - amount)

    @property
    def is_positive(self) -> bool:
        """Check if this was a positive event."""
        return self.outcome == "positive"

    @property
    def is_negative(self) -> bool:
        """Check if this was a negative event."""
        return self.outcome == "negative"

    def get_state(self) -> dict[str, Any]:
        """Get serializable state for persistence."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "participants": self.participants.copy(),
            "objects": self.objects.copy(),
            "outcome": self.outcome,
            "strength": self.strength,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "EventMemory":
        """Create an EventMemory from saved state."""
        return cls(
            event_id=state.get("event_id", str(uuid.uuid4())),
            event_type=state["event_type"],
            timestamp=state.get("timestamp", time.time()),
            participants=state.get("participants", []),
            objects=state.get("objects", []),
            outcome=state.get("outcome", "neutral"),
            strength=state.get("strength", 1.0),
        )

    def __str__(self) -> str:
        age_seconds = time.time() - self.timestamp
        if age_seconds < 60:
            age_str = f"{age_seconds:.0f}s ago"
        elif age_seconds < 3600:
            age_str = f"{age_seconds / 60:.0f}m ago"
        else:
            age_str = f"{age_seconds / 3600:.1f}h ago"
        return (
            f"Event[{self.event_type}]: {self.outcome}, "
            f"strength={self.strength:.2f}, {age_str}"
        )
