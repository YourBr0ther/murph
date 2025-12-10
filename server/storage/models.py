"""
Murph - SQLAlchemy Models for Long-Term Memory
Defines database schema for people, objects, events, and face embeddings.
"""

from datetime import UTC, datetime
from typing import Any


def utc_now() -> datetime:
    """Return current UTC time as timezone-aware datetime."""
    return datetime.now(UTC)

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


class PersonModel(Base):
    """
    Long-term memory of a person.

    Stored when:
    - familiarity_score >= 50 (is_familiar threshold)
    - OR name is set (explicitly identified)
    """

    __tablename__ = "persons"

    id: Mapped[int] = mapped_column(primary_key=True)
    person_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    name: Mapped[str | None] = mapped_column(String(128), nullable=True)

    # Relationship scores
    familiarity_score: Mapped[float] = mapped_column(Float, default=0.0)
    trust_score: Mapped[float] = mapped_column(Float, default=50.0)  # 0-100
    sentiment: Mapped[float] = mapped_column(Float, default=0.0)  # -1 to 1

    # Timestamps
    first_seen: Mapped[datetime] = mapped_column(DateTime, default=utc_now)
    last_seen: Mapped[datetime] = mapped_column(DateTime, default=utc_now)

    # Interaction tracking
    interaction_count: Mapped[int] = mapped_column(Integer, default=0)

    # Tags stored as JSON array
    tags: Mapped[list[str]] = mapped_column(JSON, default=list)

    # Relationships
    face_embeddings: Mapped[list["FaceEmbeddingModel"]] = relationship(
        back_populates="person", cascade="all, delete-orphan"
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for PersonMemory.from_state()."""
        return {
            "person_id": self.person_id,
            "name": self.name,
            "familiarity_score": self.familiarity_score,
            "trust_score": self.trust_score,
            "first_seen": self.first_seen.timestamp(),
            "last_seen": self.last_seen.timestamp(),
            "interaction_count": self.interaction_count,
            "sentiment": self.sentiment,
            "tags": self.tags or [],
        }


class FaceEmbeddingModel(Base):
    """
    FaceNet embedding (128-dim) for face recognition.

    Multiple embeddings per person allow better recognition
    under different angles/lighting conditions.
    """

    __tablename__ = "face_embeddings"

    id: Mapped[int] = mapped_column(primary_key=True)
    person_id: Mapped[int] = mapped_column(ForeignKey("persons.id"))

    # 128-dim FaceNet embedding stored as bytes (128 * 4 = 512 bytes for float32)
    embedding: Mapped[bytes] = mapped_column(LargeBinary(512))

    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)
    quality_score: Mapped[float] = mapped_column(Float, default=1.0)  # Image quality

    # Relationship
    person: Mapped["PersonModel"] = relationship(back_populates="face_embeddings")


class ObjectModel(Base):
    """
    Long-term memory of an interesting object.

    Stored when:
    - object.interesting == True (has been investigated)
    - OR times_seen >= threshold (frequently seen)
    """

    __tablename__ = "objects"

    id: Mapped[int] = mapped_column(primary_key=True)
    object_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    object_type: Mapped[str] = mapped_column(String(64))

    # Timestamps
    first_seen: Mapped[datetime] = mapped_column(DateTime, default=utc_now)
    last_seen: Mapped[datetime] = mapped_column(DateTime, default=utc_now)

    # Tracking
    times_seen: Mapped[int] = mapped_column(Integer, default=1)
    interesting: Mapped[bool] = mapped_column(Boolean, default=False)

    # Last known position (nullable)
    last_position_x: Mapped[float | None] = mapped_column(Float, nullable=True)
    last_position_y: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Additional data (extensible)
    extra_data: Mapped[dict] = mapped_column(JSON, default=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for ObjectMemory.from_state()."""
        pos = None
        if self.last_position_x is not None and self.last_position_y is not None:
            pos = (self.last_position_x, self.last_position_y)

        return {
            "object_id": self.object_id,
            "object_type": self.object_type,
            "first_seen": self.first_seen.timestamp(),
            "last_seen": self.last_seen.timestamp(),
            "times_seen": self.times_seen,
            "last_position": pos,
            "interesting": self.interesting,
        }


class EventModel(Base):
    """
    Long-term memory of a significant event.

    Stored when:
    - Event involves a familiar person
    - OR event has strong emotional valence (very positive/negative)
    - OR event is a milestone type (first_meeting, etc.)
    """

    __tablename__ = "events"

    id: Mapped[int] = mapped_column(primary_key=True)
    event_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    event_type: Mapped[str] = mapped_column(String(64), index=True)

    # When it happened
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=utc_now)

    # Participants and objects (stored as JSON arrays)
    participants: Mapped[list[str]] = mapped_column(JSON, default=list)
    objects: Mapped[list[str]] = mapped_column(JSON, default=list)

    # Outcome and significance
    outcome: Mapped[str] = mapped_column(String(32), default="neutral")
    significance: Mapped[float] = mapped_column(
        Float, default=1.0
    )  # 0-1, higher = more memorable

    # Description (optional, for LLM-generated summaries)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for EventMemory.from_state()."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp.timestamp(),
            "participants": self.participants or [],
            "objects": self.objects or [],
            "outcome": self.outcome,
            "strength": self.significance,  # Map significance -> strength
        }


class SpatialLandmarkModel(Base):
    """
    Long-term memory of a spatial landmark.

    Stored when:
    - confidence >= 0.6 (reliably recognizable)
    - OR times_visited >= 5 (frequently visited)
    - OR landmark_type is critical ("charging_station", "home_base", "edge")
    """

    __tablename__ = "spatial_landmarks"

    id: Mapped[int] = mapped_column(primary_key=True)
    landmark_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    landmark_type: Mapped[str] = mapped_column(String(32), index=True)
    name: Mapped[str | None] = mapped_column(String(128), nullable=True)

    # Timestamps
    first_seen: Mapped[datetime] = mapped_column(DateTime, default=utc_now)
    last_seen: Mapped[datetime] = mapped_column(DateTime, default=utc_now)

    # Usage tracking
    times_visited: Mapped[int] = mapped_column(Integer, default=1)
    confidence: Mapped[float] = mapped_column(Float, default=0.5)

    # Graph connections: {landmark_id: distance_cm}
    connections: Mapped[dict] = mapped_column(JSON, default=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for SpatialLandmark.from_state()."""
        return {
            "landmark_id": self.landmark_id,
            "landmark_type": self.landmark_type,
            "name": self.name,
            "first_seen": self.first_seen.timestamp(),
            "last_seen": self.last_seen.timestamp(),
            "times_visited": self.times_visited,
            "confidence": self.confidence,
            "connections": self.connections or {},
        }


class SpatialZoneModel(Base):
    """
    Long-term memory of a spatial zone.

    Stored when:
    - familiarity >= 0.5 (reasonably explored)
    - OR zone_type is critical ("charging_zone", "edge_zone")
    """

    __tablename__ = "spatial_zones"

    id: Mapped[int] = mapped_column(primary_key=True)
    zone_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    zone_type: Mapped[str] = mapped_column(String(32), index=True)
    name: Mapped[str | None] = mapped_column(String(128), nullable=True)

    # Reference landmark
    primary_landmark_id: Mapped[str] = mapped_column(String(64), index=True)

    # Experience-based scores
    safety_score: Mapped[float] = mapped_column(Float, default=0.5)
    familiarity: Mapped[float] = mapped_column(Float, default=0.0)

    # Events that occurred here
    associated_events: Mapped[list[str]] = mapped_column(JSON, default=list)

    last_visited: Mapped[datetime] = mapped_column(DateTime, default=utc_now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for SpatialZone.from_state()."""
        return {
            "zone_id": self.zone_id,
            "zone_type": self.zone_type,
            "name": self.name,
            "primary_landmark_id": self.primary_landmark_id,
            "safety_score": self.safety_score,
            "familiarity": self.familiarity,
            "associated_events": self.associated_events or [],
            "last_visited": self.last_visited.timestamp(),
        }


class SpatialObservationModel(Base):
    """
    Long-term memory of a significant spatial observation.

    Stored when:
    - Entity is important (familiar person or interesting object)
    - AND confidence >= 0.5
    """

    __tablename__ = "spatial_observations"

    id: Mapped[int] = mapped_column(primary_key=True)
    observation_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)

    # What was observed
    entity_type: Mapped[str] = mapped_column(String(16))  # "object" or "person"
    entity_id: Mapped[str] = mapped_column(String(64), index=True)

    # Where it was observed (relative to landmark)
    landmark_id: Mapped[str] = mapped_column(String(64), index=True)
    relative_direction: Mapped[float] = mapped_column(Float)  # degrees
    relative_distance: Mapped[float] = mapped_column(Float)  # cm

    # Metadata
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=utc_now)
    confidence: Mapped[float] = mapped_column(Float, default=0.5)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for SpatialObservation.from_state()."""
        return {
            "observation_id": self.observation_id,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "landmark_id": self.landmark_id,
            "relative_direction": self.relative_direction,
            "relative_distance": self.relative_distance,
            "timestamp": self.timestamp.timestamp(),
            "confidence": self.confidence,
        }


class InsightModel(Base):
    """
    LLM-generated insight stored in long-term memory.

    Types:
    - event_summary: Consolidated view of multiple events
    - relationship_narrative: Story about relationship with a person
    - behavior_reflection: Reflection on behavior outcomes

    Subject types:
    - person: Insight about a specific person
    - behavior: Insight about a behavior pattern
    - session: Insight about the current session
    - general: General observation
    """

    __tablename__ = "insights"

    id: Mapped[int] = mapped_column(primary_key=True)
    insight_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    insight_type: Mapped[str] = mapped_column(String(32), index=True)

    # Subject of the insight
    subject_type: Mapped[str] = mapped_column(String(32))
    subject_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)

    # The insight content
    content: Mapped[str] = mapped_column(Text)
    summary: Mapped[str] = mapped_column(String(256))

    # Source events that led to this insight
    source_event_ids: Mapped[list[str]] = mapped_column(JSON, default=list)

    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)
    confidence: Mapped[float] = mapped_column(Float, default=0.7)
    relevance_score: Mapped[float] = mapped_column(Float, default=1.0)

    # Tags for filtering
    tags: Mapped[list[str]] = mapped_column(JSON, default=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for InsightMemory.from_state()."""
        return {
            "insight_id": self.insight_id,
            "insight_type": self.insight_type,
            "subject_type": self.subject_type,
            "subject_id": self.subject_id,
            "content": self.content,
            "summary": self.summary,
            "source_event_ids": self.source_event_ids or [],
            "created_at": self.created_at.timestamp(),
            "confidence": self.confidence,
            "relevance_score": self.relevance_score,
            "tags": self.tags or [],
        }
