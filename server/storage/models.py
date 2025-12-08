"""
Murph - SQLAlchemy Models for Long-Term Memory
Defines database schema for people, objects, events, and face embeddings.
"""

from datetime import datetime
from typing import Any

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
    first_seen: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_seen: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

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
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
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
    first_seen: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_seen: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

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
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

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
