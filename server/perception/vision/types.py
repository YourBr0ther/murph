"""
Murph - Vision Types
Data classes for face detection and recognition pipeline.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class DetectedFace:
    """
    A detected face in an image frame.

    Attributes:
        bbox: Bounding box as (x1, y1, x2, y2) in pixels
        confidence: Detection confidence score (0-1)
        landmarks: 5 facial landmarks (eyes, nose, mouth corners) or None
        face_image: Cropped and aligned face image (160x160 RGB) or None
    """

    bbox: tuple[float, float, float, float]
    confidence: float
    landmarks: np.ndarray | None = None
    face_image: np.ndarray | None = None

    @property
    def area(self) -> float:
        """Face bounding box area in pixels squared."""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)

    @property
    def center(self) -> tuple[float, float]:
        """Center point of face bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def width(self) -> float:
        """Width of bounding box."""
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        """Height of bounding box."""
        return self.bbox[3] - self.bbox[1]


@dataclass
class FaceEncoding:
    """
    Face embedding with metadata.

    Attributes:
        embedding: 128-dimensional face embedding (L2-normalized)
        quality_score: Estimated quality of the encoding (0-1)
        face: Source DetectedFace this encoding was generated from
    """

    embedding: np.ndarray
    quality_score: float
    face: DetectedFace

    def __post_init__(self) -> None:
        """Validate embedding dimensions."""
        if self.embedding.shape != (128,):
            raise ValueError(
                f"Expected 128-dim embedding, got shape {self.embedding.shape}"
            )

    def cosine_similarity(self, other: "FaceEncoding") -> float:
        """
        Compute cosine similarity with another encoding.

        Returns:
            Similarity score between -1 and 1 (higher = more similar)
        """
        norm_product = np.linalg.norm(self.embedding) * np.linalg.norm(other.embedding)
        if norm_product == 0:
            return 0.0
        return float(np.dot(self.embedding, other.embedding) / norm_product)


@dataclass
class RecognitionResult:
    """
    Result of face recognition attempt.

    Attributes:
        encoding: The face encoding that was matched
        person_id: Matched person's ID, or None if unknown
        confidence: Match confidence (cosine similarity, 0-1)
        is_new_person: True if this appears to be a previously unseen person
        temporary_id: Temporary tracking ID for unknown persons
    """

    encoding: FaceEncoding
    person_id: str | None
    confidence: float
    is_new_person: bool = False
    temporary_id: str | None = None

    @property
    def is_identified(self) -> bool:
        """True if the person was successfully identified."""
        return self.person_id is not None

    @property
    def tracking_id(self) -> str:
        """Return the best available ID for tracking (person_id or temporary_id)."""
        return self.person_id or self.temporary_id or "unknown"


@dataclass
class TrackedPerson:
    """
    A person being tracked across video frames.

    Attributes:
        temporary_id: UUID-based temporary identifier
        confirmed_person_id: Confirmed identity after multiple matches, or None
        last_bbox: Last known bounding box position
        last_encoding: Most recent face encoding
        match_history: Recent person_id match results for confirmation
        frames_tracked: Number of frames this person has been tracked
        first_seen_time: Timestamp when first detected
    """

    temporary_id: str
    confirmed_person_id: str | None = None
    last_bbox: tuple[float, float, float, float] | None = None
    last_encoding: FaceEncoding | None = None
    match_history: list[str | None] = field(default_factory=list)
    frames_tracked: int = 0
    first_seen_time: float = 0.0
    last_seen_frame: int = 0

    @property
    def is_confirmed(self) -> bool:
        """True if identity has been confirmed via multiple matches."""
        return self.confirmed_person_id is not None

    @property
    def best_id(self) -> str:
        """Return confirmed ID if available, otherwise temporary ID."""
        return self.confirmed_person_id or self.temporary_id

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "temporary_id": self.temporary_id,
            "confirmed_person_id": self.confirmed_person_id,
            "last_bbox": self.last_bbox,
            "frames_tracked": self.frames_tracked,
            "first_seen_time": self.first_seen_time,
            "is_confirmed": self.is_confirmed,
        }
