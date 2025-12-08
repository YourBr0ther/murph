"""
Murph - Face Recognizer
Orchestrates face detection, encoding, and memory-based identification.
"""

import logging
import uuid
from typing import TYPE_CHECKING

import numpy as np

from .face_detector import FaceDetector
from .face_encoder import FaceEncoder
from .types import DetectedFace, FaceEncoding, RecognitionResult

if TYPE_CHECKING:
    from server.cognition.memory.memory_system import MemorySystem

logger = logging.getLogger(__name__)


class FaceRecognizer:
    """
    Face recognition system integrating detection, encoding, and memory.

    Handles:
    - Full detection + encoding pipeline
    - Memory lookup for known faces
    - Temporary ID assignment for unknown faces
    - Match confirmation via consecutive matches
    - Quality-based filtering for embedding storage

    Usage:
        recognizer = FaceRecognizer(memory_system)
        results = await recognizer.process_frame(frame)
    """

    def __init__(
        self,
        memory_system: "MemorySystem | None" = None,
        device: str | None = None,
        match_threshold: float = 0.6,
        quality_threshold: float = 0.5,
        min_matches_for_confirmation: int = 3,
    ) -> None:
        """
        Initialize face recognizer.

        Args:
            memory_system: MemorySystem for face lookups. If None, recognition is disabled.
            device: Compute device for models ('cuda' or 'cpu')
            match_threshold: Cosine similarity threshold for positive match (default 0.6)
            quality_threshold: Minimum quality to save embeddings (default 0.5)
            min_matches_for_confirmation: Consecutive matches needed for confirmation (default 3)
        """
        self._memory = memory_system
        self._match_threshold = match_threshold
        self._quality_threshold = quality_threshold
        self._confirmation_count = min_matches_for_confirmation

        # Initialize detector and encoder
        self.detector = FaceDetector(device=device)
        self.encoder = FaceEncoder(device=device)

        # Match history for confirmation logic: temp_id -> list of recent person_ids
        self._match_history: dict[str, list[str | None]] = {}

        # Temporary ID mapping: temp_id -> last_bbox for tracking
        self._temp_id_tracking: dict[str, tuple[float, float, float, float]] = {}

        logger.info(
            f"FaceRecognizer initialized (threshold={match_threshold}, "
            f"confirmation={min_matches_for_confirmation})"
        )

    async def process_frame(
        self,
        frame: np.ndarray,
    ) -> list[RecognitionResult]:
        """
        Process a video frame for face recognition.

        Args:
            frame: RGB image as numpy array (H, W, 3)

        Returns:
            List of RecognitionResult for each detected face
        """
        # Detect faces
        faces = self.detector.detect(frame, return_aligned=True)

        if not faces:
            return []

        # Encode faces
        encodings = self.encoder.encode_batch(faces)

        # Identify each face
        results = []
        for encoding in encodings:
            result = await self._identify_and_track(encoding)
            results.append(result)

        logger.debug(f"Processed frame: {len(results)} faces recognized")
        return results

    async def _identify_and_track(
        self,
        encoding: FaceEncoding,
    ) -> RecognitionResult:
        """
        Identify a face and track it across frames.

        Args:
            encoding: Face encoding to identify

        Returns:
            RecognitionResult with person_id if matched
        """
        # Try to match with existing tracked faces by position
        temp_id = self._find_or_assign_temp_id(encoding.face.bbox)

        # Look up in memory
        person_id, confidence = await self.identify_face(encoding)

        # Update match history
        confirmed_id = self._update_match_history(temp_id, person_id)

        # Determine if this is a new person
        is_new = person_id is None and self._is_likely_new_person(temp_id)

        return RecognitionResult(
            encoding=encoding,
            person_id=confirmed_id,
            confidence=confidence if confirmed_id else 0.0,
            is_new_person=is_new,
            temporary_id=temp_id,
        )

    async def identify_face(
        self,
        encoding: FaceEncoding,
    ) -> tuple[str | None, float]:
        """
        Identify a face by looking up in memory.

        Args:
            encoding: Face encoding to identify

        Returns:
            (person_id, confidence) or (None, 0.0) if unknown
        """
        if self._memory is None:
            return None, 0.0

        person_id, similarity = await self._memory.lookup_person_by_face(
            encoding.embedding, self._match_threshold
        )

        return person_id, similarity

    def _find_or_assign_temp_id(
        self,
        bbox: tuple[float, float, float, float],
        iou_threshold: float = 0.3,
    ) -> str:
        """
        Find existing temp_id by position or assign new one.

        Uses bounding box IoU to associate detections across frames.
        """
        best_iou = 0.0
        best_id = None

        for temp_id, last_bbox in self._temp_id_tracking.items():
            iou = self.detector.compute_iou(bbox, last_bbox)
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_id = temp_id

        if best_id:
            # Update position
            self._temp_id_tracking[best_id] = bbox
            return best_id

        # Assign new temp_id
        new_id = f"unknown_{uuid.uuid4().hex[:8]}"
        self._temp_id_tracking[new_id] = bbox
        self._match_history[new_id] = []
        return new_id

    def _update_match_history(
        self,
        temp_id: str,
        matched_person_id: str | None,
    ) -> str | None:
        """
        Update match history and return confirmed person_id.

        Requires min_matches_for_confirmation consecutive matches
        to the same person_id before confirming identity.

        Returns:
            Confirmed person_id if enough consecutive matches, else None
        """
        if temp_id not in self._match_history:
            self._match_history[temp_id] = []

        history = self._match_history[temp_id]
        history.append(matched_person_id)

        # Keep only recent matches (2x confirmation count)
        max_history = self._confirmation_count * 2
        if len(history) > max_history:
            history.pop(0)

        # Check for consecutive matches
        if len(history) >= self._confirmation_count:
            recent = history[-self._confirmation_count:]
            # All recent matches must be the same non-None person_id
            if all(pid == recent[0] and pid is not None for pid in recent):
                logger.debug(f"Confirmed identity: {temp_id} -> {recent[0]}")
                return recent[0]

        return None

    def _is_likely_new_person(self, temp_id: str) -> bool:
        """
        Determine if a temp_id likely represents a new person.

        A person is considered "new" if they've been tracked for a while
        but never matched to a known person.
        """
        history = self._match_history.get(temp_id, [])
        # Consider new if tracked for at least confirmation_count frames
        # and never matched
        return (
            len(history) >= self._confirmation_count
            and all(pid is None for pid in history)
        )

    def should_save_embedding(
        self,
        result: RecognitionResult,
    ) -> bool:
        """
        Determine if this embedding should be saved to long-term memory.

        Criteria:
        - Quality above threshold
        - Person is identified (not unknown)
        - Or: is a confirmed new person who should be added

        Returns:
            True if embedding should be saved
        """
        # Must meet quality threshold
        if result.encoding.quality_score < self._quality_threshold:
            return False

        # Save if person is identified
        if result.is_identified:
            return True

        # Save if this is a likely new person (we want to remember them)
        if result.is_new_person:
            return True

        return False

    async def save_embedding(
        self,
        result: RecognitionResult,
        person_id: str | None = None,
    ) -> bool:
        """
        Save a high-quality embedding to long-term memory.

        Args:
            result: Recognition result containing the embedding
            person_id: Optional override for person_id (e.g., for new persons)

        Returns:
            True if saved successfully
        """
        if self._memory is None or self._memory.long_term is None:
            return False

        target_id = person_id or result.person_id
        if target_id is None:
            logger.warning("Cannot save embedding: no person_id")
            return False

        return await self._memory.long_term.save_face_embedding(
            target_id,
            result.encoding.embedding,
            result.encoding.quality_score,
        )

    def cleanup_stale_tracks(self, max_age_frames: int = 30) -> int:
        """
        Remove stale tracking entries that haven't been updated.

        Args:
            max_age_frames: Maximum frames without update before removal

        Returns:
            Number of entries removed
        """
        # For now, simple cleanup by limiting total tracked IDs
        max_tracked = 50
        if len(self._temp_id_tracking) > max_tracked:
            # Remove oldest entries (those with shortest match history)
            sorted_ids = sorted(
                self._temp_id_tracking.keys(),
                key=lambda x: len(self._match_history.get(x, [])),
            )
            to_remove = sorted_ids[: len(sorted_ids) - max_tracked]
            for temp_id in to_remove:
                del self._temp_id_tracking[temp_id]
                self._match_history.pop(temp_id, None)
            return len(to_remove)
        return 0

    def get_stats(self) -> dict:
        """Get recognizer statistics."""
        return {
            "tracked_faces": len(self._temp_id_tracking),
            "match_threshold": self._match_threshold,
            "quality_threshold": self._quality_threshold,
            "confirmation_count": self._confirmation_count,
        }
