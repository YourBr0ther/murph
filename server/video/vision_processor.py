"""
Murph - Vision Processor
Orchestrates vision pipeline with async processing and caching.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from shared.constants import (
    VISION_FACE_DISTANCE_CALIBRATION_PX,
    VISION_RESULT_TTL_MS,
)

if TYPE_CHECKING:
    from server.cognition.behavior.context import WorldContext
    from server.cognition.memory.memory_system import MemorySystem
    from server.perception.vision.face_recognizer import FaceRecognizer
    from server.perception.vision.person_tracker import PersonTracker
    from server.perception.vision.types import RecognitionResult, TrackedPerson

logger = logging.getLogger(__name__)


@dataclass
class VisionResult:
    """Result of vision processing for a single frame."""

    results: list[RecognitionResult] = field(default_factory=list)
    tracked: list[TrackedPerson] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    processing_time_ms: float = 0.0

    def get_primary_person(self) -> TrackedPerson | None:
        """
        Get the most prominent tracked person.

        Uses largest face area as the primary selection criterion.

        Returns:
            The primary TrackedPerson, or None if no people tracked
        """
        if not self.tracked:
            return None

        def face_area(t: TrackedPerson) -> float:
            if not t.last_bbox:
                return 0.0
            x1, y1, x2, y2 = t.last_bbox
            return (x2 - x1) * (y2 - y1)

        return max(self.tracked, key=face_area, default=None)

    @property
    def person_count(self) -> int:
        """Number of people currently tracked."""
        return len(self.tracked)

    @property
    def has_familiar_person(self) -> bool:
        """Check if any tracked person is confirmed/familiar."""
        return any(t.is_confirmed for t in self.tracked)

    @property
    def age_ms(self) -> float:
        """Age of this result in milliseconds."""
        return (time.time() - self.timestamp) * 1000

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for debugging."""
        primary = self.get_primary_person()
        return {
            "person_count": self.person_count,
            "has_familiar": self.has_familiar_person,
            "primary_id": primary.best_id if primary else None,
            "age_ms": self.age_ms,
            "processing_time_ms": self.processing_time_ms,
        }


class VisionProcessor:
    """
    Orchestrates vision pipeline with async processing and caching.

    Handles:
    - FaceRecognizer and PersonTracker coordination
    - Skip-if-busy semantics (avoid blocking on slow frames)
    - Result caching with TTL for perception loop
    - WorldContext updates from vision results

    Usage:
        processor = VisionProcessor(recognizer, tracker)

        # In perception loop
        result = await processor.process_if_available(frame)
        if result:
            processor.update_world_context(world_context, result)
    """

    def __init__(
        self,
        recognizer: FaceRecognizer | None = None,
        tracker: PersonTracker | None = None,
        memory_system: MemorySystem | None = None,
        result_ttl_ms: float = VISION_RESULT_TTL_MS,
    ) -> None:
        """
        Initialize vision processor.

        Args:
            recognizer: FaceRecognizer instance (created if None)
            tracker: PersonTracker instance (created if None)
            memory_system: MemorySystem for face lookups
            result_ttl_ms: How long cached results are valid
        """
        self._memory = memory_system
        self._result_ttl = result_ttl_ms / 1000.0

        # Lazy-load vision components to avoid import issues
        self._recognizer = recognizer
        self._tracker = tracker
        self._initialized = False

        # Processing state
        self._processing = False
        self._last_result: VisionResult | None = None
        self._last_result_time: float = 0.0

        # Stats
        self._frames_processed = 0
        self._frames_skipped = 0
        self._total_processing_time_ms = 0.0

        logger.info(f"VisionProcessor created (ttl={result_ttl_ms}ms)")

    def _ensure_initialized(self) -> bool:
        """Lazy-initialize vision components."""
        if self._initialized:
            return True

        try:
            if self._recognizer is None:
                from server.perception.vision.face_recognizer import FaceRecognizer

                self._recognizer = FaceRecognizer(memory_system=self._memory)

            if self._tracker is None:
                from server.perception.vision.person_tracker import PersonTracker

                self._tracker = PersonTracker()

            self._initialized = True
            logger.info("VisionProcessor initialized with vision components")
            return True

        except ImportError as e:
            logger.error(f"Failed to import vision components: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize vision components: {e}")
            return False

    async def process_if_available(
        self,
        frame: np.ndarray | None,
    ) -> VisionResult | None:
        """
        Process frame if not already processing.

        If already busy processing a previous frame, returns cached result.
        This implements skip-if-busy semantics to avoid backlog.

        Args:
            frame: RGB numpy array (H, W, 3) or None

        Returns:
            VisionResult if processing completed, cached result if busy, None if unavailable
        """
        # Return cached result if still processing or no frame
        if self._processing:
            self._frames_skipped += 1
            return self._get_cached_result()

        if frame is None:
            return self._get_cached_result()

        if not self._ensure_initialized():
            return None

        # Start processing
        self._processing = True
        start_time = time.time()

        try:
            # Run face recognition
            results = await self._recognizer.process_frame(frame)

            # Update tracker
            tracked = self._tracker.update(results)

            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000

            # Create result
            result = VisionResult(
                results=results,
                tracked=tracked,
                timestamp=time.time(),
                processing_time_ms=processing_time_ms,
            )

            # Update cache
            self._last_result = result
            self._last_result_time = time.time()

            # Update stats
            self._frames_processed += 1
            self._total_processing_time_ms += processing_time_ms

            if self._frames_processed % 100 == 0:
                avg_time = self._total_processing_time_ms / self._frames_processed
                logger.info(
                    f"Vision stats: processed={self._frames_processed}, "
                    f"skipped={self._frames_skipped}, avg_time={avg_time:.1f}ms"
                )

            return result

        except Exception as e:
            logger.error(f"Vision processing error: {e}", exc_info=True)
            return self._get_cached_result()

        finally:
            self._processing = False

    def _get_cached_result(self) -> VisionResult | None:
        """Get cached result if still valid."""
        if self._last_result is None:
            return None

        age = time.time() - self._last_result_time
        if age > self._result_ttl:
            return None

        return self._last_result

    def get_cached_result(self) -> VisionResult | None:
        """Public access to cached result."""
        return self._get_cached_result()

    def update_world_context(
        self,
        context: WorldContext,
        result: VisionResult | None = None,
    ) -> None:
        """
        Update WorldContext from vision results.

        Args:
            context: WorldContext to update
            result: VisionResult to use (uses cached if None)
        """
        if result is None:
            result = self._get_cached_result()

        if result is None:
            # Clear vision state
            self._clear_vision_context(context)
            return

        primary = result.get_primary_person()

        if primary:
            context.person_detected = True
            context.person_is_familiar = primary.is_confirmed

            # Get familiarity score from recognition confidence
            if primary.is_confirmed and result.results:
                # Find the matching result for this person
                for r in result.results:
                    if r.tracking_id == primary.best_id:
                        context.person_familiarity_score = r.confidence * 100
                        break
            else:
                context.person_familiarity_score = 0.0

            # Estimate distance from face size
            context.person_distance = self._estimate_distance(primary.last_bbox)

            # Set person name if confirmed
            if primary.confirmed_person_id:
                context.remembered_person_name = primary.confirmed_person_id
            else:
                context.remembered_person_name = None

        else:
            self._clear_vision_context(context)

    def _clear_vision_context(self, context: WorldContext) -> None:
        """Clear vision-related fields in WorldContext."""
        context.person_detected = False
        context.person_is_familiar = False
        context.person_familiarity_score = 0.0
        context.person_distance = None
        context.remembered_person_name = None

    def _estimate_distance(
        self,
        bbox: tuple[float, float, float, float] | None,
    ) -> float | None:
        """
        Estimate distance to person based on face bounding box size.

        Uses a simple inverse relationship: larger face = closer distance.
        Calibrated assuming VISION_FACE_DISTANCE_CALIBRATION_PX face height = 50cm distance.

        Args:
            bbox: Face bounding box (x1, y1, x2, y2) or None

        Returns:
            Estimated distance in centimeters, or None if no bbox
        """
        if bbox is None:
            return None

        x1, y1, x2, y2 = bbox
        face_height = y2 - y1

        if face_height <= 0:
            return None

        # Calibration: 160px face height ~ 50cm distance
        # Distance is inversely proportional to face size
        calibration_distance_cm = 50.0
        estimated_distance = (
            VISION_FACE_DISTANCE_CALIBRATION_PX / face_height
        ) * calibration_distance_cm

        return estimated_distance

    @property
    def is_processing(self) -> bool:
        """Check if currently processing a frame."""
        return self._processing

    def get_stats(self) -> dict[str, Any]:
        """Get processor statistics."""
        avg_time = (
            self._total_processing_time_ms / self._frames_processed
            if self._frames_processed > 0
            else 0.0
        )
        return {
            "initialized": self._initialized,
            "processing": self._processing,
            "frames_processed": self._frames_processed,
            "frames_skipped": self._frames_skipped,
            "avg_processing_time_ms": avg_time,
            "has_cached_result": self._last_result is not None,
            "cached_result_age_ms": (
                (time.time() - self._last_result_time) * 1000
                if self._last_result_time > 0
                else -1
            ),
        }

    def __repr__(self) -> str:
        return (
            f"VisionProcessor(initialized={self._initialized}, "
            f"processing={self._processing}, "
            f"processed={self._frames_processed})"
        )
