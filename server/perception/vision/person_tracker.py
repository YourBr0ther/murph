"""
Murph - Person Tracker
Tracks persons across video frames using position and appearance.
"""

import logging
import time
import uuid
from typing import Literal

import numpy as np

from .types import FaceEncoding, RecognitionResult, TrackedPerson

logger = logging.getLogger(__name__)


class PersonTracker:
    """
    Track persons across video frames.

    Handles:
    - Associating detections across frames via position/embedding similarity
    - Maintaining identity during brief detection failures
    - Generating stable temporary IDs for unknown persons
    - Confirming identity after multiple consistent matches

    Usage:
        tracker = PersonTracker()
        tracked = tracker.update(recognition_results)
        primary = tracker.get_primary_person()
    """

    def __init__(
        self,
        max_frames_missing: int = 10,
        iou_threshold: float = 0.3,
        embedding_threshold: float = 0.5,
        confirmation_frames: int = 3,
    ) -> None:
        """
        Initialize person tracker.

        Args:
            max_frames_missing: Frames before dropping a track (default 10)
            iou_threshold: IoU threshold for bbox matching (default 0.3)
            embedding_threshold: Embedding similarity threshold (default 0.5)
            confirmation_frames: Frames needed to confirm identity (default 3)
        """
        self._max_frames_missing = max_frames_missing
        self._iou_threshold = iou_threshold
        self._embedding_threshold = embedding_threshold
        self._confirmation_frames = confirmation_frames

        # Active tracks: temp_id -> TrackedPerson
        self._tracked: dict[str, TrackedPerson] = {}
        self._frame_count = 0

    def update(self, results: list[RecognitionResult]) -> list[TrackedPerson]:
        """
        Update tracker with new recognition results.

        Associates new detections with existing tracks using:
        1. Bounding box IoU (position continuity)
        2. Embedding similarity (appearance matching)

        Args:
            results: Recognition results from current frame

        Returns:
            Updated list of tracked persons
        """
        self._frame_count += 1

        # Match results to existing tracks
        matched_tracks: set[str] = set()
        unmatched_results: list[RecognitionResult] = []

        for result in results:
            best_track_id = self._find_best_match(result)

            if best_track_id:
                self._update_track(best_track_id, result)
                matched_tracks.add(best_track_id)
            else:
                unmatched_results.append(result)

        # Create new tracks for unmatched results
        for result in unmatched_results:
            self._create_track(result)

        # Update unmatched tracks (increase missing frame count)
        for track_id in list(self._tracked.keys()):
            if track_id not in matched_tracks:
                track = self._tracked[track_id]
                frames_since_seen = self._frame_count - track.last_seen_frame
                if frames_since_seen > self._max_frames_missing:
                    logger.debug(f"Dropping stale track: {track_id}")
                    del self._tracked[track_id]

        return list(self._tracked.values())

    def _find_best_match(self, result: RecognitionResult) -> str | None:
        """
        Find the best matching existing track for a result.

        Uses weighted combination of IoU and embedding similarity.
        """
        best_score = 0.0
        best_id = None

        bbox = result.encoding.face.bbox

        for track_id, track in self._tracked.items():
            score = 0.0

            # IoU score (position continuity)
            if track.last_bbox:
                iou = self._compute_iou(bbox, track.last_bbox)
                if iou >= self._iou_threshold:
                    score += iou * 0.5  # 50% weight for position

            # Embedding similarity (appearance)
            if track.last_encoding:
                similarity = result.encoding.cosine_similarity(track.last_encoding)
                if similarity >= self._embedding_threshold:
                    score += similarity * 0.5  # 50% weight for appearance

            if score > best_score:
                best_score = score
                best_id = track_id

        # Require minimum combined score
        if best_score < 0.3:
            return None

        return best_id

    def _update_track(self, track_id: str, result: RecognitionResult) -> None:
        """Update an existing track with new detection."""
        track = self._tracked[track_id]

        # Update position and appearance
        track.last_bbox = result.encoding.face.bbox
        track.last_encoding = result.encoding
        track.last_seen_frame = self._frame_count
        track.frames_tracked += 1

        # Update match history for identity confirmation
        track.match_history.append(result.person_id)
        if len(track.match_history) > self._confirmation_frames * 2:
            track.match_history.pop(0)

        # Check for identity confirmation
        if not track.is_confirmed:
            confirmed = self._check_confirmation(track)
            if confirmed:
                track.confirmed_person_id = confirmed
                logger.info(f"Confirmed identity: {track_id} -> {confirmed}")

    def _create_track(self, result: RecognitionResult) -> TrackedPerson:
        """Create a new track for an unmatched detection."""
        temp_id = result.temporary_id or f"track_{uuid.uuid4().hex[:8]}"

        track = TrackedPerson(
            temporary_id=temp_id,
            confirmed_person_id=result.person_id if result.confidence > 0.8 else None,
            last_bbox=result.encoding.face.bbox,
            last_encoding=result.encoding,
            match_history=[result.person_id],
            frames_tracked=1,
            first_seen_time=time.time(),
            last_seen_frame=self._frame_count,
        )

        self._tracked[temp_id] = track
        logger.debug(f"Created new track: {temp_id}")
        return track

    def _check_confirmation(self, track: TrackedPerson) -> str | None:
        """
        Check if track has enough consistent matches to confirm identity.

        Returns confirmed person_id or None.
        """
        if len(track.match_history) < self._confirmation_frames:
            return None

        recent = track.match_history[-self._confirmation_frames:]

        # All recent matches must be the same non-None person
        if all(pid == recent[0] and pid is not None for pid in recent):
            return recent[0]

        return None

    def _compute_iou(
        self,
        bbox1: tuple[float, float, float, float],
        bbox2: tuple[float, float, float, float],
    ) -> float:
        """Compute Intersection over Union of two bounding boxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        if union == 0:
            return 0.0

        return intersection / union

    def get_active_tracks(self) -> list[TrackedPerson]:
        """Get all currently active tracked persons."""
        return list(self._tracked.values())

    def get_primary_person(
        self,
        preference: Literal["largest", "central", "longest_tracked"] = "largest",
        frame_shape: tuple[int, int] | None = None,
    ) -> TrackedPerson | None:
        """
        Get the primary (most relevant) tracked person.

        Args:
            preference: Selection strategy:
                - "largest": Largest face bounding box
                - "central": Closest to frame center
                - "longest_tracked": Most frames tracked
            frame_shape: (height, width) required for "central" preference

        Returns:
            The primary tracked person, or None if no tracks
        """
        if not self._tracked:
            return None

        tracks = list(self._tracked.values())

        if preference == "largest":
            return max(
                tracks,
                key=lambda t: (
                    (t.last_bbox[2] - t.last_bbox[0]) * (t.last_bbox[3] - t.last_bbox[1])
                    if t.last_bbox
                    else 0
                ),
            )

        elif preference == "longest_tracked":
            return max(tracks, key=lambda t: t.frames_tracked)

        elif preference == "central":
            if frame_shape is None:
                # Fall back to largest if no frame shape
                return self.get_primary_person("largest")

            center_x = frame_shape[1] / 2
            center_y = frame_shape[0] / 2

            def distance_to_center(track: TrackedPerson) -> float:
                if track.last_bbox is None:
                    return float("inf")
                bbox_center_x = (track.last_bbox[0] + track.last_bbox[2]) / 2
                bbox_center_y = (track.last_bbox[1] + track.last_bbox[3]) / 2
                return (
                    (bbox_center_x - center_x) ** 2 + (bbox_center_y - center_y) ** 2
                ) ** 0.5

            return min(tracks, key=distance_to_center)

        else:
            raise ValueError(f"Unknown preference: {preference}")

    def get_confirmed_persons(self) -> list[TrackedPerson]:
        """Get all tracks with confirmed identity."""
        return [t for t in self._tracked.values() if t.is_confirmed]

    def get_track(self, track_id: str) -> TrackedPerson | None:
        """Get a specific track by ID."""
        return self._tracked.get(track_id)

    def clear(self) -> None:
        """Clear all tracks."""
        self._tracked.clear()
        self._frame_count = 0

    def get_stats(self) -> dict:
        """Get tracker statistics."""
        confirmed = sum(1 for t in self._tracked.values() if t.is_confirmed)
        return {
            "total_tracks": len(self._tracked),
            "confirmed_tracks": confirmed,
            "unconfirmed_tracks": len(self._tracked) - confirmed,
            "frame_count": self._frame_count,
        }
