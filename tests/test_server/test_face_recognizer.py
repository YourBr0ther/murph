"""
Unit tests for Murph's Face Recognizer and Person Tracker.
"""

import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from server.perception.vision import (
    DetectedFace,
    FaceEncoding,
    FaceRecognizer,
    PersonTracker,
    RecognitionResult,
    TrackedPerson,
)


def make_face(
    bbox: tuple[float, float, float, float] = (0, 0, 100, 100),
    confidence: float = 0.95,
) -> DetectedFace:
    """Create a DetectedFace with default values."""
    return DetectedFace(
        bbox=bbox,
        confidence=confidence,
        face_image=np.random.randint(0, 256, (160, 160, 3), dtype=np.uint8),
    )


def make_encoding(
    face: DetectedFace | None = None,
    quality: float = 0.8,
) -> FaceEncoding:
    """Create a FaceEncoding with random embedding."""
    if face is None:
        face = make_face()
    embedding = np.random.randn(128).astype(np.float32)
    embedding = embedding / np.linalg.norm(embedding)
    return FaceEncoding(embedding=embedding, quality_score=quality, face=face)


class TestRecognitionResult:
    """Tests for RecognitionResult dataclass."""

    def test_is_identified_true(self):
        """Test is_identified when person_id is set."""
        encoding = make_encoding()
        result = RecognitionResult(
            encoding=encoding,
            person_id="person_123",
            confidence=0.85,
        )
        assert result.is_identified is True

    def test_is_identified_false(self):
        """Test is_identified when person_id is None."""
        encoding = make_encoding()
        result = RecognitionResult(
            encoding=encoding,
            person_id=None,
            confidence=0.0,
        )
        assert result.is_identified is False

    def test_tracking_id_prefers_person_id(self):
        """Test tracking_id returns person_id when available."""
        encoding = make_encoding()
        result = RecognitionResult(
            encoding=encoding,
            person_id="person_123",
            confidence=0.85,
            temporary_id="unknown_abc",
        )
        assert result.tracking_id == "person_123"

    def test_tracking_id_falls_back_to_temp(self):
        """Test tracking_id returns temporary_id when no person_id."""
        encoding = make_encoding()
        result = RecognitionResult(
            encoding=encoding,
            person_id=None,
            confidence=0.0,
            temporary_id="unknown_abc",
        )
        assert result.tracking_id == "unknown_abc"


class TestTrackedPerson:
    """Tests for TrackedPerson dataclass."""

    def test_is_confirmed_true(self):
        """Test is_confirmed when confirmed_person_id is set."""
        tracked = TrackedPerson(
            temporary_id="unknown_123",
            confirmed_person_id="person_456",
        )
        assert tracked.is_confirmed is True

    def test_is_confirmed_false(self):
        """Test is_confirmed when not confirmed."""
        tracked = TrackedPerson(temporary_id="unknown_123")
        assert tracked.is_confirmed is False

    def test_best_id_confirmed(self):
        """Test best_id returns confirmed ID when available."""
        tracked = TrackedPerson(
            temporary_id="unknown_123",
            confirmed_person_id="person_456",
        )
        assert tracked.best_id == "person_456"

    def test_best_id_temporary(self):
        """Test best_id returns temporary ID when not confirmed."""
        tracked = TrackedPerson(temporary_id="unknown_123")
        assert tracked.best_id == "unknown_123"

    def test_to_dict(self):
        """Test serialization to dictionary."""
        tracked = TrackedPerson(
            temporary_id="unknown_123",
            confirmed_person_id="person_456",
            last_bbox=(10, 20, 110, 120),
            frames_tracked=15,
            first_seen_time=1000.0,
        )
        d = tracked.to_dict()

        assert d["temporary_id"] == "unknown_123"
        assert d["confirmed_person_id"] == "person_456"
        assert d["last_bbox"] == (10, 20, 110, 120)
        assert d["frames_tracked"] == 15
        assert d["is_confirmed"] is True


class TestFaceRecognizerInit:
    """Tests for FaceRecognizer initialization."""

    @patch("server.perception.vision.face_recognizer.FaceDetector")
    @patch("server.perception.vision.face_recognizer.FaceEncoder")
    def test_init_without_memory(self, mock_encoder, mock_detector):
        """Test initialization without memory system."""
        recognizer = FaceRecognizer(memory_system=None, device="cpu")

        assert recognizer._memory is None
        mock_detector.assert_called_once()
        mock_encoder.assert_called_once()

    @patch("server.perception.vision.face_recognizer.FaceDetector")
    @patch("server.perception.vision.face_recognizer.FaceEncoder")
    def test_init_with_custom_thresholds(self, mock_encoder, mock_detector):
        """Test initialization with custom thresholds."""
        recognizer = FaceRecognizer(
            memory_system=None,
            device="cpu",
            match_threshold=0.7,
            quality_threshold=0.6,
            min_matches_for_confirmation=5,
        )

        assert recognizer._match_threshold == 0.7
        assert recognizer._quality_threshold == 0.6
        assert recognizer._confirmation_count == 5


class TestFaceRecognizerIdentify:
    """Tests for face identification."""

    @patch("server.perception.vision.face_recognizer.FaceDetector")
    @patch("server.perception.vision.face_recognizer.FaceEncoder")
    async def test_identify_without_memory(self, mock_encoder, mock_detector):
        """Test identification returns None without memory."""
        recognizer = FaceRecognizer(memory_system=None, device="cpu")
        encoding = make_encoding()

        person_id, confidence = await recognizer.identify_face(encoding)

        assert person_id is None
        assert confidence == 0.0

    @patch("server.perception.vision.face_recognizer.FaceDetector")
    @patch("server.perception.vision.face_recognizer.FaceEncoder")
    async def test_identify_with_memory_match(self, mock_encoder, mock_detector):
        """Test identification with matching person in memory."""
        mock_memory = MagicMock()
        mock_memory.lookup_person_by_face = AsyncMock(
            return_value=("person_123", 0.85)
        )

        recognizer = FaceRecognizer(memory_system=mock_memory, device="cpu")
        encoding = make_encoding()

        person_id, confidence = await recognizer.identify_face(encoding)

        assert person_id == "person_123"
        assert confidence == 0.85

    @patch("server.perception.vision.face_recognizer.FaceDetector")
    @patch("server.perception.vision.face_recognizer.FaceEncoder")
    async def test_identify_with_memory_no_match(self, mock_encoder, mock_detector):
        """Test identification with no matching person."""
        mock_memory = MagicMock()
        mock_memory.lookup_person_by_face = AsyncMock(return_value=(None, 0.0))

        recognizer = FaceRecognizer(memory_system=mock_memory, device="cpu")
        encoding = make_encoding()

        person_id, confidence = await recognizer.identify_face(encoding)

        assert person_id is None
        assert confidence == 0.0


class TestMatchConfirmation:
    """Tests for match confirmation logic."""

    @patch("server.perception.vision.face_recognizer.FaceDetector")
    @patch("server.perception.vision.face_recognizer.FaceEncoder")
    def test_single_match_not_confirmed(self, mock_encoder, mock_detector):
        """Test that single match doesn't confirm identity."""
        recognizer = FaceRecognizer(
            memory_system=None, device="cpu", min_matches_for_confirmation=3
        )
        temp_id = "unknown_123"
        recognizer._match_history[temp_id] = []

        confirmed = recognizer._update_match_history(temp_id, "person_456")

        assert confirmed is None

    @patch("server.perception.vision.face_recognizer.FaceDetector")
    @patch("server.perception.vision.face_recognizer.FaceEncoder")
    def test_consecutive_matches_confirms(self, mock_encoder, mock_detector):
        """Test that consecutive matches confirm identity."""
        recognizer = FaceRecognizer(
            memory_system=None, device="cpu", min_matches_for_confirmation=3
        )
        temp_id = "unknown_123"
        recognizer._match_history[temp_id] = []

        # First two matches
        recognizer._update_match_history(temp_id, "person_456")
        recognizer._update_match_history(temp_id, "person_456")
        # Third match should confirm
        confirmed = recognizer._update_match_history(temp_id, "person_456")

        assert confirmed == "person_456"

    @patch("server.perception.vision.face_recognizer.FaceDetector")
    @patch("server.perception.vision.face_recognizer.FaceEncoder")
    def test_mixed_matches_not_confirmed(self, mock_encoder, mock_detector):
        """Test that mixed matches don't confirm."""
        recognizer = FaceRecognizer(
            memory_system=None, device="cpu", min_matches_for_confirmation=3
        )
        temp_id = "unknown_123"
        recognizer._match_history[temp_id] = []

        recognizer._update_match_history(temp_id, "person_456")
        recognizer._update_match_history(temp_id, "person_789")  # Different!
        confirmed = recognizer._update_match_history(temp_id, "person_456")

        assert confirmed is None

    @patch("server.perception.vision.face_recognizer.FaceDetector")
    @patch("server.perception.vision.face_recognizer.FaceEncoder")
    def test_none_matches_not_confirmed(self, mock_encoder, mock_detector):
        """Test that None matches don't confirm."""
        recognizer = FaceRecognizer(
            memory_system=None, device="cpu", min_matches_for_confirmation=3
        )
        temp_id = "unknown_123"
        recognizer._match_history[temp_id] = []

        recognizer._update_match_history(temp_id, None)
        recognizer._update_match_history(temp_id, None)
        confirmed = recognizer._update_match_history(temp_id, None)

        assert confirmed is None


class TestShouldSaveEmbedding:
    """Tests for embedding save decisions."""

    @patch("server.perception.vision.face_recognizer.FaceDetector")
    @patch("server.perception.vision.face_recognizer.FaceEncoder")
    def test_save_identified_high_quality(self, mock_encoder, mock_detector):
        """Test saving embedding for identified person with high quality."""
        recognizer = FaceRecognizer(
            memory_system=None, device="cpu", quality_threshold=0.5
        )
        encoding = make_encoding(quality=0.8)
        result = RecognitionResult(
            encoding=encoding,
            person_id="person_123",
            confidence=0.85,
        )

        assert recognizer.should_save_embedding(result) is True

    @patch("server.perception.vision.face_recognizer.FaceDetector")
    @patch("server.perception.vision.face_recognizer.FaceEncoder")
    def test_dont_save_low_quality(self, mock_encoder, mock_detector):
        """Test not saving low quality embedding."""
        recognizer = FaceRecognizer(
            memory_system=None, device="cpu", quality_threshold=0.5
        )
        encoding = make_encoding(quality=0.3)  # Below threshold
        result = RecognitionResult(
            encoding=encoding,
            person_id="person_123",
            confidence=0.85,
        )

        assert recognizer.should_save_embedding(result) is False

    @patch("server.perception.vision.face_recognizer.FaceDetector")
    @patch("server.perception.vision.face_recognizer.FaceEncoder")
    def test_save_new_person_high_quality(self, mock_encoder, mock_detector):
        """Test saving embedding for new person."""
        recognizer = FaceRecognizer(
            memory_system=None, device="cpu", quality_threshold=0.5
        )
        encoding = make_encoding(quality=0.8)
        result = RecognitionResult(
            encoding=encoding,
            person_id=None,
            confidence=0.0,
            is_new_person=True,
        )

        assert recognizer.should_save_embedding(result) is True

    @patch("server.perception.vision.face_recognizer.FaceDetector")
    @patch("server.perception.vision.face_recognizer.FaceEncoder")
    def test_dont_save_unknown_not_new(self, mock_encoder, mock_detector):
        """Test not saving unknown non-new person."""
        recognizer = FaceRecognizer(
            memory_system=None, device="cpu", quality_threshold=0.5
        )
        encoding = make_encoding(quality=0.8)
        result = RecognitionResult(
            encoding=encoding,
            person_id=None,
            confidence=0.0,
            is_new_person=False,
        )

        assert recognizer.should_save_embedding(result) is False


class TestPersonTracker:
    """Tests for PersonTracker."""

    def test_update_creates_new_track(self):
        """Test that new detection creates a track."""
        tracker = PersonTracker()
        encoding = make_encoding()
        result = RecognitionResult(
            encoding=encoding,
            person_id=None,
            confidence=0.0,
            temporary_id="unknown_123",
        )

        tracks = tracker.update([result])

        assert len(tracks) == 1
        assert tracks[0].temporary_id == "unknown_123"
        assert tracks[0].frames_tracked == 1

    def test_update_matches_existing_track(self):
        """Test that similar detection matches existing track."""
        tracker = PersonTracker()

        # First detection
        encoding1 = make_encoding(face=make_face(bbox=(100, 100, 200, 200)))
        result1 = RecognitionResult(
            encoding=encoding1,
            person_id=None,
            confidence=0.0,
            temporary_id="unknown_123",
        )
        tracker.update([result1])

        # Second detection at similar position
        encoding2 = make_encoding(face=make_face(bbox=(105, 105, 205, 205)))
        result2 = RecognitionResult(
            encoding=encoding2,
            person_id=None,
            confidence=0.0,
            temporary_id="unknown_456",  # Different temp_id
        )
        tracks = tracker.update([result2])

        # Should still be one track with updated frame count
        assert len(tracks) == 1
        assert tracks[0].frames_tracked == 2

    def test_update_drops_stale_tracks(self):
        """Test that old tracks are dropped after max_frames_missing."""
        tracker = PersonTracker(max_frames_missing=2)

        # Create initial track
        encoding = make_encoding()
        result = RecognitionResult(
            encoding=encoding,
            person_id=None,
            confidence=0.0,
            temporary_id="unknown_123",
        )
        tracker.update([result])
        assert len(tracker.get_active_tracks()) == 1

        # Update with no results (person left)
        tracker.update([])
        tracker.update([])
        tracks = tracker.update([])  # Third update should drop

        assert len(tracks) == 0

    def test_get_primary_person_largest(self):
        """Test getting primary person by size."""
        tracker = PersonTracker()

        # Create two tracks with different sizes
        small_face = make_face(bbox=(0, 0, 50, 50))
        large_face = make_face(bbox=(100, 100, 300, 300))

        results = [
            RecognitionResult(
                encoding=make_encoding(face=small_face),
                person_id=None,
                confidence=0.0,
                temporary_id="small",
            ),
            RecognitionResult(
                encoding=make_encoding(face=large_face),
                person_id=None,
                confidence=0.0,
                temporary_id="large",
            ),
        ]
        tracker.update(results)

        primary = tracker.get_primary_person("largest")

        assert primary is not None
        assert primary.temporary_id == "large"

    def test_get_primary_person_longest_tracked(self):
        """Test getting primary person by tracking duration."""
        tracker = PersonTracker()

        # Create first track
        face1 = make_face(bbox=(0, 0, 100, 100))
        result1 = RecognitionResult(
            encoding=make_encoding(face=face1),
            person_id=None,
            confidence=0.0,
            temporary_id="first",
        )
        tracker.update([result1])
        tracker.update([result1])  # Second frame

        # Create second track
        face2 = make_face(bbox=(200, 200, 400, 400))
        result2 = RecognitionResult(
            encoding=make_encoding(face=face2),
            person_id=None,
            confidence=0.0,
            temporary_id="second",
        )
        tracker.update([result1, result2])  # Third frame for first, first for second

        primary = tracker.get_primary_person("longest_tracked")

        assert primary is not None
        assert primary.frames_tracked == 3  # First track has 3 frames

    def test_get_confirmed_persons(self):
        """Test getting only confirmed persons."""
        tracker = PersonTracker(confirmation_frames=2)

        face = make_face()
        # Create track with confirmed identity
        results = [
            RecognitionResult(
                encoding=make_encoding(face=face),
                person_id="person_123",
                confidence=0.9,
                temporary_id="track1",
            )
        ]

        tracker.update(results)
        tracker.update(results)  # Second match confirms

        confirmed = tracker.get_confirmed_persons()

        assert len(confirmed) == 1
        assert confirmed[0].confirmed_person_id == "person_123"

    def test_clear(self):
        """Test clearing all tracks."""
        tracker = PersonTracker()

        result = RecognitionResult(
            encoding=make_encoding(),
            person_id=None,
            confidence=0.0,
            temporary_id="track1",
        )
        tracker.update([result])
        assert len(tracker.get_active_tracks()) == 1

        tracker.clear()

        assert len(tracker.get_active_tracks()) == 0

    def test_get_stats(self):
        """Test getting tracker statistics."""
        tracker = PersonTracker(confirmation_frames=2)

        face = make_face()
        result = RecognitionResult(
            encoding=make_encoding(face=face),
            person_id="person_123",
            confidence=0.9,
            temporary_id="track1",
        )
        tracker.update([result])
        tracker.update([result])  # Confirms

        stats = tracker.get_stats()

        assert stats["total_tracks"] == 1
        assert stats["confirmed_tracks"] == 1
        assert stats["unconfirmed_tracks"] == 0
        assert stats["frame_count"] == 2


class TestIoUComputation:
    """Tests for IoU computation in PersonTracker."""

    def test_iou_no_overlap(self):
        """Test IoU of non-overlapping boxes."""
        tracker = PersonTracker()
        bbox1 = (0, 0, 100, 100)
        bbox2 = (200, 200, 300, 300)
        assert tracker._compute_iou(bbox1, bbox2) == 0.0

    def test_iou_same_box(self):
        """Test IoU of identical boxes."""
        tracker = PersonTracker()
        bbox = (0, 0, 100, 100)
        assert tracker._compute_iou(bbox, bbox) == 1.0

    def test_iou_half_overlap(self):
        """Test IoU of 50% overlapping boxes."""
        tracker = PersonTracker()
        bbox1 = (0, 0, 100, 100)  # 10000 area
        bbox2 = (50, 0, 150, 100)  # 10000 area, 5000 overlap
        # Union = 10000 + 10000 - 5000 = 15000
        # IoU = 5000 / 15000 = 0.333...
        iou = tracker._compute_iou(bbox1, bbox2)
        assert abs(iou - 1 / 3) < 0.01


@pytest.mark.hardware
class TestFaceRecognizerIntegration:
    """Integration tests with real models and database."""

    @pytest.fixture
    async def memory_system(self):
        """Create a memory system with database."""
        from server.cognition.memory import MemorySystem
        from server.storage import Database

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = Database(db_path)
            await db.initialize()

            memory = MemorySystem()
            await memory.initialize_long_term(db)

            yield memory

            await db.close()

    async def test_full_pipeline_no_match(self, memory_system):
        """Test full recognition pipeline with no matching person."""
        recognizer = FaceRecognizer(
            memory_system=memory_system,
            device="cpu",
        )

        # Create a fake frame
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

        # This will likely find no faces in random noise, but shouldn't crash
        results = await recognizer.process_frame(frame)

        assert isinstance(results, list)
