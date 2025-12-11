"""Tests for the VisionProcessor class."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from server.video.vision_processor import VisionProcessor, VisionResult


class MockTrackedPerson:
    """Mock TrackedPerson for testing."""

    def __init__(
        self,
        best_id: str = "person_1",
        last_bbox: tuple | None = (100, 100, 200, 200),
        is_confirmed: bool = False,
        confirmed_person_id: str | None = None,
    ):
        self.best_id = best_id
        self.last_bbox = last_bbox
        self.is_confirmed = is_confirmed
        self.confirmed_person_id = confirmed_person_id


class MockRecognitionResult:
    """Mock RecognitionResult for testing."""

    def __init__(
        self,
        tracking_id: str = "person_1",
        confidence: float = 0.8,
    ):
        self.tracking_id = tracking_id
        self.confidence = confidence


class TestVisionResult:
    """Tests for VisionResult dataclass."""

    def test_empty_result(self) -> None:
        """Test empty vision result."""
        result = VisionResult()
        assert result.person_count == 0
        assert result.has_familiar_person is False
        assert result.get_primary_person() is None

    def test_with_tracked_persons(self) -> None:
        """Test result with tracked persons."""
        person1 = MockTrackedPerson(last_bbox=(100, 100, 200, 200))  # 100x100
        person2 = MockTrackedPerson(last_bbox=(50, 50, 200, 200))  # 150x150 (larger)

        result = VisionResult(tracked=[person1, person2])

        assert result.person_count == 2
        # Primary should be the one with larger face
        primary = result.get_primary_person()
        assert primary == person2

    def test_has_familiar_person(self) -> None:
        """Test familiar person detection."""
        person1 = MockTrackedPerson(is_confirmed=False)
        person2 = MockTrackedPerson(is_confirmed=True)

        result1 = VisionResult(tracked=[person1])
        assert result1.has_familiar_person is False

        result2 = VisionResult(tracked=[person1, person2])
        assert result2.has_familiar_person is True

    def test_age_ms(self) -> None:
        """Test result age tracking."""
        result = VisionResult(timestamp=time.time() - 0.5)
        assert 400 < result.age_ms < 600

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        person = MockTrackedPerson(best_id="test_person")
        result = VisionResult(tracked=[person], processing_time_ms=50.0)

        d = result.to_dict()
        assert d["person_count"] == 1
        assert d["primary_id"] == "test_person"
        assert d["processing_time_ms"] == 50.0


class TestVisionProcessor:
    """Tests for VisionProcessor."""

    def test_init(self) -> None:
        """Test processor initialization."""
        processor = VisionProcessor()
        assert processor.is_processing is False
        assert processor.get_cached_result() is None

    @pytest.mark.asyncio
    async def test_process_none_frame(self) -> None:
        """Test processing with None frame."""
        processor = VisionProcessor()
        result = await processor.process_if_available(None)
        assert result is None

    @pytest.mark.asyncio
    async def test_skip_if_busy(self) -> None:
        """Test skip-if-busy semantics."""
        processor = VisionProcessor()

        # Simulate busy state
        processor._processing = True

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = await processor.process_if_available(frame)

        # Should return cached result (None since no previous processing)
        assert result is None

        stats = processor.get_stats()
        assert stats["frames_skipped"] == 1

    @pytest.mark.asyncio
    async def test_result_caching(self) -> None:
        """Test that results are cached with TTL."""
        processor = VisionProcessor(result_ttl_ms=1000)

        # Set up cached result
        cached = VisionResult(timestamp=time.time())
        processor._last_result = cached
        processor._last_result_time = time.time()

        # Should return cached result
        result = processor.get_cached_result()
        assert result is cached

    @pytest.mark.asyncio
    async def test_result_expiry(self) -> None:
        """Test that expired results are not returned."""
        processor = VisionProcessor(result_ttl_ms=100)

        # Set up old cached result
        cached = VisionResult(timestamp=time.time() - 1.0)
        processor._last_result = cached
        processor._last_result_time = time.time() - 1.0

        # Should return None (expired)
        result = processor.get_cached_result()
        assert result is None

    def test_estimate_distance(self) -> None:
        """Test distance estimation from face bbox."""
        processor = VisionProcessor()

        # At calibration distance (160px face = 50cm)
        dist = processor._estimate_distance((0, 0, 160, 160))
        assert dist is not None
        assert 45 < dist < 55  # Around 50cm

        # Larger face = closer
        dist = processor._estimate_distance((0, 0, 320, 320))
        assert dist is not None
        assert dist < 30  # Closer than 30cm

        # Smaller face = farther
        dist = processor._estimate_distance((0, 0, 80, 80))
        assert dist is not None
        assert dist > 80  # Farther than 80cm

        # No bbox
        dist = processor._estimate_distance(None)
        assert dist is None

    def test_clear_vision_context(self) -> None:
        """Test clearing vision fields in WorldContext."""
        from server.cognition.behavior.context import WorldContext

        processor = VisionProcessor()
        context = WorldContext()

        # Set some values
        context.person_detected = True
        context.person_is_familiar = True
        context.person_familiarity_score = 80.0
        context.person_distance = 50.0
        context.remembered_person_name = "John"

        # Clear with None result
        processor.update_world_context(context, None)

        assert context.person_detected is False
        assert context.person_is_familiar is False
        assert context.person_familiarity_score == 0.0
        assert context.person_distance is None
        assert context.remembered_person_name is None

    def test_update_world_context(self) -> None:
        """Test updating WorldContext from vision result."""
        from server.cognition.behavior.context import WorldContext

        processor = VisionProcessor()
        context = WorldContext()

        # Create result with confirmed person
        person = MockTrackedPerson(
            best_id="john",
            last_bbox=(100, 100, 260, 260),  # 160px face
            is_confirmed=True,
            confirmed_person_id="john",
        )
        recognition = MockRecognitionResult(
            tracking_id="john",
            confidence=0.85,
        )
        result = VisionResult(tracked=[person], results=[recognition])

        processor.update_world_context(context, result)

        assert context.person_detected is True
        assert context.person_is_familiar is True
        assert context.person_familiarity_score == 85.0
        assert context.person_distance is not None
        assert context.remembered_person_name == "john"

    def test_stats(self) -> None:
        """Test statistics tracking."""
        processor = VisionProcessor()

        stats = processor.get_stats()
        assert stats["initialized"] is False
        assert stats["processing"] is False
        assert stats["frames_processed"] == 0
        assert stats["frames_skipped"] == 0

    def test_repr(self) -> None:
        """Test string representation."""
        processor = VisionProcessor()
        repr_str = repr(processor)
        assert "VisionProcessor" in repr_str

    def test_set_memory_system_before_init(self) -> None:
        """Test setting memory system before lazy initialization."""
        processor = VisionProcessor()
        mock_memory = MagicMock()

        # Set memory before _ensure_initialized has run
        processor.set_memory_system(mock_memory)

        assert processor._memory is mock_memory
        # Recognizer not yet created
        assert processor._recognizer is None

    def test_set_memory_system_after_init(self) -> None:
        """Test setting memory system after recognizer is initialized."""
        processor = VisionProcessor()

        # Force initialization with no memory
        processor._ensure_initialized()
        old_recognizer = processor._recognizer

        # Now set memory - should recreate recognizer
        mock_memory = MagicMock()
        with patch(
            "server.perception.vision.face_recognizer.FaceRecognizer"
        ) as mock_recognizer_class:
            mock_recognizer_class.return_value = MagicMock()
            processor.set_memory_system(mock_memory)

            # Recognizer should be recreated with memory
            mock_recognizer_class.assert_called_once_with(memory_system=mock_memory)
            assert processor._recognizer is not old_recognizer
