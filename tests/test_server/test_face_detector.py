"""
Unit tests for Murph's Face Detector.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from server.perception.vision import DetectedFace, FaceDetector


class TestDetectedFace:
    """Tests for DetectedFace dataclass."""

    def test_area_calculation(self):
        """Test bounding box area calculation."""
        face = DetectedFace(
            bbox=(0, 0, 100, 100),
            confidence=0.95,
        )
        assert face.area == 10000

    def test_area_non_square(self):
        """Test area calculation for non-square bbox."""
        face = DetectedFace(
            bbox=(10, 20, 110, 220),
            confidence=0.95,
        )
        # (110-10) * (220-20) = 100 * 200 = 20000
        assert face.area == 20000

    def test_center_calculation(self):
        """Test bounding box center calculation."""
        face = DetectedFace(
            bbox=(0, 0, 100, 100),
            confidence=0.95,
        )
        assert face.center == (50, 50)

    def test_center_offset_bbox(self):
        """Test center calculation for offset bbox."""
        face = DetectedFace(
            bbox=(100, 200, 200, 400),
            confidence=0.95,
        )
        # center = ((100+200)/2, (200+400)/2) = (150, 300)
        assert face.center == (150, 300)

    def test_width_height(self):
        """Test width and height properties."""
        face = DetectedFace(
            bbox=(10, 20, 110, 220),
            confidence=0.95,
        )
        assert face.width == 100
        assert face.height == 200


class TestFaceDetectorInit:
    """Tests for FaceDetector initialization."""

    @patch("server.perception.vision.face_detector.MTCNN")
    def test_init_default_device_cpu(self, mock_mtcnn):
        """Test initialization defaults to CPU when CUDA unavailable."""
        with patch.object(torch.cuda, "is_available", return_value=False):
            detector = FaceDetector()
            assert detector.device == "cpu"

    @patch("server.perception.vision.face_detector.MTCNN")
    def test_init_explicit_device(self, mock_mtcnn):
        """Test initialization with explicit device."""
        detector = FaceDetector(device="cpu")
        assert detector.device == "cpu"

    @patch("server.perception.vision.face_detector.MTCNN")
    def test_init_custom_params(self, mock_mtcnn):
        """Test initialization with custom parameters."""
        detector = FaceDetector(
            device="cpu",
            min_face_size=60,
            confidence_threshold=0.8,
            image_size=224,
        )
        mock_mtcnn.assert_called_once()
        call_kwargs = mock_mtcnn.call_args[1]
        assert call_kwargs["min_face_size"] == 60
        assert call_kwargs["image_size"] == 224


class TestFaceDetectorDetect:
    """Tests for face detection."""

    @patch("server.perception.vision.face_detector.MTCNN")
    def test_detect_no_faces(self, mock_mtcnn_class):
        """Test detection with no faces returns empty list."""
        mock_mtcnn = MagicMock()
        mock_mtcnn.detect.return_value = (None, None, None)
        mock_mtcnn_class.return_value = mock_mtcnn

        detector = FaceDetector(device="cpu")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = detector.detect(frame)

        assert faces == []

    @patch("server.perception.vision.face_detector.MTCNN")
    def test_detect_single_face(self, mock_mtcnn_class):
        """Test detection with single face."""
        mock_mtcnn = MagicMock()
        # Mock detection returns
        boxes = np.array([[100, 100, 200, 200]])
        probs = np.array([0.95])
        landmarks = np.array([[[120, 130], [180, 130], [150, 160], [125, 185], [175, 185]]])
        mock_mtcnn.detect.return_value = (boxes, probs, landmarks)
        mock_mtcnn.return_value = None  # No aligned faces
        mock_mtcnn_class.return_value = mock_mtcnn

        detector = FaceDetector(device="cpu")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = detector.detect(frame, return_aligned=False)

        assert len(faces) == 1
        assert faces[0].confidence == 0.95
        assert faces[0].bbox == (100.0, 100.0, 200.0, 200.0)

    @patch("server.perception.vision.face_detector.MTCNN")
    def test_detect_filters_low_confidence(self, mock_mtcnn_class):
        """Test that low confidence detections are filtered."""
        mock_mtcnn = MagicMock()
        boxes = np.array([[100, 100, 200, 200], [300, 100, 400, 200]])
        probs = np.array([0.95, 0.5])  # Second face below threshold
        landmarks = np.array([
            [[120, 130], [180, 130], [150, 160], [125, 185], [175, 185]],
            [[320, 130], [380, 130], [350, 160], [325, 185], [375, 185]],
        ])
        mock_mtcnn.detect.return_value = (boxes, probs, landmarks)
        mock_mtcnn.return_value = None
        mock_mtcnn_class.return_value = mock_mtcnn

        detector = FaceDetector(device="cpu", confidence_threshold=0.9)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = detector.detect(frame, return_aligned=False)

        # Only high confidence face should be returned
        assert len(faces) == 1
        assert faces[0].confidence == 0.95

    @patch("server.perception.vision.face_detector.MTCNN")
    def test_detect_sorted_by_area(self, mock_mtcnn_class):
        """Test that faces are sorted by area (largest first)."""
        mock_mtcnn = MagicMock()
        # First face is smaller
        boxes = np.array([[100, 100, 150, 150], [300, 100, 500, 300]])
        probs = np.array([0.95, 0.95])
        landmarks = np.array([
            [[110, 120], [140, 120], [125, 135], [115, 145], [135, 145]],
            [[350, 150], [450, 150], [400, 200], [360, 250], [440, 250]],
        ])
        mock_mtcnn.detect.return_value = (boxes, probs, landmarks)
        mock_mtcnn.return_value = None
        mock_mtcnn_class.return_value = mock_mtcnn

        detector = FaceDetector(device="cpu")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = detector.detect(frame, return_aligned=False)

        assert len(faces) == 2
        # Larger face (200x200 = 40000) should be first
        assert faces[0].area > faces[1].area

    def test_detect_invalid_input(self):
        """Test detection with invalid input raises error."""
        with patch("server.perception.vision.face_detector.MTCNN"):
            detector = FaceDetector(device="cpu")

            # Grayscale image
            with pytest.raises(ValueError, match="Expected RGB image"):
                detector.detect(np.zeros((480, 640), dtype=np.uint8))

            # 4 channel image
            with pytest.raises(ValueError, match="Expected RGB image"):
                detector.detect(np.zeros((480, 640, 4), dtype=np.uint8))


class TestSelectPrimaryFace:
    """Tests for primary face selection."""

    def test_select_largest(self):
        """Test selecting largest face."""
        faces = [
            DetectedFace(bbox=(0, 0, 50, 50), confidence=0.95),
            DetectedFace(bbox=(100, 100, 300, 300), confidence=0.90),
            DetectedFace(bbox=(400, 100, 480, 180), confidence=0.98),
        ]
        frame_shape = (480, 640)

        primary = FaceDetector.select_primary_face(faces, frame_shape, "largest")

        assert primary is not None
        assert primary.area == 40000  # 200x200

    def test_select_central(self):
        """Test selecting most central face."""
        faces = [
            DetectedFace(bbox=(0, 0, 50, 50), confidence=0.95),  # Corner
            DetectedFace(bbox=(295, 215, 345, 265), confidence=0.90),  # Center
            DetectedFace(bbox=(600, 400, 640, 440), confidence=0.98),  # Far corner
        ]
        frame_shape = (480, 640)  # Center at (320, 240)

        primary = FaceDetector.select_primary_face(faces, frame_shape, "central")

        assert primary is not None
        # Center face bbox center is at (320, 240)
        assert primary.center == (320, 240)

    def test_select_most_confident(self):
        """Test selecting most confident face."""
        faces = [
            DetectedFace(bbox=(0, 0, 100, 100), confidence=0.85),
            DetectedFace(bbox=(100, 100, 200, 200), confidence=0.98),
            DetectedFace(bbox=(200, 200, 300, 300), confidence=0.90),
        ]
        frame_shape = (480, 640)

        primary = FaceDetector.select_primary_face(faces, frame_shape, "most_confident")

        assert primary is not None
        assert primary.confidence == 0.98

    def test_select_empty_list(self):
        """Test selection with empty list returns None."""
        primary = FaceDetector.select_primary_face([], (480, 640), "largest")
        assert primary is None

    def test_select_invalid_preference(self):
        """Test selection with invalid preference raises error."""
        faces = [DetectedFace(bbox=(0, 0, 100, 100), confidence=0.95)]
        with pytest.raises(ValueError, match="Unknown preference"):
            FaceDetector.select_primary_face(faces, (480, 640), "invalid")  # type: ignore


class TestComputeIoU:
    """Tests for IoU computation."""

    @patch("server.perception.vision.face_detector.MTCNN")
    def test_iou_no_overlap(self, mock_mtcnn):
        """Test IoU of non-overlapping boxes."""
        detector = FaceDetector(device="cpu")
        bbox1 = (0, 0, 100, 100)
        bbox2 = (200, 200, 300, 300)
        assert detector.compute_iou(bbox1, bbox2) == 0.0

    @patch("server.perception.vision.face_detector.MTCNN")
    def test_iou_same_box(self, mock_mtcnn):
        """Test IoU of identical boxes."""
        detector = FaceDetector(device="cpu")
        bbox = (0, 0, 100, 100)
        assert detector.compute_iou(bbox, bbox) == 1.0

    @patch("server.perception.vision.face_detector.MTCNN")
    def test_iou_partial_overlap(self, mock_mtcnn):
        """Test IoU of partially overlapping boxes."""
        detector = FaceDetector(device="cpu")
        bbox1 = (0, 0, 100, 100)  # Area = 10000
        bbox2 = (50, 50, 150, 150)  # Area = 10000, overlap = 50x50 = 2500
        # Union = 10000 + 10000 - 2500 = 17500
        # IoU = 2500 / 17500 = ~0.143
        iou = detector.compute_iou(bbox1, bbox2)
        assert abs(iou - 0.143) < 0.01

    @patch("server.perception.vision.face_detector.MTCNN")
    def test_iou_one_inside_other(self, mock_mtcnn):
        """Test IoU when one box is inside the other."""
        detector = FaceDetector(device="cpu")
        bbox1 = (0, 0, 100, 100)  # Area = 10000
        bbox2 = (25, 25, 75, 75)  # Area = 2500, fully inside
        # Intersection = 2500, Union = 10000
        # IoU = 2500 / 10000 = 0.25
        iou = detector.compute_iou(bbox1, bbox2)
        assert abs(iou - 0.25) < 0.01


@pytest.mark.hardware
class TestFaceDetectorIntegration:
    """Integration tests requiring actual MTCNN model."""

    def test_detect_real_model_no_faces(self):
        """Test detection on blank image with real model."""
        detector = FaceDetector(device="cpu")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = detector.detect(frame)
        assert faces == []

    def test_detect_real_model_random_noise(self):
        """Test detection on random noise (unlikely to have faces)."""
        detector = FaceDetector(device="cpu")
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        faces = detector.detect(frame)
        # May or may not detect false positives, but shouldn't crash
        assert isinstance(faces, list)
