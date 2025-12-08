"""
Murph - Face Detector
MTCNN-based face detection optimized for video streaming.
"""

import logging
from typing import Literal

import numpy as np
import torch
from facenet_pytorch import MTCNN

from .types import DetectedFace

logger = logging.getLogger(__name__)


class FaceDetector:
    """
    Face detection using MTCNN from facenet-pytorch.

    Optimized for video streaming with configurable minimum face size
    and confidence thresholds.

    Usage:
        detector = FaceDetector()
        faces = detector.detect(frame)  # Returns list of DetectedFace
    """

    def __init__(
        self,
        device: str | None = None,
        min_face_size: int = 40,
        confidence_threshold: float = 0.9,
        image_size: int = 160,
    ) -> None:
        """
        Initialize face detector.

        Args:
            device: 'cuda' or 'cpu'. Auto-detects if None.
            min_face_size: Minimum face size in pixels (default 40)
            confidence_threshold: Detection confidence threshold (default 0.9)
            image_size: Output size for aligned face images (default 160)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._device = device
        self._min_face_size = min_face_size
        self._confidence_threshold = confidence_threshold
        self._image_size = image_size

        # Initialize MTCNN
        # keep_all=True returns all faces above threshold
        # selection_method=None disables internal selection
        self._mtcnn = MTCNN(
            image_size=image_size,
            margin=0,
            min_face_size=min_face_size,
            thresholds=[0.6, 0.7, 0.7],  # Default MTCNN thresholds for 3 stages
            factor=0.709,  # Scale factor for image pyramid
            post_process=True,  # Normalize output
            device=device,
            keep_all=True,
            selection_method=None,
        )

        logger.info(
            f"FaceDetector initialized on {device}, "
            f"min_size={min_face_size}, threshold={confidence_threshold}"
        )

    @property
    def device(self) -> str:
        """Return the compute device being used."""
        return self._device

    def detect(
        self,
        frame: np.ndarray,
        return_aligned: bool = True,
    ) -> list[DetectedFace]:
        """
        Detect faces in a video frame.

        Args:
            frame: RGB image as numpy array (H, W, 3), dtype uint8
            return_aligned: If True, include aligned face crops in results

        Returns:
            List of DetectedFace objects, sorted by area (largest first)
        """
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected RGB image (H, W, 3), got shape {frame.shape}")

        # MTCNN.detect returns (boxes, probs, landmarks)
        # boxes: Nx4 array of [x1, y1, x2, y2]
        # probs: N array of confidence scores
        # landmarks: Nx5x2 array of facial landmarks (if detect_landmarks=True)
        boxes, probs, landmarks = self._mtcnn.detect(frame, landmarks=True)

        if boxes is None:
            return []

        # Get aligned faces if requested
        aligned_faces = None
        if return_aligned:
            # MTCNN forward pass returns aligned faces as tensor
            aligned_tensor = self._mtcnn(frame)
            if aligned_tensor is not None:
                # Convert from (N, C, H, W) tensor to list of (H, W, C) numpy arrays
                aligned_faces = []
                for i in range(aligned_tensor.shape[0]):
                    # Tensor is normalized, convert back to 0-255 uint8
                    face_tensor = aligned_tensor[i]
                    # facenet-pytorch normalizes to [-1, 1] range with post_process=True
                    # Convert back: (x + 1) / 2 * 255
                    face_np = face_tensor.permute(1, 2, 0).cpu().numpy()
                    face_np = ((face_np + 1) / 2 * 255).astype(np.uint8)
                    aligned_faces.append(face_np)

        # Build DetectedFace objects
        faces = []
        for i in range(len(boxes)):
            prob = float(probs[i])

            # Filter by confidence threshold
            if prob < self._confidence_threshold:
                continue

            bbox = tuple(float(x) for x in boxes[i])
            face_landmarks = landmarks[i] if landmarks is not None else None
            face_image = aligned_faces[i] if aligned_faces and i < len(aligned_faces) else None

            face = DetectedFace(
                bbox=bbox,  # type: ignore[arg-type]
                confidence=prob,
                landmarks=face_landmarks,
                face_image=face_image,
            )
            faces.append(face)

        # Sort by area (largest first)
        faces.sort(key=lambda f: f.area, reverse=True)

        logger.debug(f"Detected {len(faces)} faces in frame")
        return faces

    @staticmethod
    def select_primary_face(
        faces: list[DetectedFace],
        frame_shape: tuple[int, int],
        preference: Literal["largest", "central", "most_confident"] = "largest",
    ) -> DetectedFace | None:
        """
        Select the primary face from multiple detections.

        Args:
            faces: List of detected faces
            frame_shape: (height, width) of source frame
            preference: Selection strategy:
                - "largest": Select face with largest bounding box
                - "central": Select face closest to frame center
                - "most_confident": Select face with highest confidence

        Returns:
            The selected face, or None if no faces provided
        """
        if not faces:
            return None

        if preference == "largest":
            return max(faces, key=lambda f: f.area)

        elif preference == "most_confident":
            return max(faces, key=lambda f: f.confidence)

        elif preference == "central":
            frame_center = (frame_shape[1] / 2, frame_shape[0] / 2)  # (cx, cy)

            def distance_to_center(face: DetectedFace) -> float:
                cx, cy = face.center
                return ((cx - frame_center[0]) ** 2 + (cy - frame_center[1]) ** 2) ** 0.5

            return min(faces, key=distance_to_center)

        else:
            raise ValueError(f"Unknown preference: {preference}")

    def compute_iou(
        self,
        bbox1: tuple[float, float, float, float],
        bbox2: tuple[float, float, float, float],
    ) -> float:
        """
        Compute Intersection over Union of two bounding boxes.

        Args:
            bbox1: First bbox as (x1, y1, x2, y2)
            bbox2: Second bbox as (x1, y1, x2, y2)

        Returns:
            IoU value between 0 and 1
        """
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
