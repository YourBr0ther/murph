"""
Murph - Face Encoder
InceptionResnetV1-based face encoding with 128-dim projection.
"""

import logging

import numpy as np
import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1

from .types import DetectedFace, FaceEncoding

logger = logging.getLogger(__name__)


class ProjectionLayer(nn.Module):
    """
    Linear projection from 512-dim to 128-dim embeddings.

    Initialized with orthogonal weights to preserve distance relationships.
    """

    def __init__(self, input_dim: int = 512, output_dim: int = 128) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        # Orthogonal initialization preserves norms and angles better
        nn.init.orthogonal_(self.linear.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class FaceEncoder:
    """
    Face encoding using InceptionResnetV1 from facenet-pytorch.

    Generates 128-dimensional embeddings compatible with the existing
    FaceEmbeddingModel storage (128 * 4 = 512 bytes for float32).

    Usage:
        encoder = FaceEncoder()
        encoding = encoder.encode(face)  # Returns FaceEncoding with 128-dim embedding
    """

    def __init__(
        self,
        device: str | None = None,
        pretrained: str = "vggface2",
    ) -> None:
        """
        Initialize face encoder.

        Args:
            device: 'cuda' or 'cpu'. Auto-detects if None.
            pretrained: Pretrained model weights:
                - 'vggface2': Trained on VGGFace2 dataset
                - 'casia-webface': Trained on CASIA-WebFace dataset
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._device = device

        # Load pretrained InceptionResnetV1
        self._model = InceptionResnetV1(pretrained=pretrained).eval().to(device)

        # Projection layer: 512 -> 128 dimensions
        self._projection = ProjectionLayer(512, 128).eval().to(device)

        logger.info(
            f"FaceEncoder initialized on {device}, pretrained={pretrained}"
        )

    @property
    def device(self) -> str:
        """Return the compute device being used."""
        return self._device

    def encode(self, face: DetectedFace) -> FaceEncoding:
        """
        Generate embedding for a detected face.

        Args:
            face: DetectedFace with aligned face_image (160x160 RGB)

        Returns:
            FaceEncoding with 128-dim L2-normalized embedding

        Raises:
            ValueError: If face has no aligned image
        """
        if face.face_image is None:
            raise ValueError("DetectedFace must have face_image for encoding")

        # Prepare input tensor
        face_tensor = self._prepare_input(face.face_image)

        # Get 512-dim embedding from InceptionResnetV1
        with torch.no_grad():
            embedding_512 = self._model(face_tensor)
            # Project to 128 dimensions
            embedding_128 = self._projection(embedding_512)

        # Convert to numpy and normalize
        embedding = embedding_128.cpu().numpy().flatten()
        embedding = self._l2_normalize(embedding)

        # Estimate quality
        quality = self.estimate_quality(face, embedding)

        return FaceEncoding(
            embedding=embedding,
            quality_score=quality,
            face=face,
        )

    def encode_batch(self, faces: list[DetectedFace]) -> list[FaceEncoding]:
        """
        Batch encoding for multiple faces.

        Args:
            faces: List of DetectedFace objects with face_image

        Returns:
            List of FaceEncoding objects
        """
        if not faces:
            return []

        # Filter faces without images
        valid_faces = [f for f in faces if f.face_image is not None]
        if not valid_faces:
            return []

        # Stack face images into batch tensor
        tensors = [self._prepare_input(f.face_image) for f in valid_faces]
        batch = torch.cat(tensors, dim=0)

        # Get embeddings
        with torch.no_grad():
            embeddings_512 = self._model(batch)
            embeddings_128 = self._projection(embeddings_512)

        # Convert to FaceEncoding objects
        encodings = []
        for i, face in enumerate(valid_faces):
            embedding = embeddings_128[i].cpu().numpy().flatten()
            embedding = self._l2_normalize(embedding)
            quality = self.estimate_quality(face, embedding)

            encodings.append(FaceEncoding(
                embedding=embedding,
                quality_score=quality,
                face=face,
            ))

        return encodings

    def estimate_quality(self, face: DetectedFace, embedding: np.ndarray) -> float:
        """
        Estimate embedding quality (0-1).

        Factors considered:
        - Detection confidence (higher = better quality image)
        - Face size (larger = more detail)
        - Embedding norm before normalization (unusual norms may indicate issues)

        Args:
            face: The detected face
            embedding: The generated embedding

        Returns:
            Quality score between 0 and 1
        """
        # Detection confidence contributes 40%
        confidence_score = face.confidence * 0.4

        # Face size contributes 30% (normalize assuming 40-400 pixel range)
        face_size = (face.width + face.height) / 2
        size_score = min(1.0, max(0.0, (face_size - 40) / 360)) * 0.3

        # Face area contributes 30% (larger faces have more detail)
        # Normalize assuming faces between 1600 (40x40) and 160000 (400x400) pixels
        area_score = min(1.0, max(0.0, (face.area - 1600) / 158400)) * 0.3

        quality = confidence_score + size_score + area_score

        return min(1.0, max(0.0, quality))

    def _prepare_input(self, face_image: np.ndarray) -> torch.Tensor:
        """
        Prepare face image for model input.

        Args:
            face_image: RGB image as numpy array (H, W, 3), uint8 or float

        Returns:
            Tensor of shape (1, 3, H, W) normalized to [-1, 1]
        """
        # Ensure float type
        if face_image.dtype == np.uint8:
            img = face_image.astype(np.float32) / 255.0
        else:
            img = face_image.astype(np.float32)

        # Normalize to [-1, 1] (InceptionResnetV1 expects this)
        img = (img - 0.5) / 0.5

        # Convert to tensor: (H, W, C) -> (1, C, H, W)
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

        return tensor.to(self._device)

    @staticmethod
    def _l2_normalize(embedding: np.ndarray) -> np.ndarray:
        """L2 normalize an embedding vector."""
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding
