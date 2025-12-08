"""
Unit tests for Murph's Face Encoder.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from server.perception.vision import DetectedFace, FaceEncoding, FaceEncoder


class TestFaceEncoding:
    """Tests for FaceEncoding dataclass."""

    def test_valid_embedding(self):
        """Test creation with valid 128-dim embedding."""
        embedding = np.random.randn(128).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        face = DetectedFace(bbox=(0, 0, 100, 100), confidence=0.95)

        encoding = FaceEncoding(
            embedding=embedding,
            quality_score=0.8,
            face=face,
        )

        assert encoding.embedding.shape == (128,)
        assert encoding.quality_score == 0.8

    def test_invalid_embedding_shape(self):
        """Test that wrong embedding dimension raises error."""
        embedding = np.random.randn(512).astype(np.float32)  # Wrong dim
        face = DetectedFace(bbox=(0, 0, 100, 100), confidence=0.95)

        with pytest.raises(ValueError, match="Expected 128-dim"):
            FaceEncoding(
                embedding=embedding,
                quality_score=0.8,
                face=face,
            )

    def test_cosine_similarity_same(self):
        """Test cosine similarity of identical embeddings."""
        embedding = np.random.randn(128).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        face = DetectedFace(bbox=(0, 0, 100, 100), confidence=0.95)

        enc1 = FaceEncoding(embedding=embedding.copy(), quality_score=0.8, face=face)
        enc2 = FaceEncoding(embedding=embedding.copy(), quality_score=0.8, face=face)

        similarity = enc1.cosine_similarity(enc2)
        assert abs(similarity - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity of orthogonal embeddings."""
        # Create orthogonal vectors
        embedding1 = np.zeros(128, dtype=np.float32)
        embedding1[0] = 1.0
        embedding2 = np.zeros(128, dtype=np.float32)
        embedding2[1] = 1.0

        face = DetectedFace(bbox=(0, 0, 100, 100), confidence=0.95)

        enc1 = FaceEncoding(embedding=embedding1, quality_score=0.8, face=face)
        enc2 = FaceEncoding(embedding=embedding2, quality_score=0.8, face=face)

        similarity = enc1.cosine_similarity(enc2)
        assert abs(similarity) < 1e-6

    def test_cosine_similarity_opposite(self):
        """Test cosine similarity of opposite embeddings."""
        embedding1 = np.random.randn(128).astype(np.float32)
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = -embedding1

        face = DetectedFace(bbox=(0, 0, 100, 100), confidence=0.95)

        enc1 = FaceEncoding(embedding=embedding1, quality_score=0.8, face=face)
        enc2 = FaceEncoding(embedding=embedding2, quality_score=0.8, face=face)

        similarity = enc1.cosine_similarity(enc2)
        assert abs(similarity - (-1.0)) < 1e-6


class TestProjectionLayer:
    """Tests for the 512->128 projection layer."""

    def test_projection_output_shape(self):
        """Test projection produces 128-dim output."""
        from server.perception.vision.face_encoder import ProjectionLayer

        projection = ProjectionLayer(512, 128)
        input_tensor = torch.randn(1, 512)
        output = projection(input_tensor)

        assert output.shape == (1, 128)

    def test_projection_batch(self):
        """Test projection with batch input."""
        from server.perception.vision.face_encoder import ProjectionLayer

        projection = ProjectionLayer(512, 128)
        input_tensor = torch.randn(4, 512)
        output = projection(input_tensor)

        assert output.shape == (4, 128)


class TestFaceEncoderInit:
    """Tests for FaceEncoder initialization."""

    @patch("server.perception.vision.face_encoder.InceptionResnetV1")
    def test_init_default_device(self, mock_inception):
        """Test initialization with default device."""
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_inception.return_value = mock_model

        with patch.object(torch.cuda, "is_available", return_value=False):
            encoder = FaceEncoder()
            assert encoder.device == "cpu"

    @patch("server.perception.vision.face_encoder.InceptionResnetV1")
    def test_init_explicit_pretrained(self, mock_inception):
        """Test initialization with explicit pretrained model."""
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_inception.return_value = mock_model

        encoder = FaceEncoder(device="cpu", pretrained="casia-webface")

        mock_inception.assert_called_once_with(pretrained="casia-webface")


class TestFaceEncoderEncode:
    """Tests for face encoding."""

    @patch("server.perception.vision.face_encoder.InceptionResnetV1")
    def test_encode_produces_128_dim(self, mock_inception):
        """Test encoding produces 128-dim vector."""
        # Setup mock
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_model.return_value = torch.randn(1, 512)
        mock_inception.return_value = mock_model

        encoder = FaceEncoder(device="cpu")

        face = DetectedFace(
            bbox=(0, 0, 160, 160),
            confidence=0.95,
            face_image=np.random.randint(0, 256, (160, 160, 3), dtype=np.uint8),
        )

        encoding = encoder.encode(face)

        assert encoding.embedding.shape == (128,)

    @patch("server.perception.vision.face_encoder.InceptionResnetV1")
    def test_encode_normalized(self, mock_inception):
        """Test encoding is L2-normalized."""
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_model.return_value = torch.randn(1, 512) * 10  # Scale up
        mock_inception.return_value = mock_model

        encoder = FaceEncoder(device="cpu")

        face = DetectedFace(
            bbox=(0, 0, 160, 160),
            confidence=0.95,
            face_image=np.random.randint(0, 256, (160, 160, 3), dtype=np.uint8),
        )

        encoding = encoder.encode(face)

        # L2 norm should be approximately 1
        norm = np.linalg.norm(encoding.embedding)
        assert abs(norm - 1.0) < 1e-5

    @patch("server.perception.vision.face_encoder.InceptionResnetV1")
    def test_encode_no_face_image_raises(self, mock_inception):
        """Test encoding without face_image raises error."""
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_inception.return_value = mock_model

        encoder = FaceEncoder(device="cpu")

        face = DetectedFace(
            bbox=(0, 0, 160, 160),
            confidence=0.95,
            face_image=None,  # No image
        )

        with pytest.raises(ValueError, match="must have face_image"):
            encoder.encode(face)

    @patch("server.perception.vision.face_encoder.InceptionResnetV1")
    def test_encode_batch(self, mock_inception):
        """Test batch encoding of multiple faces."""
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_model.return_value = torch.randn(3, 512)
        mock_inception.return_value = mock_model

        encoder = FaceEncoder(device="cpu")

        faces = [
            DetectedFace(
                bbox=(0, 0, 160, 160),
                confidence=0.95,
                face_image=np.random.randint(0, 256, (160, 160, 3), dtype=np.uint8),
            )
            for _ in range(3)
        ]

        encodings = encoder.encode_batch(faces)

        assert len(encodings) == 3
        for enc in encodings:
            assert enc.embedding.shape == (128,)

    @patch("server.perception.vision.face_encoder.InceptionResnetV1")
    def test_encode_batch_empty(self, mock_inception):
        """Test batch encoding with empty list."""
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_inception.return_value = mock_model

        encoder = FaceEncoder(device="cpu")
        encodings = encoder.encode_batch([])

        assert encodings == []

    @patch("server.perception.vision.face_encoder.InceptionResnetV1")
    def test_encode_batch_filters_no_image(self, mock_inception):
        """Test batch encoding filters faces without images."""
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_model.return_value = torch.randn(2, 512)
        mock_inception.return_value = mock_model

        encoder = FaceEncoder(device="cpu")

        faces = [
            DetectedFace(
                bbox=(0, 0, 160, 160),
                confidence=0.95,
                face_image=np.random.randint(0, 256, (160, 160, 3), dtype=np.uint8),
            ),
            DetectedFace(
                bbox=(100, 100, 260, 260),
                confidence=0.90,
                face_image=None,  # No image - should be filtered
            ),
            DetectedFace(
                bbox=(200, 200, 360, 360),
                confidence=0.92,
                face_image=np.random.randint(0, 256, (160, 160, 3), dtype=np.uint8),
            ),
        ]

        encodings = encoder.encode_batch(faces)

        assert len(encodings) == 2


class TestQualityEstimation:
    """Tests for embedding quality estimation."""

    @patch("server.perception.vision.face_encoder.InceptionResnetV1")
    def test_quality_high_confidence_large_face(self, mock_inception):
        """Test quality estimation for high confidence, large face."""
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_inception.return_value = mock_model

        encoder = FaceEncoder(device="cpu")

        face = DetectedFace(
            bbox=(0, 0, 300, 300),  # Large face
            confidence=0.99,  # High confidence
            face_image=np.random.randint(0, 256, (160, 160, 3), dtype=np.uint8),
        )
        embedding = np.random.randn(128).astype(np.float32)

        quality = encoder.estimate_quality(face, embedding)

        assert quality > 0.7  # Should be high quality

    @patch("server.perception.vision.face_encoder.InceptionResnetV1")
    def test_quality_low_confidence_small_face(self, mock_inception):
        """Test quality estimation for low confidence, small face."""
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_inception.return_value = mock_model

        encoder = FaceEncoder(device="cpu")

        face = DetectedFace(
            bbox=(0, 0, 45, 45),  # Small face (just above min)
            confidence=0.91,  # Lower confidence
            face_image=np.random.randint(0, 256, (160, 160, 3), dtype=np.uint8),
        )
        embedding = np.random.randn(128).astype(np.float32)

        quality = encoder.estimate_quality(face, embedding)

        assert quality < 0.5  # Should be lower quality

    @patch("server.perception.vision.face_encoder.InceptionResnetV1")
    def test_quality_bounds(self, mock_inception):
        """Test quality is always in [0, 1] range."""
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_inception.return_value = mock_model

        encoder = FaceEncoder(device="cpu")
        embedding = np.random.randn(128).astype(np.float32)

        # Test extreme cases
        face_tiny = DetectedFace(bbox=(0, 0, 10, 10), confidence=0.5)
        face_huge = DetectedFace(bbox=(0, 0, 1000, 1000), confidence=1.0)

        q_tiny = encoder.estimate_quality(face_tiny, embedding)
        q_huge = encoder.estimate_quality(face_huge, embedding)

        assert 0.0 <= q_tiny <= 1.0
        assert 0.0 <= q_huge <= 1.0


@pytest.mark.hardware
class TestFaceEncoderIntegration:
    """Integration tests requiring actual model."""

    def test_encode_real_model(self):
        """Test encoding with real InceptionResnetV1 model."""
        encoder = FaceEncoder(device="cpu", pretrained="vggface2")

        face = DetectedFace(
            bbox=(0, 0, 160, 160),
            confidence=0.95,
            face_image=np.random.randint(0, 256, (160, 160, 3), dtype=np.uint8),
        )

        encoding = encoder.encode(face)

        assert encoding.embedding.shape == (128,)
        # Check normalization
        norm = np.linalg.norm(encoding.embedding)
        assert abs(norm - 1.0) < 1e-5

    def test_similar_images_similar_embeddings(self):
        """Test that similar images produce similar embeddings."""
        encoder = FaceEncoder(device="cpu", pretrained="vggface2")

        # Create a base image
        base_image = np.random.randint(100, 150, (160, 160, 3), dtype=np.uint8)

        # Create slightly modified versions
        image1 = base_image.copy()
        image2 = base_image.copy()
        image2 = np.clip(image2.astype(np.int32) + 5, 0, 255).astype(np.uint8)

        face1 = DetectedFace(bbox=(0, 0, 160, 160), confidence=0.95, face_image=image1)
        face2 = DetectedFace(bbox=(0, 0, 160, 160), confidence=0.95, face_image=image2)

        enc1 = encoder.encode(face1)
        enc2 = encoder.encode(face2)

        similarity = enc1.cosine_similarity(enc2)
        # Similar images should have high similarity
        assert similarity > 0.9
