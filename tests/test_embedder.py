"""
Unit and integration tests for InsightFaceEmbedder.

Test Categories:
- Unit tests: Use mocked embedder (fast, no model downloads)
- Integration tests: Use real embedder (slow, requires model)
- Edge cases: Error handling and validation
"""

import pytest
import numpy as np
from pydantic import ValidationError
from app.models.insightface import (
    InsightFaceEmbedder,
    Device,
    NoFaceDetectedError,
    MultipleFacesDetectedError,
    FaceEmbedding,
)


# ============================================================================
# UNIT TESTS - Fast tests using mocks
# ============================================================================

class TestEmbedderUnitTests:
    """Unit tests using mocked dependencies (fast, no model required)."""

    def test_single_face_detection_success(self, mock_embedder, single_face_image):
        """
        Test successful embedding extraction from single face.
        Uses mock to avoid model inference.
        """
        result = mock_embedder.embed(single_face_image)

        # Verify return type
        assert isinstance(result, FaceEmbedding)

        # Verify embedding properties
        assert isinstance(result.embedding, np.ndarray)
        assert result.embedding.shape == (512,)
        assert result.embedding.dtype in [np.float32, np.float64]

        # Verify detection score
        assert isinstance(result.detection_score, float)
        assert 0.0 <= result.detection_score <= 1.0

    def test_no_face_raises_error(self, mock_embedder_no_face, no_face_image):
        """
        Test that NoFaceDetectedError is raised when no face detected.
        """
        with pytest.raises(NoFaceDetectedError) as exc_info:
            mock_embedder_no_face.embed(no_face_image)

        # Verify error message
        assert "No face detected" in str(exc_info.value)

    def test_multiple_faces_raises_error(self, mock_embedder_multi_face, multi_face_image):
        """
        Test that MultipleFacesDetectedError is raised for multiple faces.
        """
        with pytest.raises(MultipleFacesDetectedError) as exc_info:
            mock_embedder_multi_face.embed(multi_face_image)

        # Verify exception has num_faces attribute
        assert exc_info.value.num_faces == 2

        # Verify error message contains face count
        assert "2 faces" in str(exc_info.value)

    def test_embedding_dtype_is_numeric(self, mock_embedder, single_face_image):
        """Verify embedding is numeric (float32 or float64)."""
        result = mock_embedder.embed(single_face_image)
        assert np.issubdtype(result.embedding.dtype, np.floating)

    def test_detection_score_range(self, mock_embedder, single_face_image):
        """Verify detection score is between 0 and 1."""
        result = mock_embedder.embed(single_face_image)
        assert 0.0 <= result.detection_score <= 1.0

    def test_embedding_not_all_zeros(self, mock_embedder, single_face_image):
        """Verify embedding contains meaningful values (not all zeros)."""
        result = mock_embedder.embed(single_face_image)
        assert not np.allclose(result.embedding, 0.0)

    def test_embedding_has_variance(self, mock_embedder, single_face_image):
        """Verify embedding has variance (not constant values)."""
        result = mock_embedder.embed(single_face_image)
        assert np.std(result.embedding) > 0.0


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================

class TestEmbedderInitialization:
    """Test embedder initialization with different configurations."""

    def test_cpu_device_initialization(self):
        """Test CPU initialization."""
        embedder = InsightFaceEmbedder(device=Device.CPU)

        assert embedder.device == Device.CPU
        assert embedder.app is not None
        assert embedder.model_name == 'buffalo_l'  # default

    def test_gpu_device_initialization(self):
        """Test GPU initialization (falls back to CPU if no GPU)."""
        embedder = InsightFaceEmbedder(device=Device.GPU)

        assert embedder.device == Device.GPU
        assert embedder.app is not None

    def test_custom_model_name(self):
        """Test initialization with custom model name."""
        embedder = InsightFaceEmbedder(model_name='buffalo_sc', device=Device.CPU)
        assert embedder.model_name == 'buffalo_sc'

    def test_default_device_is_cpu(self):
        """Test that default device is CPU."""
        embedder = InsightFaceEmbedder()
        assert embedder.device == Device.CPU


# ============================================================================
# PYDANTIC SCHEMA TESTS
# ============================================================================

class TestFaceEmbeddingSchema:
    """Test Pydantic validation for FaceEmbedding schema."""

    def test_valid_face_embedding_creation(self):
        """Test creating valid FaceEmbedding instance."""
        embedding = np.random.randn(512).astype(np.float32)
        face_emb = FaceEmbedding(embedding=embedding, detection_score=0.95)

        assert face_emb.detection_score == 0.95
        assert np.array_equal(face_emb.embedding, embedding)

    def test_detection_score_lower_bound(self):
        """Test that detection score cannot be negative."""
        embedding = np.random.randn(512).astype(np.float32)

        with pytest.raises(ValidationError):
            FaceEmbedding(embedding=embedding, detection_score=-0.1)

    def test_detection_score_upper_bound(self):
        """Test that detection score cannot exceed 1.0."""
        embedding = np.random.randn(512).astype(np.float32)

        with pytest.raises(ValidationError):
            FaceEmbedding(embedding=embedding, detection_score=1.5)

    def test_detection_score_boundary_values(self):
        """Test boundary values (0.0 and 1.0) are valid."""
        embedding = np.random.randn(512).astype(np.float32)

        # Should not raise
        face_emb_zero = FaceEmbedding(embedding=embedding, detection_score=0.0)
        face_emb_one = FaceEmbedding(embedding=embedding, detection_score=1.0)

        assert face_emb_zero.detection_score == 0.0
        assert face_emb_one.detection_score == 1.0

    def test_embedding_required_field(self):
        """Test that embedding is a required field."""
        with pytest.raises(ValidationError):
            FaceEmbedding(detection_score=0.95)

    def test_detection_score_required_field(self):
        """Test that detection_score is a required field."""
        embedding = np.random.randn(512).astype(np.float32)

        with pytest.raises(ValidationError):
            FaceEmbedding(embedding=embedding)


# ============================================================================
# CUSTOM EXCEPTION TESTS
# ============================================================================

class TestCustomExceptions:
    """Test custom exception behavior."""

    def test_no_face_detected_error_message(self):
        """Test NoFaceDetectedError can be raised with custom message."""
        error = NoFaceDetectedError("Custom message")
        assert str(error) == "Custom message"

    def test_multiple_faces_error_stores_count(self):
        """Test MultipleFacesDetectedError stores face count."""
        error = MultipleFacesDetectedError(num_faces=3)
        assert error.num_faces == 3

    def test_multiple_faces_error_message_format(self):
        """Test MultipleFacesDetectedError message formatting."""
        error = MultipleFacesDetectedError(num_faces=5)
        assert "Expected 1 face" in str(error)
        assert "detected 5 faces" in str(error)

    def test_exceptions_are_catchable(self):
        """Test that custom exceptions can be caught."""
        try:
            raise NoFaceDetectedError("Test")
        except NoFaceDetectedError as e:
            assert "Test" in str(e)

        try:
            raise MultipleFacesDetectedError(2)
        except MultipleFacesDetectedError as e:
            assert e.num_faces == 2


# ============================================================================
# INTEGRATION TESTS - Real model inference (slow)
# ============================================================================

@pytest.mark.slow
@pytest.mark.integration
class TestEmbedderIntegration:
    """
    Integration tests using real InsightFace model.
    These are marked as 'slow' and can be skipped with: pytest -m "not slow"

    Note: These require model downloads (~200MB) on first run.
    """

    def test_real_model_initialization(self, embedder_cpu):
        """Test that real model initializes successfully."""
        assert embedder_cpu.app is not None
        assert embedder_cpu.device == Device.CPU

    def test_embedding_consistency(self, embedder_cpu, single_face_image):
        """
        Test that same image produces consistent embeddings.
        Note: This uses synthetic image which may not be detected as a face.
        For production, use real face images.
        """
        # This test may need to be adjusted based on whether
        # the synthetic image is actually detected as a face
        try:
            result1 = embedder_cpu.embed(single_face_image)
            result2 = embedder_cpu.embed(single_face_image)

            # Embeddings should be identical for same image
            assert np.allclose(result1.embedding, result2.embedding)
            assert result1.detection_score == result2.detection_score
        except NoFaceDetectedError:
            # Synthetic image not detected, which is expected
            pytest.skip("Synthetic image not detected by real model")

    @pytest.mark.skipif(
        not pytest.importorskip("onnxruntime", reason="onnxruntime not installed"),
        reason="GPU tests require onnxruntime"
    )
    def test_gpu_fallback_to_cpu(self, embedder_gpu):
        """
        Test that GPU embedder falls back gracefully if CUDA unavailable.
        """
        assert embedder_gpu.app is not None
        # Should initialize even without GPU


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Test edge cases and unusual inputs."""

    def test_different_image_sizes(self, mock_embedder):
        """Test that different image sizes are handled."""
        sizes = [(224, 224, 3), (640, 480, 3), (1920, 1080, 3), (100, 100, 3)]

        for size in sizes:
            img = np.zeros(size, dtype=np.uint8)
            try:
                result = mock_embedder.embed(img)
                assert result.embedding.shape == (512,)
            except NoFaceDetectedError:
                # Expected if no face in blank image
                pass

    def test_grayscale_converted_to_bgr(self, mock_embedder):
        """
        Test handling of grayscale images.
        Note: User should convert to BGR before calling embed.
        """
        # Grayscale image (H, W)
        gray_img = np.zeros((480, 640), dtype=np.uint8)

        # InsightFace (and our implementation) expects (H, W, 3)
        # This should raise a ValueError due to channel mismatch
        with pytest.raises(ValueError, match="must have 3 channels"):
            mock_embedder.embed(gray_img)

    def test_invalid_input_type(self, mock_embedder):
        """Test handling of non-numpy array input."""
        with pytest.raises(ValueError, match="must be a numpy array"):
            mock_embedder.embed([1, 2, 3])  # List instead of array

    def test_empty_image(self, mock_embedder_no_face):
        """Test handling of completely black image."""
        empty_img = np.zeros((480, 640, 3), dtype=np.uint8)

        with pytest.raises(NoFaceDetectedError):
            mock_embedder_no_face.embed(empty_img)

    def test_very_small_image(self, mock_embedder):
        """Test handling of very small images."""
        tiny_img = np.zeros((10, 10, 3), dtype=np.uint8)

        # Should either work or raise NoFaceDetectedError
        # (depending on whether face is detected)
        try:
            result = mock_embedder.embed(tiny_img)
            assert isinstance(result, FaceEmbedding)
        except NoFaceDetectedError:
            pass  # Expected for tiny images


# ============================================================================
# PARAMETRIZED TESTS
# ============================================================================

class TestParametrized:
    """Parametrized tests for testing multiple scenarios efficiently."""

    @pytest.mark.parametrize("score", [0.0, 0.25, 0.5, 0.75, 0.99, 1.0])
    def test_various_detection_scores(self, score):
        """Test FaceEmbedding accepts various valid detection scores."""
        embedding = np.random.randn(512).astype(np.float32)
        face_emb = FaceEmbedding(embedding=embedding, detection_score=score)
        assert face_emb.detection_score == score

    @pytest.mark.parametrize("invalid_score", [-1.0, -0.001, 1.001, 2.0, 100.0])
    def test_invalid_detection_scores(self, invalid_score):
        """Test FaceEmbedding rejects invalid detection scores."""
        embedding = np.random.randn(512).astype(np.float32)

        with pytest.raises(ValidationError):
            FaceEmbedding(embedding=embedding, detection_score=invalid_score)

    @pytest.mark.parametrize("num_faces", [2, 3, 5, 10])
    def test_multiple_faces_various_counts(self, num_faces):
        """Test MultipleFacesDetectedError with various face counts."""
        error = MultipleFacesDetectedError(num_faces=num_faces)
        assert error.num_faces == num_faces
        assert str(num_faces) in str(error)

    @pytest.mark.parametrize("device", [Device.CPU, Device.GPU])
    def test_initialization_both_devices(self, device):
        """Test initialization works for both CPU and GPU."""
        embedder = InsightFaceEmbedder(device=device)
        assert embedder.device == device


# ============================================================================
# PERFORMANCE / BENCHMARK TESTS (Optional)
# ============================================================================

@pytest.mark.slow
class TestPerformance:
    """Performance and benchmark tests."""

    @pytest.mark.skipif(
        not pytest.importorskip("pytest_benchmark", reason="pytest-benchmark not installed"),
        reason="Performance tests require pytest-benchmark"
    )
    def test_embedding_speed_single_image(self, mock_embedder, single_face_image, benchmark):
        """
        Benchmark embedding extraction speed.
        Requires pytest-benchmark: pip install pytest-benchmark
        Run with: pytest --benchmark-only
        """
        # Only run if pytest-benchmark is available
        pytest.importorskip("pytest_benchmark")

        def run_embed():
            return mock_embedder.embed(single_face_image)

        result = benchmark(run_embed)
        assert isinstance(result, FaceEmbedding)