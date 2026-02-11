"""
Pytest configuration and shared fixtures.
This file is automatically discovered by pytest and makes fixtures available to all test files.
"""

import pytest
import numpy as np
from pathlib import Path
import cv2
from unittest.mock import Mock, MagicMock
from app.models.insightface import InsightFaceEmbedder, Device


# ============================================================================
# Test Data Directory
# ============================================================================

@pytest.fixture(scope="session")
def test_data_dir():
    """
    Path to test data directory.
    Scope: session - created once per test session.
    """
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


# ============================================================================
# Mock Face Objects (for unit testing without real model)
# ============================================================================

@pytest.fixture
def mock_face_single():
    """
    Mock Face object representing a single detected face.
    Use this to test without running actual model inference.
    """
    face = Mock()
    face.embedding = np.random.randn(512).astype(np.float32)  # 512-dim embedding
    face.det_score = 0.95  # High confidence score
    face.bbox = np.array([100, 100, 300, 300])  # [x1, y1, x2, y2]
    face.kps = np.random.randn(5, 2).astype(np.float32)  # 5 keypoints
    return face


@pytest.fixture
def mock_face_low_confidence():
    """Mock Face object with low detection confidence."""
    face = Mock()
    face.embedding = np.random.randn(512).astype(np.float32)
    face.det_score = 0.3  # Low confidence
    face.bbox = np.array([100, 100, 300, 300])
    face.kps = np.random.randn(5, 2).astype(np.float32)
    return face


# ============================================================================
# Synthetic Test Images
# ============================================================================

@pytest.fixture
def single_face_image():
    """
    Generate a synthetic image for testing.
    In production, replace with real test image.
    Returns: BGR uint8 numpy array (H, W, 3)
    """
    # Create a simple colored image (640x480)
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    # Draw a simple "face" (circle with features)
    # This won't actually be detected by InsightFace, but serves as placeholder
    center = (320, 240)
    cv2.circle(img, center, 80, (255, 200, 150), -1)  # Face circle
    cv2.circle(img, (280, 220), 10, (0, 0, 0), -1)  # Left eye
    cv2.circle(img, (360, 220), 10, (0, 0, 0), -1)  # Right eye
    cv2.ellipse(img, (320, 270), (30, 15), 0, 0, 180, (0, 0, 0), 2)  # Mouth

    return img


@pytest.fixture
def no_face_image():
    """
    Generate an image with no faces (blank/landscape).
    Returns: BGR uint8 numpy array
    """
    # Simple gradient background
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(480):
        # Ensure values don't exceed 255
        img[i, :] = [(100 + i // 3) % 256, 150, (200 - i // 3) % 256]
    return img


@pytest.fixture
def multi_face_image():
    """
    Generate an image with multiple faces.
    Returns: BGR uint8 numpy array
    """
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    # Draw two "faces"
    centers = [(200, 240), (440, 240)]
    for center in centers:
        cv2.circle(img, center, 60, (255, 200, 150), -1)
        cv2.circle(img, (center[0] - 25, center[1] - 20), 8, (0, 0, 0), -1)
        cv2.circle(img, (center[0] + 25, center[1] - 20), 8, (0, 0, 0), -1)

    return img


@pytest.fixture
def invalid_format_image():
    """
    Image in wrong format (RGB instead of BGR, float instead of uint8).
    Useful for testing format validation.
    """
    # RGB, float32, [0, 1] range
    img = np.random.rand(480, 640, 3).astype(np.float32)
    return img


# ============================================================================
# Embedder Instances
# ============================================================================

@pytest.fixture
def embedder_cpu():
    """
    InsightFaceEmbedder instance with CPU device.
    Scope: function - new instance for each test.

    Note: This will download models on first run (~200MB).
    For faster tests, consider mocking in unit tests.
    """
    return InsightFaceEmbedder(device=Device.CPU)


@pytest.fixture
def embedder_gpu():
    """
    InsightFaceEmbedder instance with GPU device.
    Will fall back to CPU if CUDA not available.
    """
    return InsightFaceEmbedder(device=Device.GPU)


@pytest.fixture
def mock_embedder(monkeypatch, mock_face_single):
    """
    Mocked InsightFaceEmbedder for fast unit tests without model inference.
    """
    # Create a mock for FaceAnalysis
    mock_app = MagicMock()
    mock_app.get.return_value = [mock_face_single]
    
    # Patch the class constructor or the instance
    with monkeypatch.context() as m:
        m.setattr("app.models.insightface.FaceAnalysis", MagicMock(return_value=mock_app))
        embedder = InsightFaceEmbedder(device=Device.CPU)
    
    return embedder


@pytest.fixture
def mock_embedder_no_face(monkeypatch):
    """Mocked embedder that returns no faces."""
    mock_app = MagicMock()
    mock_app.get.return_value = []
    
    with monkeypatch.context() as m:
        m.setattr("app.models.insightface.FaceAnalysis", MagicMock(return_value=mock_app))
        embedder = InsightFaceEmbedder(device=Device.CPU)
    
    return embedder


@pytest.fixture
def mock_embedder_multi_face(monkeypatch):
    """Mocked embedder that returns multiple faces."""
    face1 = Mock()
    face1.embedding = np.random.randn(512).astype(np.float32)
    face1.det_score = 0.92

    face2 = Mock()
    face2.embedding = np.random.randn(512).astype(np.float32)
    face2.det_score = 0.88

    mock_app = MagicMock()
    mock_app.get.return_value = [face1, face2]
    
    with monkeypatch.context() as m:
        m.setattr("app.models.insightface.FaceAnalysis", MagicMock(return_value=mock_app))
        embedder = InsightFaceEmbedder(device=Device.CPU)
    
    return embedder


# ============================================================================
# Utility Functions
# ============================================================================

def is_gpu_available():
    """Check if CUDA GPU is available."""
    try:
        import onnxruntime as ort
        return 'CUDAExecutionProvider' in ort.get_available_providers()
    except ImportError:
        return False


@pytest.fixture(scope="session")
def gpu_available():
    """Fixture that returns True if GPU is available."""
    return is_gpu_available()