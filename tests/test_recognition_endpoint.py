"""
Integration tests for the /recognize endpoint.
Tests the full API with mocked ML dependencies.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import io
from uuid import uuid4
import numpy as np

from app.main import app
from app.schemas.recognize_schema import RecognizeResponse


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def sample_image_bytes():
    """Generate sample JPEG bytes for upload."""
    # Minimal valid JPEG header
    return b'\xff\xd8\xff\xe0' + b'\x00' * 100


@pytest.fixture
def sample_embedding():
    """Normalized embedding."""
    emb = np.random.randn(512).astype(np.float32)
    return emb / np.linalg.norm(emb)


# ============================================================================
# Test: Successful Recognition via API
# ============================================================================

@patch('app.api.routes.recognize.recognize_user')
def test_recognize_endpoint_success(mock_recognize_user, client, sample_image_bytes):
    """Test /recognize endpoint returns match on success."""
    # Arrange
    expected_user_id = uuid4()
    mock_recognize_user.return_value = RecognizeResponse(
        match=True,
        user_id=expected_user_id,
        similarity=0.95
    )

    files = {"file": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}

    # Act
    response = client.post("/recognize", files=files)

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["match"] is True
    assert data["user_id"] == str(expected_user_id)
    assert data["similarity"] == 0.95


# ============================================================================
# Test: No Match Found
# ============================================================================

@patch('app.api.routes.recognize.recognize_user')
def test_recognize_endpoint_no_match(mock_recognize_user, client, sample_image_bytes):
    """Test /recognize endpoint returns no match."""
    # Arrange
    mock_recognize_user.return_value = RecognizeResponse(
        match=False,
        user_id=None,
        similarity=0.45
    )

    files = {"file": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}

    # Act
    response = client.post("/recognize", files=files)

    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["match"] is False
    assert data["user_id"] is None
    assert data["similarity"] == 0.45


# ============================================================================
# Test: File Too Large
# ============================================================================

def test_recognize_endpoint_file_too_large(client):
    """Test that files exceeding size limit are rejected."""
    # Arrange
    large_file = b'\xff\xd8\xff\xe0' + b'\x00' * (11 * 1024 * 1024)  # 11 MB
    files = {"file": ("large.jpg", io.BytesIO(large_file), "image/jpeg")}

    # Act
    response = client.post("/recognize", files=files)

    # Assert
    assert response.status_code == 413
    assert "too large" in response.json()["detail"].lower()


# ============================================================================
# Test: No Face Detected (422 Error)
# ============================================================================

@patch('app.api.routes.recognize.recognize_user')
def test_recognize_endpoint_no_face_detected(mock_recognize_user, client, sample_image_bytes):
    """Test that no face detection returns 422."""
    # Arrange
    from fastapi import HTTPException
    mock_recognize_user.side_effect = HTTPException(
        status_code=422,
        detail="No face detected in the provided image"
    )

    files = {"file": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}

    # Act
    response = client.post("/recognize", files=files)

    # Assert
    assert response.status_code == 422
    assert "No face detected" in response.json()["detail"]


# ============================================================================
# Test: Multiple Faces Detected
# ============================================================================

@patch('app.api.routes.recognize.recognize_user')
def test_recognize_endpoint_multiple_faces(mock_recognize_user, client, sample_image_bytes):
    """Test that multiple faces detection returns 422."""
    # Arrange
    from fastapi import HTTPException
    mock_recognize_user.side_effect = HTTPException(
        status_code=422,
        detail="Expected 1 face, but detected 2 faces"
    )

    files = {"file": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}

    # Act
    response = client.post("/recognize", files=files)

    # Assert
    assert response.status_code == 422
    assert "2 faces" in response.json()["detail"]


# ============================================================================
# Test: Database Error (500)
# ============================================================================

@patch('app.api.routes.recognize.recognize_user')
def test_recognize_endpoint_database_error(mock_recognize_user, client, sample_image_bytes):
    """Test that database errors return 500."""
    # Arrange
    from fastapi import HTTPException
    mock_recognize_user.side_effect = HTTPException(
        status_code=500,
        detail="Database error occurred"
    )

    files = {"file": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}

    # Act
    response = client.post("/recognize", files=files)

    # Assert
    assert response.status_code == 500
    assert "Database error" in response.json()["detail"]


# ============================================================================
# Test: Missing File Parameter
# ============================================================================

def test_recognize_endpoint_missing_file(client):
    """Test that missing file parameter returns 422."""
    # Act
    response = client.post("/recognize")

    # Assert
    assert response.status_code == 422  # Validation error