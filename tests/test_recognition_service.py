"""
Unit tests for the recognition service layer.
Tests core recognition logic in isolation with mocked dependencies.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from uuid import uuid4
from sqlalchemy.orm import Session

from app.services.recognition import recognize_user
from app.schemas.recognize_schema import RecognizeResponse
from app.core.config import (
    NoFaceDetectedError,
    MultipleFacesDetectedError,
    ImageProcessingError
)
from fastapi import HTTPException


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_db_session():
    """Mock database session."""
    return MagicMock(spec=Session)


@pytest.fixture
def mock_upload_file():
    """Mock UploadFile with file attribute."""
    mock_file = MagicMock()
    mock_file.file = MagicMock()
    mock_file.filename = "test_image.jpg"
    return mock_file


@pytest.fixture
def sample_embedding():
    """Generate a normalized 512-dim embedding."""
    emb = np.random.randn(512).astype(np.float32)
    emb = emb / np.linalg.norm(emb)  # L2 normalize
    return emb


@pytest.fixture
def mock_face_in_db(sample_embedding):
    """Mock Face object from database."""
    face = Mock()
    face.face_id = uuid4()
    face.user_id = uuid4()
    face.embedding = sample_embedding.tolist()

    # Mock the user relationship
    user = Mock()
    user.user_id = face.user_id
    user.name = "John"
    user.surname = "Doe"
    face.user = user

    return face


# ============================================================================
# Test: Successful Recognition
# ============================================================================

@patch('app.services.recognition.validate_image')
@patch('app.services.recognition.decode_image')
@patch('app.services.recognition.load_image')
def test_recognize_user_success(
        mock_load_image,
        mock_decode_image,
        mock_validate_image,
        mock_upload_file,
        mock_db_session,
        mock_embedder,
        sample_embedding,
        mock_face_in_db,
        single_face_image
):
    """Test successful user recognition with high similarity."""
    # Arrange
    from app.models.matcher import InsightFaceMatcher

    mock_validate_image.return_value = True
    mock_decode_image.return_value = Mock()  # PIL Image
    mock_load_image.return_value = single_face_image

    # Mock embedder to return same embedding as stored in DB
    mock_embedding_obj = Mock()
    mock_embedding_obj.embedding = sample_embedding
    mock_embedder.embed = Mock(return_value=mock_embedding_obj)

    # Mock database query
    mock_db_session.query.return_value.join.return_value.all.return_value = [mock_face_in_db]

    matcher = InsightFaceMatcher(threshold=0.7)

    # Act
    result = recognize_user(
        file=mock_upload_file,
        embedder=mock_embedder,
        matcher=matcher,
        db=mock_db_session
    )

    # Assert
    assert isinstance(result, RecognizeResponse)
    assert result.match is True
    assert result.user_id == mock_face_in_db.user_id
    assert result.similarity >= 0.7  # Should be ~1.0 since embeddings are identical
    assert result.similarity <= 1.0


# ============================================================================
# Test: No Match Found (Low Similarity)
# ============================================================================

@patch('app.services.recognition.validate_image')
@patch('app.services.recognition.decode_image')
@patch('app.services.recognition.load_image')
def test_recognize_user_no_match_low_similarity(
        mock_load_image,
        mock_decode_image,
        mock_validate_image,
        mock_upload_file,
        mock_db_session,
        mock_embedder,
        sample_embedding,
        mock_face_in_db,
        single_face_image
):
    """Test recognition fails when similarity is below threshold."""
    # Arrange
    from app.models.matcher import InsightFaceMatcher

    mock_validate_image.return_value = True
    mock_decode_image.return_value = Mock()
    mock_load_image.return_value = single_face_image

    # Create a different embedding (low similarity)
    different_embedding = np.random.randn(512).astype(np.float32)
    different_embedding = different_embedding / np.linalg.norm(different_embedding)

    mock_embedding_obj = Mock()
    mock_embedding_obj.embedding = different_embedding
    mock_embedder.embed = Mock(return_value=mock_embedding_obj)

    mock_db_session.query.return_value.join.return_value.all.return_value = [mock_face_in_db]

    matcher = InsightFaceMatcher(threshold=0.7)

    # Act
    result = recognize_user(
        file=mock_upload_file,
        embedder=mock_embedder,
        matcher=matcher,
        db=mock_db_session
    )

    # Assert
    assert isinstance(result, RecognizeResponse)
    assert result.match is False
    assert result.user_id is None
    assert result.similarity < 0.7


# ============================================================================
# Test: Empty Database
# ============================================================================

@patch('app.services.recognition.validate_image')
@patch('app.services.recognition.decode_image')
@patch('app.services.recognition.load_image')
def test_recognize_user_empty_database(
        mock_load_image,
        mock_decode_image,
        mock_validate_image,
        mock_upload_file,
        mock_db_session,
        mock_embedder,
        sample_embedding,
        single_face_image
):
    """Test recognition when database has no registered faces."""
    # Arrange
    from app.models.matcher import InsightFaceMatcher

    mock_validate_image.return_value = True
    mock_decode_image.return_value = Mock()
    mock_load_image.return_value = single_face_image

    mock_embedding_obj = Mock()
    mock_embedding_obj.embedding = sample_embedding
    mock_embedder.embed = Mock(return_value=mock_embedding_obj)

    # Empty database
    mock_db_session.query.return_value.join.return_value.all.return_value = []

    matcher = InsightFaceMatcher(threshold=0.7)

    # Act
    result = recognize_user(
        file=mock_upload_file,
        embedder=mock_embedder,
        matcher=matcher,
        db=mock_db_session
    )

    # Assert
    assert isinstance(result, RecognizeResponse)
    assert result.match is False
    assert result.user_id is None
    assert result.similarity == 0.0


# ============================================================================
# Test: Multiple Faces in Database (Best Match Selection)
# ============================================================================

@patch('app.services.recognition.validate_image')
@patch('app.services.recognition.decode_image')
@patch('app.services.recognition.load_image')
def test_recognize_user_multiple_candidates_best_match(
        mock_load_image,
        mock_decode_image,
        mock_validate_image,
        mock_upload_file,
        mock_db_session,
        mock_embedder,
        single_face_image
):
    """Test that recognition selects the best match from multiple candidates."""
    # Arrange
    from app.models.matcher import InsightFaceMatcher

    mock_validate_image.return_value = True
    mock_decode_image.return_value = Mock()
    mock_load_image.return_value = single_face_image

    # Query embedding
    query_emb = np.random.randn(512).astype(np.float32)
    query_emb = query_emb / np.linalg.norm(query_emb)

    # Create three faces with varying similarity
    face1 = Mock()
    face1.user_id = uuid4()
    face1.embedding = (query_emb * 0.6).tolist()  # Low similarity
    face1.user = Mock(name="Alice", surname="Smith")

    face2 = Mock()
    face2.user_id = uuid4()
    face2.embedding = query_emb.tolist()  # Perfect match
    face2.user = Mock(name="Bob", surname="Jones")

    face3 = Mock()
    face3.user_id = uuid4()
    face3.embedding = (query_emb * 0.8).tolist()  # Medium similarity
    face3.user = Mock(name="Charlie", surname="Brown")

    mock_embedding_obj = Mock()
    mock_embedding_obj.embedding = query_emb
    mock_embedder.embed = Mock(return_value=mock_embedding_obj)

    mock_db_session.query.return_value.join.return_value.all.return_value = [face1, face2, face3]

    matcher = InsightFaceMatcher(threshold=0.7)

    # Act
    result = recognize_user(
        file=mock_upload_file,
        embedder=mock_embedder,
        matcher=matcher,
        db=mock_db_session
    )

    # Assert
    assert result.match is True
    assert result.user_id == face2.user_id  # Should match Bob (perfect match)
    assert result.similarity > 0.9


# ============================================================================
# Test: No Face Detected
# ============================================================================

@patch('app.services.recognition.validate_image')
@patch('app.services.recognition.decode_image')
@patch('app.services.recognition.load_image')
def test_recognize_user_no_face_detected(
        mock_load_image,
        mock_decode_image,
        mock_validate_image,
        mock_upload_file,
        mock_db_session,
        mock_embedder,
        no_face_image
):
    """Test that NoFaceDetectedError raises HTTPException."""
    # Arrange
    from app.models.matcher import InsightFaceMatcher

    mock_validate_image.return_value = True
    mock_decode_image.return_value = Mock()
    mock_load_image.return_value = no_face_image

    # Embedder raises NoFaceDetectedError
    mock_embedder.embed = Mock(side_effect=NoFaceDetectedError("No face detected"))

    matcher = InsightFaceMatcher(threshold=0.7)

    # Act & Assert
    with pytest.raises(HTTPException) as exc_info:
        recognize_user(
            file=mock_upload_file,
            embedder=mock_embedder,
            matcher=matcher,
            db=mock_db_session
        )

    assert exc_info.value.status_code == 422
    assert "No face detected" in str(exc_info.value.detail)


# ============================================================================
# Test: Multiple Faces Detected
# ============================================================================

@patch('app.services.recognition.validate_image')
@patch('app.services.recognition.decode_image')
@patch('app.services.recognition.load_image')
def test_recognize_user_multiple_faces_detected(
        mock_load_image,
        mock_decode_image,
        mock_validate_image,
        mock_upload_file,
        mock_db_session,
        mock_embedder,
        multi_face_image
):
    """Test that MultipleFacesDetectedError raises HTTPException."""
    # Arrange
    from app.models.matcher import InsightFaceMatcher

    mock_validate_image.return_value = True
    mock_decode_image.return_value = Mock()
    mock_load_image.return_value = multi_face_image

    # Embedder raises MultipleFacesDetectedError
    mock_embedder.embed = Mock(side_effect=MultipleFacesDetectedError(num_faces=2))

    matcher = InsightFaceMatcher(threshold=0.7)

    # Act & Assert
    with pytest.raises(HTTPException) as exc_info:
        recognize_user(
            file=mock_upload_file,
            embedder=mock_embedder,
            matcher=matcher,
            db=mock_db_session
        )

    assert exc_info.value.status_code == 422
    assert "2 faces" in str(exc_info.value.detail)


# ============================================================================
# Test: Invalid Image Format
# ============================================================================
# Around line 375
@patch('app.services.recognition.validate_image')
def test_recognize_user_invalid_image_format(
        mock_validate_image,
        mock_upload_file,
        mock_db_session,
        mock_embedder
):
    """Test that invalid image format raises HTTPException 422."""
    from app.models.matcher import InsightFaceMatcher

    mock_validate_image.side_effect = ValueError("Unsupported image format")
    matcher = InsightFaceMatcher(threshold=0.7)

    with pytest.raises(HTTPException) as exc_info:
        recognize_user(
            file=mock_upload_file,
            embedder=mock_embedder,
            matcher=matcher,
            db=mock_db_session
        )

    assert exc_info.value.status_code == 422
    assert "Unsupported image format" in str(exc_info.value.detail)


# ============================================================================
# Test: Image Processing Error
# ============================================================================

@patch('app.services.recognition.validate_image')
@patch('app.services.recognition.decode_image')
def test_recognize_user_image_processing_error(
        mock_decode_image,
        mock_validate_image,
        mock_upload_file,
        mock_db_session,
        mock_embedder
):
    """Test that ImageProcessingError raises HTTPException."""
    # Arrange
    from app.models.matcher import InsightFaceMatcher

    mock_validate_image.return_value = True
    mock_decode_image.side_effect = ImageProcessingError("Failed to decode image")

    matcher = InsightFaceMatcher(threshold=0.7)

    # Act & Assert
    with pytest.raises(HTTPException) as exc_info:
        recognize_user(
            file=mock_upload_file,
            embedder=mock_embedder,
            matcher=matcher,
            db=mock_db_session
        )

    assert exc_info.value.status_code == 422
    assert "Failed to decode image" in str(exc_info.value.detail)


# ============================================================================
# Test: Database Error
# ============================================================================

@patch('app.services.recognition.validate_image')
@patch('app.services.recognition.decode_image')
@patch('app.services.recognition.load_image')
def test_recognize_user_database_error(
        mock_load_image,
        mock_decode_image,
        mock_validate_image,
        mock_upload_file,
        mock_db_session,
        mock_embedder,
        sample_embedding,
        single_face_image
):
    """Test that database errors are handled gracefully."""
    # Arrange
    from app.models.matcher import InsightFaceMatcher
    from sqlalchemy.exc import SQLAlchemyError

    mock_validate_image.return_value = True
    mock_decode_image.return_value = Mock()
    mock_load_image.return_value = single_face_image

    mock_embedding_obj = Mock()
    mock_embedding_obj.embedding = sample_embedding
    mock_embedder.embed = Mock(return_value=mock_embedding_obj)

    # Database raises error
    mock_db_session.query.side_effect = SQLAlchemyError("Database connection failed")

    matcher = InsightFaceMatcher(threshold=0.7)

    # Act & Assert
    with pytest.raises(HTTPException) as exc_info:
        recognize_user(
            file=mock_upload_file,
            embedder=mock_embedder,
            matcher=matcher,
            db=mock_db_session
        )

    assert exc_info.value.status_code == 500
    assert "Database error" in str(exc_info.value.detail)