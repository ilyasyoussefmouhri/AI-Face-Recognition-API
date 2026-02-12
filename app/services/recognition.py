from fastapi import UploadFile, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
import numpy as np
from app.db.models import User, Face
from app.services.preprocessing import decode_image, load_image
from app.services.validation import validate_image
from app.models.insightface import InsightFaceEmbedder
from app.utils.exceptions import (
    NoFaceDetectedError,
    MultipleFacesDetectedError,
    ImageProcessingError
)
from app.core.logs import logger
from app.schemas.recognize_schema import RecognizeResponse
from app.models.matcher import InsightFaceMatcher


def recognize_user(
        file: UploadFile,
        embedder: InsightFaceEmbedder,
        matcher: InsightFaceMatcher,
        db: Session,
):
    """
    Recognize a user from an uploaded image.

    Returns:
        RecognizeResponse: Recognition response
    """
    try:
        # Step 1: Validate the file
        validate_image(file.file)
    except ValueError as e:
        logger.error(f"Invalid image format: {e}")
        raise HTTPException(status_code=422, detail=str(e))


    try:
        # Step 2: Decode image into PIL image
        image_pil = decode_image(file.file)

        # Step 3: Load image into numpy array
        img_array = load_image(image_pil)

        # Step 4: Extract embedding
        embedding_obj = embedder.embed(img_array)
        query_embedding = embedding_obj.embedding

        # Step 5: Check against database
        logger.info("Recognizing user from uploaded image...")

        # Phase 5: naive 1:N scan, optimized in Phase 6
        faces = db.query(Face).join(User).all()

        if not faces:
            logger.warning("No faces in database")
            return RecognizeResponse(user_id=None, similarity=0.0, match=False)

        best_match = None
        best_similarity = -1.0

        for face in faces:
            # Convert stored embedding back to numpy
            stored_embedding = np.array(face.embedding, dtype=np.float32)

            # Calculate similarity using matcher
            similarity = matcher.similarity(query_embedding, stored_embedding)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = face

        # Use match method to check threshold
        if best_match and matcher.match(query_embedding, np.array(best_match.embedding, dtype=np.float32)):
            logger.info(
                f"User recognized: {best_match.user.name} {best_match.user.surname} (similarity: {best_similarity:.3f})")
            return RecognizeResponse(
                match=True,
                user_id=best_match.user_id,
                similarity=min(1.0, max(-1.0, best_similarity))
            )
        else:
            logger.info(f"No match found (best similarity: {best_similarity:.3f})")
            return RecognizeResponse(user_id=None, similarity=min(1.0, max(-1.0, best_similarity)), match=False)

    except (ImageProcessingError, NoFaceDetectedError, MultipleFacesDetectedError) as e:
        logger.error(f"Detection failed: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))

    except SQLAlchemyError as e:
        logger.error(f"Database error during recognition: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Database error occurred")

    except Exception as e:
        logger.error(f"Unexpected error during recognition: {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Recognition failed")