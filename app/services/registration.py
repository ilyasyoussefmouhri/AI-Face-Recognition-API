from fastapi import UploadFile, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
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
from app.schemas.register_schema import RegisterResponse


def register_user(
        file: UploadFile,
        name: str,
        surname: str,
        db: Session,
        embedder: InsightFaceEmbedder,
        auth_user_id,
) -> RegisterResponse:
    """
    Register a new user with their face embedding.

    Returns:
        RegisterResponse: Registration response

    Raises:
        HTTPException: On validation, processing, or database errors
    """
    try:
        # Step 1: Validate the file
        validate_image(file.file)

        # Step 2: Decode image into PIL image
        image_pil = decode_image(file.file)

        # Step 3: Load image into numpy array
        img_array = load_image(image_pil)

        # Step 4: Extract embedding
        embedding_obj = embedder.embed(img_array)

        # Check if user with same auth_user_id already has biometric data
        existing = db.query(User).filter(
            User.auth_user_id == auth_user_id
        ).first()

        if existing:
            logger.error(f"User {auth_user_id} already has biometric data")
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Biometric data already registered. Delete existing profile first."
            )

        # Step 5: Save to database
        logger.info(f"Registering user {name} {surname}...")

        user = User(name=name, surname=surname, auth_user_id=auth_user_id)
        logger.info(f"User object created")

        face = Face(
            user_id=user.user_id,
            embedding=embedding_obj.embedding.tolist(),
            detection_score=embedding_obj.detection_score
        )
        logger.info(f"Face object created")

        user.faces.append(face)
        db.add(user)
        db.flush()  # Persist to DB and generate IDs

        response = RegisterResponse(user_id=user.user_id, is_registered=True)
        logger.info(f"User {user.name} {user.surname} registered successfully with ID {user.user_id}")

        return response
    except HTTPException:
        raise  # Re-raise HTTP exceptions to be handled by FastAPI
    except (ImageProcessingError, NoFaceDetectedError, MultipleFacesDetectedError) as e:
        logger.error(f"Registration failed: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))

    except SQLAlchemyError as e:
        logger.error(f"Database error during registration: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Database error occurred")

    except Exception as e:
        logger.error(f"Unexpected error during registration: {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Registration failed")