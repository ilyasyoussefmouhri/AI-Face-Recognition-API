from fastapi import UploadFile
from sqlalchemy.orm import Session
from app.db.models import User, Face
from app.services.preprocessing import decode_image, load_image
from app.services.validation import validate_image
from app.models.insightface import InsightFaceEmbedder
from app.core.config import (
    NoFaceDetectedError,
    MultipleFacesDetectedError,
    ImageProcessingError
)
from app.core.logs import logger


def register_user(
        file: UploadFile,
        name: str,
        db: Session,
        embedder: InsightFaceEmbedder
) -> dict:  # Or a proper Pydantic response model
    """
    Register a new user with their face embedding.

    Returns:
        dict with user_id, face_id, detection_score

    Raises:
        ImageProcessingError: Invalid image
        NoFaceDetectedError: No face found
        MultipleFacesDetectedError: Multiple faces
    """
    try:
        # Step 1: validate the file
        if validate_image(file.file):

            # Step 2: Decode image into PIL image
            image_pil = decode_image(file.file)

            # Step 3: Load image into numpy array:
            img_array = load_image(image_pil)

            # Step 4: Extract embedding
            embedding_obj = embedder.embed(img_array)

            # Step 5: Save to database
            with db.begin():
                user = User(name=name)
                face = Face(
                    user_id=user.user_id,
                    embedding=embedding_obj.embedding.tolist(),
                    detection_score=embedding_obj.detection_score
                )
                user.faces.append(face)
                db.add(user)
                logger.info(f"User {user.name} registered successfully.")

            db.refresh(user)
            db.refresh(face)
            return {
                "user_id": user.user_id,
                "face_id": face.face_id,
                "detection_score": embedding_obj.detection_score
            }


    except (ImageProcessingError, NoFaceDetectedError, MultipleFacesDetectedError) as e:
        logger.error(f"Registration failed: {str(e)}")
        raise e