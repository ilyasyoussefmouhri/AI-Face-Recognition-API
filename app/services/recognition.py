import os
import time

from fastapi import UploadFile, HTTPException, Request
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import text
from app.db.models import User, Face
from app.services.preprocessing import decode_image, load_image
from app.services.validation import validate_image
from app.models.insightface import InsightFaceEmbedder
from app.utils.exceptions import (
    NoFaceDetectedError,
    MultipleFacesDetectedError,
    ImageProcessingError,
)
from app.core.logs import logger
from app.schemas.recognize_schema import RecognizeResponse
from app.models.matcher import InsightFaceMatcher

BENCHMARK_MODE: bool = os.getenv("BENCHMARK_MODE", "false").lower() == "true"


def recognize_user(
        file: UploadFile,
        embedder: InsightFaceEmbedder,
        matcher: InsightFaceMatcher,
        db: Session,
        request: Request | None = None,
) -> RecognizeResponse:

    try:
        validate_image(file.file)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    try:
        image_pil = decode_image(file.file)
        img_array = load_image(image_pil)
        embedding_obj = embedder.embed(img_array)
        query_embedding = embedding_obj.embedding  # numpy array, L2-normalised

        logger.info("Recognizing user via pgvector HNSW index...")

        # Format the vector as a literal string pgvector understands.
        # This is required — passing via subquery prevents index usage.
        vec_literal = "[" + ",".join(map(str, query_embedding.tolist())) + "]"

        _t0 = time.perf_counter()
        result = db.execute(
            text("""
                 SELECT f.face_id,
                        f.user_id,
                        1 - (f.embedding <=> CAST(:vec AS vector)) AS similarity
                 FROM faces f
                          JOIN users u ON f.user_id = u.user_id
                 ORDER BY f.embedding <=> CAST(:vec AS vector)
                LIMIT 1
                 """),
            {"vec": vec_literal}
        ).fetchone()
        _t1 = time.perf_counter()

        if BENCHMARK_MODE and request is not None:
            request.state.db_time_ms = (_t1 - _t0) * 1000
            request.state.similarity_time_ms = 0.0

        if result is None:
            logger.warning("No faces in database")
            return RecognizeResponse(user_id=None, similarity=0.0, match=False)

        face_id, user_id, similarity = result
        similarity = float(similarity)

        if similarity >= matcher.threshold:
            logger.info(f"User recognized: user_id={user_id} similarity={similarity:.3f}")
            return RecognizeResponse(
                match=True,
                user_id=user_id,
                similarity=min(1.0, max(-1.0, similarity)),
            )
        else:
            logger.info(f"No match found (best similarity: {similarity:.3f})")
            return RecognizeResponse(
                user_id=None,
                similarity=min(1.0, max(-1.0, similarity)),
                match=False,
            )

    except (ImageProcessingError, NoFaceDetectedError, MultipleFacesDetectedError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    except SQLAlchemyError as e:
        logger.error(f"Database error during recognition: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Database error occurred")
    except Exception as e:
        logger.error(f"Unexpected error: {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Recognition failed")