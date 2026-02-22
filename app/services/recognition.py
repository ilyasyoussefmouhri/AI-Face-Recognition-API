# app/services/recognition.py  (benchmark-aware version)
#
# Differences from the original:
#   1. Imports BENCHMARK_MODE and time.
#   2. Wraps the DB query and similarity loop with perf_counter when
#      BENCHMARK_MODE=true and a Request object is passed in.
#   3. The function signature gains an optional `request` parameter
#      (defaults to None) so existing call sites need NO changes.
#
# When BENCHMARK_MODE=false (the default), the two `if BENCHMARK_MODE`
# branches are never entered and there is zero overhead.
#


import os
import time

from fastapi import UploadFile, HTTPException, Request
import sqlalchemy as sa
from sqlalchemy import func
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

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
    """
    Recognize a user from an uploaded image.

    `request` is optional; when provided and BENCHMARK_MODE=true, timing
    data is written to request.state for the middleware to pick up.
    """
    try:
        validate_image(file.file)
    except ValueError as e:
        logger.error(f"Invalid image format: {e}")
        raise HTTPException(status_code=422, detail=str(e))

    try:
        image_pil = decode_image(file.file)
        img_array = load_image(image_pil)
        embedding_obj = embedder.embed(img_array)
        query_embedding = embedding_obj.embedding

        logger.info("Recognizing user from uploaded image...")

        # ── Single DB query replaces full table scan + Python loop ────────
        # <-> is pgvector's cosine distance operator.
        # Cosine distance = 1 - cosine similarity, so ORDER BY ASC gives
        # the most similar face first.
        # We fetch only 1 row — the closest match.
        _t0 = time.perf_counter()
        result = (
            db.query(Face, func.cast(
                1 - Face.embedding.cosine_distance(query_embedding.tolist()),
                sa.Float
            ).label("similarity"))
            .join(User)
            .order_by(Face.embedding.cosine_distance(query_embedding.tolist()))
            .first()
        )
        _t1 = time.perf_counter()

        if BENCHMARK_MODE and request is not None:
            request.state.db_time_ms = (_t1 - _t0) * 1000

        if result is None:
            logger.warning("No faces in database")
            return RecognizeResponse(user_id=None, similarity=0.0, match=False)



        best_face, similarity = result

        if similarity >= matcher.threshold:
            logger.info(
                f"User recognized: {best_face.user.name} {best_face.user.surname} "
                f"(similarity: {similarity:.3f})"
            )
            return RecognizeResponse(
                match=True,
                user_id=best_face.user_id,
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
        logger.error(f"Detection failed: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))

    except SQLAlchemyError as e:
        logger.error(f"Database error during recognition: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Database error occurred")

    except Exception as e:
        logger.error(
            f"Unexpected error during recognition: {type(e).__name__}: {e}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Recognition failed")
