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
# ── How to wire the request through ─────────────────────────────────────────
# In app/api/routes/recognize.py change:
#
#     return recognize_user(file=file, db=db, embedder=embedder, matcher=matcher)
#
# to:
#
#     return recognize_user(file=file, db=db, embedder=embedder,
#                           matcher=matcher, request=request)
#
# That single-line change is enough; the route already receives `request`.

import os
import time

from fastapi import UploadFile, HTTPException, Request
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
        request: Request | None = None,   # ← NEW optional param
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

        # ── DB fetch ──────────────────────────────────────────────────────
        _t0 = time.perf_counter()
        faces = db.query(Face).join(User).all()
        _t1 = time.perf_counter()

        if BENCHMARK_MODE and request is not None:
            request.state.db_time_ms = (_t1 - _t0) * 1000

        if not faces:
            logger.warning("No faces in database")
            return RecognizeResponse(user_id=None, similarity=0.0, match=False)

        # ── Python similarity loop ────────────────────────────────────────
        _t2 = time.perf_counter()
        best_match = None
        best_similarity = -1.0

        for face in faces:
            stored_embedding = np.array(face.embedding, dtype=np.float32)
            similarity = matcher.similarity(query_embedding, stored_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = face
        _t3 = time.perf_counter()

        if BENCHMARK_MODE and request is not None:
            request.state.similarity_time_ms = (_t3 - _t2) * 1000

        # ── Match decision ────────────────────────────────────────────────
        if best_match and matcher.match(
            query_embedding, np.array(best_match.embedding, dtype=np.float32)
        ):
            logger.info(
                f"User recognized: {best_match.user.name} {best_match.user.surname} "
                f"(similarity: {best_similarity:.3f})"
            )
            return RecognizeResponse(
                match=True,
                user_id=best_match.user_id,
                similarity=min(1.0, max(-1.0, best_similarity)),
            )
        else:
            logger.info(f"No match found (best similarity: {best_similarity:.3f})")
            return RecognizeResponse(
                user_id=None,
                similarity=min(1.0, max(-1.0, best_similarity)),
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
