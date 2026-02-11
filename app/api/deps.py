# FastAPI dependency injection
from app.db.session import SessionLocal
from fastapi import UploadFile, HTTPException, status
from app.models.insightface import InsightFaceEmbedder
from app.models.matcher import InsightFaceMatcher
from app.core.config import Device, settings
from app.core.logs import logger


# Loads embedder model once at startup
_embedder_instance = None

def get_embedder() -> InsightFaceEmbedder:
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = InsightFaceEmbedder(
            model_name='buffalo_l',
            device=Device.CPU
        )
    return _embedder_instance

# Loads matcher once at startup
_matcher_instance = None

def get_matcher() -> InsightFaceMatcher:
    global _matcher_instance
    if _matcher_instance is None:
        _matcher_instance = InsightFaceMatcher(
            threshold=0.7  # Adjust this value based on your needs
        )
    return _matcher_instance


# Database session dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Database error: {e}")
        raise e
    finally:
        db.close()

#  App level size limiting dependency
async def valid_content_length(file: UploadFile):
    # Read the file in chunks to count actual bytes
    real_size = 0
    for chunk in iter(lambda: file.file.read(1024 * 64), b''):
        real_size += len(chunk)
        if real_size > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="File too large"
            )

    # Reset the cursor so the next function can read it
    file.file.seek(0)
    return file