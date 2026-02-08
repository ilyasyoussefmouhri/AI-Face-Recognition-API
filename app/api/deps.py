# FastAPI dependency injection
from app.db.session import SessionLocal
from fastapi import UploadFile, HTTPException, status
from app.core.security import MAX_FILE_SIZE
from app.models.insightface import InsightFaceEmbedder
from app.core.config import Device

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


# Database session dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

#  App level size limiting dependency
async def valid_content_length(file: UploadFile):
    # Read the file in chunks to count actual bytes
    real_size = 0
    for chunk in iter(lambda: file.file.read(1024 * 64), b''):
        real_size += len(chunk)
        if real_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="File too large"
            )

    # Reset the cursor so the next function can read it
    file.file.seek(0)
    return file