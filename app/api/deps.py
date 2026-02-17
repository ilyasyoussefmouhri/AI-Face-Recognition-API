# FastAPI dependency injection
from app.db.session import SessionLocal
from fastapi import UploadFile, HTTPException, status, Depends
from app.models.insightface import InsightFaceEmbedder
from app.models.matcher import InsightFaceMatcher
from app.core.config import Device, settings
from app.core.logs import logger
from fastapi.security import OAuth2PasswordBearer
from app.core.security import decode_access_token
from app.db.models import AuthUser
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from jose import JWTError




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
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Database error: {e}")
        raise
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


# Authentication dependency
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> AuthUser:

    try:
        payload = decode_access_token(token)
        user_id = payload.get("sub")

        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )

    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )

    user = db.query(AuthUser).filter(
        AuthUser.auth_user_id == user_id
    ).first()

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive account"
        )

    return user


# Admin check dependency
def get_current_admin(
    current_user: AuthUser = Depends(get_current_user)
) -> AuthUser:

    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )

    return current_user


# Person profile check dependency
def get_current_person(
    current_user: AuthUser = Depends(get_current_user)
):
    if not current_user.person:
        raise HTTPException(404, "Person profile not created")
    return current_user.person


