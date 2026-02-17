from jose import jwt
from datetime import datetime, timedelta, timezone
from jose import JWTError
import bcrypt
from app.core.config import settings
from app.core.logs import logger
from app.utils.exceptions import CredentialsError


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    try:
        logger.debug("Hashing password")
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    except Exception as e:
        logger.error("Error hashing password: %s", e)
        raise


def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        logger.debug("Verifying password hash")
        return bcrypt.checkpw(plain_password.encode(), hashed_password.encode())
    except Exception as e:
        logger.error("Error verifying password hash: %s", e)
        raise


def create_access_token(data: dict) -> str:
    try:
        to_encode = data.copy()
        now = datetime.now(timezone.utc)
        expire = now + timedelta(
            seconds=settings.ACCESS_TOKEN_EXPIRE_SECONDS
        )
        to_encode.update({
            "exp": expire,
            "iat": now
        })

        encoded_jwt = jwt.encode(
            to_encode,
            settings.SECRET_KEY,
            algorithm=settings.ALGORITHM
        )

        logger.info("Access token created for subject=%s", to_encode.get("sub"))

        return encoded_jwt
    except Exception as e:
        logger.error("Error creating access token: %s", e)
        raise


def decode_access_token(token: str) -> dict:
    try:
        logger.debug("Decoding access token")
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        return payload
    except JWTError:
        logger.error("Invalid access token provided")
        raise CredentialsError()