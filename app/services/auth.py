from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from app.db.models import AuthUser
from app.core.security import hash_password, verify_password, create_access_token
from app.core.logs import logger
import uuid


def register_auth_user(db: Session, username: str, password: str) -> AuthUser:

    existing_user = db.query(AuthUser).filter(
        AuthUser.username == username
    ).first()

    if existing_user:
        logger.info("Registration attempt with existing username=%s", username)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists"
        )

    logger.debug("Creating new auth user username=%s", username)
    new_user = AuthUser(
        auth_user_id=uuid.uuid4(),
        username=username,
        password_hash=hash_password(password),
        is_active=True,
        is_admin=False
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    logger.info("Registered new user id=%s username=%s", new_user.auth_user_id, username)
    return new_user


def authenticate_user(db: Session, username: str, password: str) -> str:
    user = db.query(AuthUser).filter(
        AuthUser.username == username
    ).first()

    if not user:
        logger.warning("Login failed: unknown username=%s", username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )

    if not verify_password(password, user.password_hash):
        logger.warning("Login failed: wrong password username=%s", username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )

    if not user.is_active:
        logger.info("Inactive user login blocked username=%s", username)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive account"
        )

    token = create_access_token(
        data={
            "sub": str(user.auth_user_id),
            "role": "admin" if user.is_admin else "user"
        }
    )

    logger.info("User authenticated username=%s id=%s", username, user.auth_user_id)
    return token
