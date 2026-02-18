from uuid import UUID
from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from app.db.models import AuthUser, User
from app.core.logs import logger





def delete_me(
        delete_auth_user: bool,
        current_user: AuthUser,
        db: Session
):
    """
    Deletes the authenticated user's biometric data.
    If delete_auth_user=true, also deletes the auth account (cascade handles biometric data).
    """

    if delete_auth_user:
        # Deleting auth user cascades to biometric data
        logger.info(f"Deleting auth user {current_user.auth_user_id} (cascade will delete biometric)")
        db.delete(current_user)
    else:
        # Only delete biometric data, keep auth account
        biometric_user = db.query(User).filter(
            User.auth_user_id == current_user.auth_user_id
        ).first()

        if biometric_user:
            logger.info(f"Deleting biometric profile for user {current_user.auth_user_id}")
            db.delete(biometric_user)
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No biometric data found"
            )

    db.commit()
    return

def delete_user(
        user_id: UUID,
        delete_auth_user: bool,
        db: Session
):
    """
    Deletes a user and their biometric data.
    """

    logger.debug(f"Querying user {user_id}")
    user = db.query(User).filter(User.user_id == user_id).first()

    if not user:
        logger.error(f"User {user_id} not found")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    logger.info(f"Deleting biometric user {user_id}")
    db.delete(user)  # Deletes biometric User record

    if delete_auth_user:
        # Also delete the AuthUser
        auth_user = db.query(AuthUser).filter(
            AuthUser.auth_user_id == user.auth_user_id
        ).first()

        if auth_user:
            logger.info(f"Also deleting auth user {auth_user.auth_user_id}")
            db.delete(auth_user)

    db.commit()
    return