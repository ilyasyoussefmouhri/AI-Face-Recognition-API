from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.api.deps import get_current_user,get_current_admin, get_db
from app.db.models import AuthUser, User
from uuid import UUID




router = APIRouter()

@router.delete("/me", status_code=204)
def delete_me(
    current_user: AuthUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Delete authenticated user account and associated biometric data.
    """

    db.delete(current_user)
    db.commit()

    return


@router.delete("/{user_id}", status_code=204)
def delete_user(
    user_id: UUID,
    db: Session = Depends(get_db),
    admin: AuthUser = Depends(get_current_admin)
):
    """
    Admin deletes any biometric user.
    """

    user = db.query(User).filter(
        User.user_id == user_id
    ).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    db.delete(user)
    db.commit()
    return