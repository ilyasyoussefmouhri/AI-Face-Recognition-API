from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.api.deps import get_current_user,get_current_admin, get_db
from app.db.models import AuthUser
from uuid import UUID
from app.services.deletion import delete_user, delete_me




router = APIRouter()


@router.delete("/me", status_code=204)
def delete_me(
        delete_auth_user: bool = False,  # ← defaults to false
        current_user: AuthUser = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """
    Delete authenticated user's biometric data.
    If delete_auth_user=true, deletes auth account (cascade handles biometric data).
    """
    delete_me(delete_auth_user=delete_auth_user, current_user=current_user, db=db)

    return


@router.delete("/{user_id}", status_code=204)
def delete_user(
        user_id: UUID,
        delete_auth_user: bool = False,
        db: Session = Depends(get_db),
        admin: AuthUser = Depends(get_current_admin)
):
    """
    Admin deletes biometric data for a user.
    If delete_auth_user=true, also deletes the auth account.
    """
    delete_user(user_id=user_id, delete_auth_user=delete_auth_user,db=db)
    return