from fastapi import APIRouter, Depends
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from app.schemas.auth_schema import (
    AuthRegisterRequest,
    TokenResponse
)
from app.services.auth import (
    register_auth_user,
    authenticate_user
)
from app.api.deps import get_db

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", status_code=201)
def register(
    payload: AuthRegisterRequest,
    db: Session = Depends(get_db)
):
    user = register_auth_user(
        db=db,
        username=payload.username,
        password=payload.password
    )

    return {
        "message": "User registered successfully",
        "username": user.username
    }


@router.post("/login", response_model=TokenResponse)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    access_token = authenticate_user(
        db=db,
        username=form_data.username,
        password=form_data.password
    )
    return TokenResponse(access_token=access_token)