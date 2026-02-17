from fastapi import APIRouter, UploadFile, Form, Request
from fastapi.params import Depends
from sqlalchemy.orm import Session
from app.api.deps import get_db, valid_content_length, get_embedder, get_current_user
from app.services.registration import register_user
from app.models.insightface import InsightFaceEmbedder
from app.db.models import AuthUser
from app.core.limiter import limiter



router = APIRouter()
@router.post("/register", tags=["register"])
@limiter.limit("5/minute")
async def register(
    request: Request,
    file: UploadFile = Depends(valid_content_length),
    name: str = Form(...),
    surname: str = Form(...),
    db: Session = Depends(get_db),
    embedder: InsightFaceEmbedder = Depends(get_embedder),
    current_user: AuthUser = Depends(get_current_user)
):

    return register_user(file=file,name=name,surname=surname,db=db,embedder=embedder,auth_user_id=current_user.auth_user_id)