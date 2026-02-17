from fastapi import APIRouter, UploadFile, Form
from fastapi.params import Depends
from sqlalchemy.orm import Session
from app.api.deps import get_db, valid_content_length, get_embedder
from app.services.registration import register_user
from app.models.insightface import InsightFaceEmbedder



router = APIRouter()
@router.post("/register", tags=["register"])
async def register(
    file: UploadFile = Depends(valid_content_length),
    name: str = Form(...),
    surname: str = Form(...),
    db: Session = Depends(get_db),
    embedder: InsightFaceEmbedder = Depends(get_embedder)
):

    return register_user(file=file,name=name,surname=surname,db=db,embedder=embedder)