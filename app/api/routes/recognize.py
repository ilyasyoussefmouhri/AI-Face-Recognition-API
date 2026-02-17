from fastapi import APIRouter, UploadFile
from fastapi.params import Depends
from sqlalchemy.orm import Session
from app.api.deps import get_db, valid_content_length, get_embedder,get_matcher, get_current_user
from app.services.recognition import recognize_user
from app.models.insightface import InsightFaceEmbedder
from app.models.matcher import InsightFaceMatcher
from app.db.models import AuthUser

router = APIRouter()

@router.post("/recognize", tags=["recognize"])
async def recognize(
        file: UploadFile = Depends(valid_content_length),
        db : Session = Depends(get_db),
        embedder: InsightFaceEmbedder = Depends(get_embedder),
        matcher: InsightFaceMatcher = Depends(get_matcher),
        current_user: AuthUser = Depends(get_current_user)
        ):


    return recognize_user(file=file,db=db,embedder=embedder,matcher=matcher)
