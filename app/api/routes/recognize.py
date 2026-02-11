from fastapi import APIRouter, UploadFile
from fastapi.params import Depends
from sqlalchemy.orm import Session
from app.api.deps import get_db, valid_content_length, get_embedder,get_matcher
from app.services.recognition import recognize_user
from app.models.insightface import InsightFaceEmbedder
from app.models.matcher import InsightFaceMatcher

router = APIRouter()

@router.post("/recognize", tags=["recognize"])
async def recognize(
        file: UploadFile = Depends(valid_content_length),
        db : Session = Depends(get_db),
        embedder: InsightFaceEmbedder = Depends(get_embedder),
        matcher: InsightFaceMatcher = Depends(get_matcher),
        ):


    return recognize_user(file=file,db=db,embedder=embedder,matcher=matcher)
