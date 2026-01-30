from fastapi import APIRouter, UploadFile, File
from fastapi.params import Depends
from sqlalchemy.orm import Session
from app.api.deps import get_db
router = APIRouter()

@router.post("/recognize", tags=["recognize"])
def recognize(
        file: UploadFile = File(...),
        db : Session = Depends(get_db)
        ):


    return {
        "Endpoint": "Recognize",
        "filename": file.filename,
            }
