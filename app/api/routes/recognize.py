from fastapi import APIRouter, UploadFile, File
from fastapi.params import Depends
from sqlalchemy.orm import Session
from app.api.deps import get_db, valid_content_length
router = APIRouter()

@router.post("/recognize", tags=["recognize"])
def recognize(
        file: UploadFile = Depends(valid_content_length),
        db : Session = Depends(get_db)
        ):


    return {
        "Endpoint": "Recognize",
        "filename": file.filename,
            }
