from fastapi import APIRouter, UploadFile, File
from fastapi.params import Depends
from sqlalchemy.orm import Session
from app.api.deps import get_db

router = APIRouter()

@router.post("/register", tags=["register"])
def register(
        file: UploadFile = File(...),
        name: str = File(...),
        db : Session = Depends(get_db)
    ):


    return {
        "Endpoint": "Register",
        "filename": file.filename,
        }
