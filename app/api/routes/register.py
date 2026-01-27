from fastapi import APIRouter, UploadFile, File

router = APIRouter()

@router.post("/register", tags=["register"])
def register(file: UploadFile = File(...)):
    return {
        "Endpoint": "Register",
        "filename": file.filename,
        }