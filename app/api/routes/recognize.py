from fastapi import APIRouter, UploadFile, File

router = APIRouter()

@router.post("/recognize", tags=["recognize"])
def recognize(file: UploadFile = File(...)):
    return {
        "Endpoint": "Recognize",
        "filename": file.filename,
            }