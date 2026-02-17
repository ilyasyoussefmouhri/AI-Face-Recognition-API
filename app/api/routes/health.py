from fastapi import APIRouter, Request
from app.core.limiter import limiter

router = APIRouter()

@router.get("/health", tags=["health"])
@limiter.limit("60/minute")
def health(request: Request):
    return {"status": "ok"}
