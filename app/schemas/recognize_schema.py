from pydantic import BaseModel, Field
from uuid import UUID
from typing import Optional


class RecognizeResponse(BaseModel):
    match: bool = Field(..., description="Whether a match was found")
    user_id: Optional[UUID] = Field(None, description="Matched user ID (None if no match)")
    similarity: float = Field(0.0, ge=0.0, le=1.0, description="Similarity score")

    class Config:
        from_attributes = True