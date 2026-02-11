from pydantic import BaseModel, Field, ConfigDict
from uuid import UUID
from typing import Optional


class RecognizeResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    match: bool = Field(..., description="Whether a match was found")
    user_id: Optional[UUID] = Field(None, description="Matched user ID (None if no match)")
    similarity: float = Field(0.0, ge=-1.0, le=1.0, description="Similarity score")