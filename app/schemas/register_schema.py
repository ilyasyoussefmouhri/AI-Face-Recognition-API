# Schema for register
from pydantic import BaseModel
from uuid import UUID


class RegisterResponse(BaseModel):
    user_id: UUID
    is_registered: bool = True