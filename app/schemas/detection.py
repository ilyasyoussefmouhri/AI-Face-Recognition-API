from pydantic import BaseModel, Field, ConfigDict
import numpy as np

# Base model for face embeddings output-schema
class FaceEmbedding(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    embedding: np.ndarray = Field(..., description="Face embedding vector")
    detection_score: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")