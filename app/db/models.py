# SQLAlchemy tables
from sqlalchemy import Column, String, Integer, Float, ARRAY
from sqlalchemy.orm import declarative_base

import uuid


# Base class for ORM models
Base = declarative_base()

class Face(Base):
    __tablename__ = "faces"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    embedding = Column(ARRAY(Float), nullable=False)  # For later pgvector

