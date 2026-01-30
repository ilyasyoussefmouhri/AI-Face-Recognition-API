# SQLAlchemy tables
from sqlalchemy import Column, String, Float, ARRAY, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, relationship
from app.models.embedder import normalize_embedding
import uuid
from app.db.base import Base


class Face(Base):
    __tablename__ = "faces"

    face_id = Column(UUID(as_uuid=True), primary_key=True, default=lambda: uuid.uuid4())
    embeddings = Column(ARRAY(Float), nullable=False)

    # Relationship back to User
    user: Mapped["User"] = relationship("User", back_populates="face", uselist=False)

    @classmethod
    def normalize_embeddings(cls, embeddings):
        return normalize_embedding(embeddings)


class User(Base):
    __tablename__ = "users"

    user_id = Column(UUID(as_uuid=True), primary_key=True, default=lambda: uuid.uuid4())
    name = Column(String, nullable=False)
    face_id = Column(UUID(as_uuid=True), ForeignKey("faces.face_id"), nullable=False, unique=True)

    # Relationship to Face (one-to-one, User owns the FK)
    face: Mapped["Face"] = relationship("Face", back_populates="user")