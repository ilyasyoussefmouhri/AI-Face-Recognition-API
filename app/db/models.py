# SQLAlchemy tables
from sqlalchemy import Column, String, Float, ARRAY, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, relationship
import uuid
from app.db.base import Base


class Face(Base):
    __tablename__ = "faces"

    face_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=False)
    embedding = Column(ARRAY(Float), nullable=False) # Embedding stored as float array, will later change to Vector(512)

    user: Mapped["User"] = relationship("User", back_populates="faces")


class User(Base):
    __tablename__ = "users"

    user_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)

    faces: Mapped[list["Face"]] = relationship(
        "Face",
        back_populates="user",
        cascade="all, delete-orphan",
    )
