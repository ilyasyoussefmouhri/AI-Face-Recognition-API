# SQLAlchemy tables
from sqlalchemy import Column, String, Float, ARRAY, ForeignKey, Boolean, DateTime
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, relationship
import uuid
from app.db.base import Base
from pgvector.sqlalchemy import Vector

# Table for storing face embeddings linked to users
class Face(Base):
    __tablename__ = "faces"

    face_id = Column(UUID(as_uuid=True), primary_key=True, default=lambda: uuid.uuid4())
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    embedding = Column(Vector(512), nullable=False) # changed from ARRAY(Float) for pgvector support
    detection_score = Column(Float, nullable=True) # Detection confidence score

    user: Mapped["User"] = relationship("User", back_populates="faces")



# Biometric identity table linked to authentication credentials
class User(Base):
    __tablename__ = "users"

    user_id = Column(UUID(as_uuid=True),primary_key=True,default=uuid.uuid4)

    name = Column(String,nullable=False)

    surname = Column(String,nullable=False)

    auth_user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("auth_users.auth_user_id", ondelete="CASCADE"),
        unique=True,
        nullable=False
    )

    auth_user = relationship("AuthUser",back_populates="person")

    faces = relationship("Face",back_populates="user",cascade="all, delete-orphan")





# Authentication table for user credentials
class AuthUser(Base):
    __tablename__ = "auth_users"

    auth_user_id = Column(UUID(as_uuid=True),primary_key=True,default=uuid.uuid4)

    username = Column(String,unique=True,index=True,nullable=False)

    password_hash = Column(String,nullable=False)

    is_active = Column(Boolean,nullable=False,default=True)

    is_admin = Column(Boolean,nullable=False,default=False)

    created_at = Column(DateTime(timezone=True),server_default=func.now(),nullable=False)

    # One-to-one relationship with User (biometric identity)
    person = relationship(
        "User",
        back_populates="auth_user",
        uselist=False,
        cascade="all, delete-orphan"
    )