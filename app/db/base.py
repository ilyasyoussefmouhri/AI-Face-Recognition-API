
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    pass

# Import all models here for Alembic
from app.db.models import User
from app.db.models import Face
