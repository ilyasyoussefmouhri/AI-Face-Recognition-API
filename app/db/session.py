from sqlalchemy.orm import sessionmaker
from app.db.base import engine
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
