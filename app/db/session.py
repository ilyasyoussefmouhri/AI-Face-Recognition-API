from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from app.core.config import settings

# Load DB URL from env variable
engine = create_engine(
        str(settings.DATABASE_URL),
        pool_pre_ping=True
        )
# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


