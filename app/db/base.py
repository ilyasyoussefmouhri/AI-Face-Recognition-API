from sqlalchemy import create_engine
from app.core.config import Settings


# Load DB URL from env variable
settings = Settings()
engine = create_engine(
        str(settings.DATABASE_URL),
        pool_pre_ping=True
        )


