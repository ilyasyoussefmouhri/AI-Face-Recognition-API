from fastapi import FastAPI
from app.api.routes import register, recognize, health, auth

app = FastAPI(
    title="AI Face Recognition API",
    version="1.0",
)

app.include_router(health.router)
app.include_router(recognize.router)
app.include_router(register.router)

app.include_router(auth.router, tags=["auth"])

# This function is only for manual testing/setup

def main():
    from app.db.session import engine
    from app.db.base import Base
    from app.db.models import User, Face # Import models to ensure they are registered with Base

    print("Creating tables if they don't exist...")
    Base.metadata.create_all(bind=engine)
    print(f"Tables: {list(Base.metadata.tables.keys())}")


if __name__ == '__main__':
    main()