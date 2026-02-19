from fastapi import FastAPI
from app.api.routes import register, recognize, health, auth, delete
from app.core.limiter import limiter
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI(
    title="AI Face Recognition API",
    version="1.0",
)

app.include_router(health.router)
app.include_router(recognize.router)
app.include_router(register.router)
app.include_router(delete.router, tags=["delete"])

app.include_router(auth.router, tags=["auth"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
CORSMiddleware,
allow_origins=[""],
allow_credentials=True,
allow_methods=[""],
allow_headers=["*"],
)

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