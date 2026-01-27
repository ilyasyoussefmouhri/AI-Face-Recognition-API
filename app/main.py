# FastAPI main entrypoint
from fastapi import FastAPI
from app.api.routes import register, recognize, health

app = FastAPI(
        title="AI Face Recognition API",
        version="1.0",
    )

app.include_router(health.router)
app.include_router(recognize.router)
app.include_router(register.router)
def main():
    from app.db.base import engine, Base
    from app.db import models

    Base.metadata.create_all(bind=engine)
    print("Tables created successfully!")

if __name__ == '__main__':
    main()
