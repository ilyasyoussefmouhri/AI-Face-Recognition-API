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
    from app.db.session import engine
    from app.db.base import Base

    Base.metadata.create_all(bind=engine)
    print("Tables created successfully!")
    from app.db.session import SessionLocal
    from app.db.models import User, Face
    with SessionLocal() as session:
        user = User(name="Dummy User")

        face = Face(
            embeddings=[0.1, 0.2, 0.3],
        )
        # Explicitly link from the owning side (User holds the FK).
        user.face = face

        session.add(user)
        session.add(face)
        session.commit()
        session.refresh(user)
        session.refresh(face)
        print(f"user_id={user.user_id}, face_id={face.face_id}")

if __name__ == '__main__':
    main()