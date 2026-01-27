# Face Recognition API – Project Structure (PostgreSQL + Advanced Stack)

## High-Level Architecture

```
Client (HTTP / Webcam)
        |
        v
-------------------------
 FastAPI Application
-------------------------
 |  API Layer (Routes)
 |   - /register
 |   - /recognize
 |   - /health
 |
 |  Services Layer
 |   - FaceDetectionService
 |   - EmbeddingService
 |   - RecognitionService
 |
 |  ML Models (Pretrained)
 |   - Face Detector (MTCNN / RetinaFace)
 |   - Embedding Model (ArcFace / FaceNet via DeepFace)
 |
 |  Persistence Layer
 |   - PostgreSQL
 |   - SQLAlchemy (async)
 |   - Alembic migrations
 |
 |  Utilities
 |   - Image validation
 |   - Logging
 |   - Error handling
 |
-------------------------
```

## Repository Layout

```
face-recognition-api/
├── app/
│   ├── main.py              # FastAPI app entrypoint
│   ├── api/
│   │   ├── routes/
│   │   │   ├── register.py
│   │   │   ├── recognize.py
│   │   │   └── health.py
│   │   └── deps.py          # Dependencies 
│   │
│   ├── core/
│   │   ├── config.py        # env, settings
│   │   ├── logging.py
│   │   └── security.py      # future auth
│   │
│   ├── models/
│   │   ├── detector.py      # MTCNN / RetinaFace abstraction
│   │   ├── embedder.py      # ArcFace / FaceNet
│   │   └── matcher.py       # cosine similarity, thresholds
│   │
│   ├── services/
│   │   ├── registration.py
│   │   ├── recognition.py
│   │   └── preprocessing.py
│   │
│   ├── db/
│   │   ├── base.py
│   │   ├── session.py
│   │   ├── models.py        # SQLAlchemy tables
│   │   └── migrations/
│   │
│   ├── schemas/
│   │   ├── requests.py
│   │   └── responses.py
│   │
│   └── utils/
│       ├── image_io.py
│       ├── validation.py
│       └── exceptions.py
│
├── tests/
├── scripts/
│   └── local_test_client.py
│
├── requirements.txt
├── README.md
└── docker-compose.yml
```

## Design Philosophy

- API-first, production-oriented
- Explicit ML steps (no black-box calls)
- PostgreSQL as core persistence
- Easy future upgrade to pgvector
- Clear separation of concerns
