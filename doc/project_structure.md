# Project Structure (Updated – Matches Repository)

This document reflects the **actual on‑disk project structure** and explains the role of each component.

---

## Top‑Level (on disk)

```
AI Face Recognition/
├── README.md                 # Narrative + technical overview
├── alembic/                  # Database migrations
├── alembic.ini
├── app/                      # Application source
├── coverage.xml              # Test coverage report (XML)
├── data/                     # Sample assets (no real faces committed)
├── doc/                      # Project documentation
├── face_recognition_backend.egg-info/
├── htmlcov/                  # HTML coverage report
├── pyproject.toml
├── pytest.ini
├── reports/                  # Generated analysis / exports
├── requirements.txt
├── scripts/                  # Local utilities
└── tests/                    # Test suite
```

---

## `app/` – Core Application

```
app/
├── api/
│   ├── routes/
│   │   ├── auth.py          # /auth/register + /auth/login (JWT issuance)
│   │   ├── delete.py        # Auth-protected deletion (self + admin)
│   │   ├── health.py        # Health check endpoint
│   │   ├── register.py      # Face registration (auth required, rate limited)
│   │   └── recognize.py     # Face recognition (auth required, rate limited)
│   ├── deps.py              # Dependency injection (DB, ML singletons, auth)
│   └── __init__.py
│
├── core/
│   ├── config.py            # Settings & env config
│   ├── limiter.py           # SlowAPI rate limiter instance
│   ├── logs.py              # Logging setup
│   ├── security.py          # Security utilities (bcrypt hashing, JWT encode/decode)
│   └── logs/
│       └── app.log
│
├── db/
│   ├── base.py              # SQLAlchemy Base
│   ├── models.py            # ORM models
│   ├── session.py           # DB session handling
│   └── __init__.py
│
├── models/                  # ML / CV components
│   ├── insightface.py       # Embedding model wrapper
│   ├── matcher.py           # Embedding similarity logic
│   └── __init__.py
│
├── schemas/                 # Pydantic schemas (request/response contracts)
│   ├── detection.py         # Face detection schema
│   ├── auth_schema.py       # Auth registration + token responses
│   ├── register_schema.py   # Registration response schemas
│   ├── recognize_schema.py  # Recognition response schemas
│   └── __init__.py
│
├── services/                # Business logic
│   ├── auth.py              # Auth user creation + authentication
│   ├── deletion.py          # Cascade-safe deletion flows
│   ├── preprocessing.py     # Image preprocessing
│   ├── registration.py      # Registration pipeline
│   ├── recognition.py       # Recognition pipeline (matcher-backed)
│   ├── validation.py        # Domain validation rules
│   └── __init__.py
│
├── utils/
│   ├── exceptions.py        # Custom exceptions
│   └── __init__.py
│
├── main.py                  # FastAPI entrypoint + router wiring + rate-limit handler
└── __init__.py
```

---

## `tests/` – Test Suite

```
tests/
├── conftest.py
├── test_embedder.py
├── test_recognition_service.py
├── test_recognition_endpoint.py
├── run_tests.sh
└── README.md
```

---

## Design Notes

* Clear separation between **API**, **services**, **ML models**, and **DB**
* Registration and recognition reuse the same detection + embedding stack
* Business logic is isolated from FastAPI routing
* Tests cover both service and API layers

---

**Status:** This structure is accurate as of the latest repository state.
