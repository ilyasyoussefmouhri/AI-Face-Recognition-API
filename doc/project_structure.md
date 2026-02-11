# Project Structure (Updated – Matches Repository)

This document reflects the **actual on‑disk project structure** and explains the role of each component.

---

## Top‑Level

```
AI Face Recognition/
├── .benchmarks/          # Performance benchmarks
├── .venv/                # Local virtual environment
├── alembic/              # Database migrations
├── app/                  # Application source code
├── data/                 # Local data / assets (non‑code)
├── doc/                  # Documentation
├── reports/              # Generated reports
├── scripts/              # Utility / maintenance scripts
├── tests/                # Test suite
├── .env                  # Environment variables
├── alembic.ini
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## `app/` – Core Application

```
app/
├── api/
│   ├── routes/
│   │   ├── health.py        # Health check endpoint
│   │   ├── register.py      # Face registration endpoint
│   │   └── recognize.py     # Face recognition endpoint
│   ├── deps.py              # Dependency injection
│   └── __init__.py
│
├── core/
│   ├── config.py            # Settings & env config
│   ├── logs.py              # Logging setup
│   ├── security.py          # Security helpers (auth, hashing, etc.)
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
│   ├── detector.py          # Face detection abstraction
│   ├── insightface.py       # Embedding model wrapper
│   ├── matcher.py           # Embedding similarity logic
│   └── __init__.py
│
├── schemas/                 # Pydantic schemas
│   ├── detection.py
│   ├── register_schema.py
│   ├── recognize_schema.py
│   └── __init__.py
│
├── services/                # Business logic
│   ├── preprocessing.py     # Image preprocessing
│   ├── registration.py      # Registration pipeline
│   ├── recognition.py       # Recognition pipeline
│   ├── validation.py        # Domain validation rules
│   └── __init__.py
│
├── utils/
│   ├── exceptions.py        # Custom exceptions
│   └── __init__.py
│
├── main.py                  # FastAPI entrypoint
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
