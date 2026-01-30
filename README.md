# Face Recognition API (Production-Oriented, API-First)

> **Status:** In active development  
> **Goal:** Build a production-grade facial recognition REST API using pretrained models, PostgreSQL, and modern Python backend practices.  
> **Target audience:** ML / Backend internship recruiters, engineers, and technical reviewers.

---

## 🚀 Project Motivation

This project is **not a demo or tutorial exercise**.

It is intentionally designed to:
- Mirror **real-world ML backend systems**
- Use **industry-relevant technologies**
- Demonstrate **engineering judgment and trade-offs**
- Prioritize **system design over model training**

The goal is to showcase my ability to build, reason about, and evolve an applied computer vision system under realistic constraints.

---

## 🧠 Core Problem

Build a REST API that can:
1. **Register** a person by extracting and storing a facial embedding from an image
2. **Recognize** a person by matching an input image against stored embeddings
3. Be extensible to **real-time (webcam) recognition**
4. Scale beyond naive in-memory matching

All while using **pretrained models only** (no fine-tuning).

---

## 🏗️ High-Level Architecture

```
Client (HTTP / Webcam)
        |
        v
-------------------------
 FastAPI Application
-------------------------
 |  API Layer
 |   - /register
 |   - /recognize
 |
 |  Services Layer
 |   - Image preprocessing
 |   - Registration logic
 |   - Recognition logic
 |
 |  ML Layer (Pretrained)
 |   - Face Detection
 |   - Face Embeddings
 |
 |  Persistence Layer
 |   - PostgreSQL
 |   - SQLAlchemy (async)
 |
-------------------------
```

---

## 🔧 Technology Choices & Rationale

### FastAPI
**Why:**
- Async-first, high performance
- Automatic OpenAPI / Swagger docs
- Strong ecosystem for production APIs

---

### PostgreSQL
**Why:**
- Industry-standard relational database
- Used by Supabase and production backends
- Supports schema evolution and indexing
- Enables future vector search (`pgvector`)

---

### SQLAlchemy (Async) + Alembic
**Why:**
- Explicit data modeling
- Schema migrations
- Separation of persistence and business logic

---

### Face Detection (MTCNN / RetinaFace)
**Why:**
- Dedicated face detectors
- Clear separation of detection vs recognition
- Robust failure handling

---

### Face Embeddings (ArcFace / FaceNet via DeepFace)
**Why:**
- Industry-proven pretrained models
- No training required
- Explicit embedding extraction (no black-box verification)

---

### Similarity Matching (Cosine Distance)
**Why:**
- Standard for face embeddings
- Compatible with pgvector
- Transparent thresholding

---

## 📦 Project Structure

```
app/
├── api/
├── services/
├── models/
├── db/
├── schemas/
└── utils/
```

---

## ⚠️ Ethics & Privacy

- Explicit consent required
- No real faces committed to repo
- Embeddings are deletable
- Educational / evaluative use only

---

## 🛣️ Roadmap

- [ ] API skeleton & PostgreSQL
- [ ] Image ingestion & validation
- [ ] Face detection
- [ ] Embeddings & storage
- [ ] Recognition endpoint
- [ ] Engineering hardening
- [ ] Advanced optimization

---

> This README will evolve as the project matures.

---

## Current Status / Progress
- FastAPI application skeleton is set up with modular routing across `api`, `services`, `db`, and `core`.
- PostgreSQL 18 is installed, running, and verified via systemd and `psql`.
- Dedicated database and application user are created.
- Database access is configured using SQLAlchemy; connection URL validated for the psycopg driver.
- Environment-based configuration is in place via Pydantic Settings and `.env`.
- Alembic is initialized at the project root (migrations planned, not yet written).
- Face detection/embedding/recognition logic is not implemented yet—API-first scaffolding only.

## Why These Technologies (Concise)
- **FastAPI**: Clear dependency injection and automatic OpenAPI docs; good for production REST services.
- **PostgreSQL 18**: Stable relational store with future headroom for vector search extensions.
- **SQLAlchemy**: Keeps persistence concerns separate and explicit; portable, well-supported with Postgres.
- **Alembic**: Versioned schema migrations to evolve safely; kept at repo root to avoid app coupling.
- **Pydantic Settings + .env**: Environment-driven config to keep secrets out of code and support per-environment overrides.

## What I Learned So Far (≈10 hours)
- Wiring FastAPI routes and dependencies to keep endpoints thin and testable.
- Creating and validating PostgreSQL roles/databases; confirming service health via systemd and `psql`.
- Building and validating SQLAlchemy engine/session configuration for PostgreSQL with psycopg DSNs.
- Using Pydantic Settings to drive configuration from environment variables and `.env`.
- Alembic initialization strategy and migration planning (versioned, root-scoped).
- Separation of concerns: API layer vs. services vs. persistence.

## Next Steps (from roadmap_and_milestones.md)
- Author initial Alembic migrations for the current schema and run them against PostgreSQL.
- Harden health checks and basic integration coverage between API and DB.
- Add robust image ingestion: multipart uploads, validation, preprocessing.
- Integrate a face detector (e.g., MTCNN/RetinaFace) with bounding box extraction and single-face enforcement.
- Implement embedding extraction via pretrained models (ArcFace/FaceNet), normalization, and storage in PostgreSQL.
- Build `/recognize` with threshold logic and clear responses.
- Add logging, exception hierarchy, and pytest coverage for engineering polish.
