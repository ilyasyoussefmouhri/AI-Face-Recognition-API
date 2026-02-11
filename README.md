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
 |   - Image validation + decode (PIL)
 |   - NumPy/OpenCV conversion (BGR)
 |   - Registration logic
 |   - Recognition logic (in progress)
 |
 |  ML Layer (InsightFace)
 |   - FaceAnalysis (detect + align)
 |   - Embedding extraction (ArcFace)
 |
 |  Matching Layer
 |   - L2-normalized embeddings
 |   - Cosine similarity (dot product)
 |
 |  Persistence Layer
 |   - PostgreSQL
 |   - SQLAlchemy + Alembic
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

### SQLAlchemy + Alembic
**Why:**
- Explicit data modeling
- Versioned migrations for schema evolution
- Separation of persistence and business logic

---

### Face Detection + Alignment (InsightFace FaceAnalysis)
**Why:**
- InsightFace is a pretrained, production-grade stack with no fine-tuning required
- FaceAnalysis handles detection + alignment with model-consistent preprocessing
- Simplifies edge-case handling (no face / multiple faces) and surfaces detection confidence

---

### Face Embeddings (ArcFace via InsightFace)
**Why:**
- ArcFace embeddings come from the same InsightFace pipeline, keeping assumptions aligned
- L2-normalized embeddings are stored once to keep matching stable
- Explicit embedding extraction keeps verification logic outside the model

---

### Similarity Matching (Cosine Distance)
**Why:**
- With normalized embeddings, cosine similarity is a dot product
- Compatible with pgvector
- Transparent, centralized thresholding

---

## 🧭 Architectural Decisions (Why / How / When)

- Early exploration: I tried manual resizing/cropping and mixing detectors/embedders; it was easy to over-process and hard to keep alignment consistent.
- Later stabilization: I standardized on InsightFace because it is pretrained and production-grade, which let me focus on system design instead of fine-tuning.
- Later stabilization: FaceAnalysis now owns detection + alignment; preprocessing stops at BGR uint8 NumPy arrays because that is the interface it expects.
- Later stabilization: Embeddings are L2-normalized once at extraction and stored normalized; cosine similarity becomes a dot product and avoids double-normalization.
- Later stabilization: Matching lives in its own component (`app/models/matcher.py`) so threshold logic is isolated from model inference.

---

## 🔄 Current Pipeline (As Implemented)

- Validate image bytes by signature and size before decoding.
- Decode via PIL, enforce allowed formats and max dimensions, and apply EXIF orientation.
- Convert to NumPy and OpenCV BGR uint8; preprocessing intentionally stops here.
- Run InsightFace FaceAnalysis for detection + alignment and embedding extraction.
- L2-normalize the embedding and keep the detection score.
- Store normalized embeddings in PostgreSQL (`faces` linked to `users`).
- Match with cosine similarity (dot product) in `InsightFaceMatcher`; the matcher exists even though `/recognize` wiring is still in progress.

---

## 📦 Project Structure

```
app/
├── api/
├── services/
├── models/
├── db/
├── core/
├── schemas/
└── utils/
```

I started with a flatter layout, but the codebase now has clearer ownership boundaries:

- `services/` owns validation + decoding and keeps preprocessing minimal (PIL -> BGR uint8).
- `models/` isolates ML inference (`InsightFaceEmbedder`) and matching (`InsightFaceMatcher`), and the match threshold lives there (default 0.7) so it stays configurable in one place.
- `core/` centralizes config, device selection (`Device`), logging, and shared errors; the embedder is instantiated once via DI for consistent device handling.
- `db/` + `schemas/` define persistence and API contracts; `utils/` stays intentionally thin to avoid a grab-bag.

---

## 🧱 Major Problems & Turning Points

- I started with manual resizing/cropping logic, which quickly became fragile across different inputs.
- The turning point was realizing FaceAnalysis already performs detection + alignment with its own preprocessing, so manual steps were hurting consistency.
- I nearly normalized embeddings twice (once at extraction, once before matching), which made similarity scores drift; normalization is now a single, explicit step.
- Threshold selection turned out to be empirical rather than theoretical, so it lives in the matcher for careful tuning.
- To avoid context rot, I froze these choices into clear boundaries (validation/decoding vs inference vs matching) and stopped moving them around.

---

## ⚠️ Ethics & Privacy

- Explicit consent required
- No real faces committed to repo
- Embeddings are deletable
- Educational / evaluative use only

---

## 🛣️ Roadmap

- [x] API skeleton & PostgreSQL
- [x] Image ingestion & validation
- [x] Face detection + alignment (InsightFace)
- [x] Embeddings & storage
- [ ] Recognition endpoint + matching integration
- [ ] Engineering hardening
- [ ] Advanced optimization

---

> This README will evolve as the project matures.

---

## Current Status / Progress
- FastAPI application is modularized across `api`, `services`, `db`, and `core`.
- SQLAlchemy + Alembic migrations define `users` and `faces`, including embeddings and detection scores.
- Image validation and decoding are implemented (signature checks, size limits, PIL decode with EXIF correction).
- Preprocessing converts to BGR uint8; InsightFace FaceAnalysis handles detection, alignment, and embedding extraction.
- Embeddings are L2-normalized and stored during registration; detection scores are persisted alongside them.
- Matching logic exists in `InsightFaceMatcher` (cosine dot product), but `/recognize` is still being wired.
- Embedder tests cover mocked behavior and schema validation; more end-to-end coverage is planned.

## Why These Technologies (Concise)
- **FastAPI**: Clear dependency injection and automatic OpenAPI docs; good for production REST services.
- **PostgreSQL 18**: Stable relational store with future headroom for vector search extensions.
- **SQLAlchemy + Alembic**: Explicit models plus versioned migrations for safe schema evolution.
- **InsightFace (FaceAnalysis + ArcFace)**: Pretrained, production-grade detection + alignment + embeddings without fine-tuning.
- **Cosine similarity (dot product)**: Works with normalized embeddings and keeps thresholding transparent.
- **Pydantic Settings + .env**: Environment-driven config to keep secrets out of code and support per-environment overrides.

## What I Learned So Far (≈25 hours)
- I learned that generic preprocessing habits (resize, normalize) don't transfer; model-specific preprocessing matters more than my own tricks.
- Respecting the model's training assumptions (BGR uint8 input, alignment) is more reliable than custom pipelines.
- Embeddings are descriptors, not identities; identity decisions live in matching logic and thresholds.
- Normalization and distance metrics are not a detail; L2 + cosine define similarity behavior and can be broken by double-normalization.
- Clear system boundaries (validation/decoding vs inference vs matching) are what keep ML-backed APIs maintainable.
- Keeping knobs explicit (size limits, device selection, centralized thresholds) prevents hidden drift.

## Next Steps (from roadmap_and_milestones.md)
- Wire `/recognize` end-to-end: reuse validation + preprocessing, extract an embedding, and match against stored vectors.
- Make the match threshold configurable (settings/env) and tune it empirically on real samples.
- Add integration tests that hit the full registration + recognition flow with the database.
- Tighten health checks and error handling around the ML pipeline.
- Evaluate `pgvector` or indexing strategy once matching is working reliably.
