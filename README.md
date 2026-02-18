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

The goal is to showcase my ability to build, reason about, and evolve an applied computer vision system under realistic constraints — first as a barebones ML API, now as a secure, authenticated service that can survive real traffic.

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
- Security turn: Introduced an `AuthUser` (credentials, roles) separate from `User` (biometrics) so credentials are never coupled to embeddings; the one-to-one link keeps ownership explicit.
- Security turn: Chose stateless HS256 JWTs over running a full OAuth2 server to keep deployment light while still enforcing expiry, roles, and bearer flows that clients already understand.
- Security turn: Added rate limiting at the edge (SlowAPI) because `/recognize` and `/register` are the most expensive and most abusable paths; limits are per route so we can tune recognition differently from auth.
- Security turn: Cascade deletes are intentional — removing an `AuthUser` scrubs the biometric shadow it owns, while admins can prune biometric users independently when needed.

---

## 🔄 Current Pipeline (As Implemented)

- Validate image bytes by signature and size before decoding.
- Decode via PIL, enforce allowed formats and max dimensions, and apply EXIF orientation.
- Convert to NumPy and OpenCV BGR uint8; preprocessing intentionally stops here.
- Run InsightFace FaceAnalysis for detection + alignment and embedding extraction.
- L2-normalize the embedding and keep the detection score.
- Store normalized embeddings in PostgreSQL (`faces` linked to `users`).
- Match with cosine similarity (dot product) in `InsightFaceMatcher`; `/recognize` now wires the matcher and returns the best-scoring identity behind auth.

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

I started with a flatter layout, but the codebase now has clearer ownership boundaries (and an explicit auth layer):

- `services/` owns validation + decoding, and now also authentication, authorization, and deletion flows.
- `models/` isolates ML inference (`InsightFaceEmbedder`) and matching (`InsightFaceMatcher`), and the match threshold lives there (default 0.7) so it stays configurable in one place.
- `core/` centralizes config, device selection (`Device`), logging, shared errors, the rate limiter, and security helpers (bcrypt + JWT).
- `db/` + `schemas/` define persistence and API contracts; `utils/` stays intentionally thin to avoid a grab-bag.

---

## 🧱 Major Problems & Turning Points

- I started with manual resizing/cropping logic, which quickly became fragile across different inputs.
- The turning point was realizing FaceAnalysis already performs detection + alignment with its own preprocessing, so manual steps were hurting consistency.
- I nearly normalized embeddings twice (once at extraction, once before matching), which made similarity scores drift; normalization is now a single, explicit step.
- Threshold selection turned out to be empirical rather than theoretical, so it lives in the matcher for careful tuning.
- To avoid context rot, I froze these choices into clear boundaries (validation/decoding vs inference vs matching) and stopped moving them around.
- The API hardening phase forced domain separation (`AuthUser` vs `User`), explicit cascade behavior, and bearer tokens with expirations — small decisions that collectively reduce blast radius.

---

## ⚠️ Ethics & Privacy

- Explicit consent required
- No real faces committed to repo
- Embeddings are deletable
- Educational / evaluative use only

---

## 🔐 API Hardening Milestone (Phase 3)

What changed: the API grew up. Auth lives in `app/api/routes/auth.py` with bcrypt-hashed credentials, HS256 JWTs, and per-route rate limits. The domain is split: `AuthUser` accounts own exactly one biometric `User`, and cascade deletes ensure credentials and embeddings cannot drift apart. Admins can prune biometric users, while anyone can self-delete; all destructive paths are dependency-guarded (`get_current_user`, `get_current_admin`).

Why JWT (and not a full OAuth2 server): the system is a single API, not an auth provider. Stateless JWTs keep deployment light, let us encode roles and expirations, and align with FastAPI’s bearer tooling without running extra infrastructure.

Why rate limiting: `/recognize` and `/register` are CPU-heavy and attractive for brute force; SlowAPI enforces route-specific ceilings so auth and inference can be tuned independently.

Why domain separation: credentials are mutable; biometric embeddings are not. Keeping them in separate tables with a one-to-one relationship lets us rotate passwords, disable accounts, or cascade-delete safely without touching embeddings by accident.

## 🛣️ Roadmap

- [x] API skeleton & PostgreSQL
- [x] Image ingestion & validation
- [x] Face detection + alignment (InsightFace)
- [x] Embeddings & storage
- [x] Recognition endpoint + matching integration
- [x] API hardening (auth, JWT, RBAC, rate limits, cascade-safe deletes)
- [ ] Scaling with pgvector + indexed similarity search
- [ ] Advanced optimization

---

> This README will evolve as the project matures.

---

## Current Status / Progress
- Secure FastAPI application with JWT auth, per-route rate limits, and RBAC-protected registration/recognition/deletion.
- SQLAlchemy models separate `AuthUser` (credentials/roles) from biometric `User` + `Face`; cascade deletion is intentional.
- Image validation and decoding are implemented (signature checks, size limits, PIL decode with EXIF correction).
- Preprocessing converts to BGR uint8; InsightFace FaceAnalysis handles detection, alignment, and embedding extraction.
- Embeddings are L2-normalized and stored during registration; detection scores are persisted alongside them.
- `/recognize` runs end-to-end: embed, cosine-match via `InsightFaceMatcher`, return best identity behind auth.
- Tests cover the embedder and service layers; broader end-to-end auth + inference tests are next.
- Ready to enter the pgvector scaling phase after validating thresholds under load.

## Why These Technologies (Concise)
- **FastAPI**: Clear dependency injection and automatic OpenAPI docs; good for production REST services.
- **PostgreSQL 18**: Stable relational store with future headroom for vector search extensions.
- **SQLAlchemy + Alembic**: Explicit models plus versioned migrations for safe schema evolution.
- **InsightFace (FaceAnalysis + ArcFace)**: Pretrained, production-grade detection + alignment + embeddings without fine-tuning.
- **Cosine similarity (dot product)**: Works with normalized embeddings and keeps thresholding transparent.
- **Pydantic Settings + .env**: Environment-driven config to keep secrets out of code and support per-environment overrides.
- **JWT + bcrypt + SlowAPI**: Lightweight auth with hashed credentials, expiring tokens, and abuse resistance without standing up extra services.

## What I Learned So Far (≈25 hours)
- I learned that generic preprocessing habits (resize, normalize) don't transfer; model-specific preprocessing matters more than my own tricks.
- Respecting the model's training assumptions (BGR uint8 input, alignment) is more reliable than custom pipelines.
- Embeddings are descriptors, not identities; identity decisions live in matching logic and thresholds.
- Normalization and distance metrics are not a detail; L2 + cosine define similarity behavior and can be broken by double-normalization.
- Clear system boundaries (validation/decoding vs inference vs matching) are what keep ML-backed APIs maintainable.
- Keeping knobs explicit (size limits, device selection, centralized thresholds) prevents hidden drift.

## Next Steps (from roadmap_and_milestones.md)
- Move embeddings to `pgvector` and add an index (IVFFLAT/HNSW) to replace the linear scan.
- Expose and tune match thresholds via settings with validation data.
- Add end-to-end tests that cover auth + registration + recognition + deletion.
- Harden observability around auth failures and rate-limit breaches before load testing.
