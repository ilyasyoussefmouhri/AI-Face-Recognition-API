# Face Recognition API (Production-Oriented, API-First)

> **Status:** Actively benchmarked and performance-profiled  
> **Goal:** Build a production-grade facial recognition REST API using pretrained models, PostgreSQL + pgvector, and modern Python backend practices.  
> **Target audience:** ML / Backend internship recruiters, engineers, and technical reviewers.

---

## 🚀 Project Motivation

This project is not a demo or tutorial exercise.

It is intentionally designed to:

- Mirror real-world ML backend systems
- Use industry-relevant technologies
- Demonstrate engineering judgment, trade-offs, and performance reasoning
- Prioritize system design and scaling over model training

The early versions proved functionality. The current stage proves scalability, bottleneck isolation, and architectural maturity.

This project now includes:

- Vector-indexed similarity search
- Structured multi-layer benchmarking
- Concurrency analysis
- Worker scaling experiments
- CPU bottleneck isolation

It has moved from "it works" to "it scales — and we know why."

---

## 🧠 Core Problem

Build a REST API that can:

- Register a person by extracting and storing a facial embedding from an image
- Recognize a person by matching an input image against stored embeddings
- Remain secure under authentication + rate limiting
- Scale beyond naive in-memory similarity matching
- Provide measurable, benchmarked performance characteristics

All while using pretrained models only (no fine-tuning).

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

## 🔄 Current Pipeline (As Implemented, Phase 2) 

- Validate image bytes by signature and size before decoding.
- Decode via PIL, enforce allowed formats and max dimensions, and apply EXIF orientation.
- Convert to NumPy and OpenCV BGR uint8; preprocessing intentionally stops here.
- Run InsightFace FaceAnalysis for detection + alignment and embedding extraction.
- L2-normalize the embedding and keep the detection score.
- Store normalized embeddings in PostgreSQL (`faces` linked to `users`).
- Match with cosine similarity (dot product) in `InsightFaceMatcher`; `/recognize` now wires the matcher and returns the best-scoring identity behind auth.

---

## 🏗️ High-Level Architecture (Current State)

```
Client (HTTP / Load Test)
        |
        v
-------------------------
 FastAPI Application
-------------------------
 |  API Layer
 |   - /auth/*
 |   - /register
 |   - /recognize
 |
 |  Services Layer
 |   - Image validation + decode
 |   - Registration / Recognition pipelines
 |   - Auth & deletion flows
 |
 |  ML Layer (InsightFace)
 |   - FaceAnalysis (detect + align)
 |   - Embedding extraction (ArcFace)
 |   - ThreadPoolExecutor execution
 |
 |  Matching Layer
 |   - Cosine similarity
 |   - pgvector HNSW index (DB-side similarity)
 |
 |  Persistence Layer
 |   - PostgreSQL 18
 |   - pgvector (Vector(512))
 |   - HNSW index (cosine_ops)
 |
-------------------------
```

Major evolution:

- Similarity search now runs inside PostgreSQL
- Inference runs in a thread pool
- Concurrency behavior is benchmarked
- Worker scaling behavior is measured

---

## 🔧 Major Architectural Evolution

### Phase 2 → Phase 4 Turning Point: pgvector Migration

**Originally:**

- Embeddings stored as `ARRAY(Float)`
- Full table scan
- Python cosine loop
- O(N) scaling

At 10,000 rows: ~4,600ms per request (DB + similarity), 17MB transferred to Python per call.

**Now:**

- `Vector(512)` column
- HNSW index (m=16, ef_construction=64)
- Cosine similarity computed inside PostgreSQL

**Result:** 241× speedup at 5,000 rows, DB latency flat at 2–8ms, no embedding transfer to Python, dataset size irrelevant up to 10k.

The database bottleneck is now solved.

---

## 📊 Performance Benchmarking

This project includes structured benchmarking across three layers:

- **Layer 1 — Direct DB:** pre/post pgvector comparison, O(N) → flat scaling
- **Layer 2 — Sequential HTTP:** full pipeline breakdown, inference vs DB cost
- **Layer 3 — Concurrent HTTP:** queuing behaviour, 1 vs 4 workers, throughput characterization

For full results, data tables, and analysis see **[PERFORMANCE.md](./PERFORMANCE.md)**.

### Current System Cost Breakdown

| Component | Time | % of Total |
|---|---|---|
| InsightFace inference (CPU) | ~1,300ms | ~99.8% |
| pgvector query | 2–8ms | ~0.2% |
| JWT validation | <1ms | negligible |

The database layer is no longer a scaling concern. Inference is the only true bottleneck.

---

## 🧠 Concurrency Architecture Improvements

Two important changes were introduced:

### 1. ThreadPoolExecutor

Inference now runs via `asyncio.run_in_executor`.

- Prevents blocking the FastAPI event loop
- Allows concurrent request acceptance
- Enables realistic async load behavior

Without this, concurrency testing would be meaningless.

### 2. Multi-Worker Uvicorn Deployment

Using `uvicorn app.main:app --workers 4`. Each worker is a separate process with its own GIL and its own model instance.

**Trade-off:** N× model memory, improved throughput, slight single-user latency regression due to CPU contention.

---

## 📦 Updated Project Structure

```
app/
├── api/routes/
├── services/
├── models/
├── db/
├── core/
├── schemas/
├── utils/
└── main.py
```

Additional:

- `tests/` includes embedder + recognition tests
- Benchmark scripts exist under dedicated benchmark directory
- Coverage reports generated
- Alembic migrations track schema evolution

Separation remains clean: API layer thin, services contain orchestration, ML models isolated, matching isolated, DB layer explicit, security centralized.

---

## ⚠️ Ethics & Privacy

- Explicit consent required
- No real faces committed to repo
- Embeddings are deletable
- Educational / evaluative use only

---

## 🔐 Security & Production Hardening (Completed)

- ✔ JWT authentication
- ✔ bcrypt password hashing
- ✔ Role-based access control
- ✔ Rate limiting per route
- ✔ Cascade-safe deletion
- ✔ Environment-based configuration
- ✔ Structured logging

The API is now secure enough for real deployment scenarios.

---

## 🛣️ Updated Roadmap Alignment

### Phase 4 – Vector Storage & Scaling

- ✔ pgvector integration
- ✔ HNSW index
- ✔ DB-side similarity
- ✔ Performance benchmarking

### Next Steps

- Threshold tuning with validation data
- End-to-end tests (auth + register + recognize + delete)
- GPU inference benchmarking
- Model comparison (buffalo_l vs buffalo_sc)
- Production observability improvements
- Memory profiling under multi-worker load

---

## 🧠 What I Learned So Far (~40+ hours)

- Scaling problems rarely sit where you expect.
- Optimizing the wrong layer wastes time — benchmarking first prevents that.
- Vector databases eliminate O(N) similarity cost entirely.
- CPU-bound ML inference behaves very differently under concurrency.
- Async does not magically create parallelism.
- GIL-bound workloads require multi-process scaling or GPU.
- Production ML systems are 80% systems engineering, 20% model usage.
- Worker count trades memory for throughput.
- Profiling before optimization is non-negotiable.

---

## 🎯 Current Status

This is no longer just a face recognition API. It is now:

- A production-oriented ML backend
- With indexed vector similarity search
- With security and lifecycle management
- With structured performance benchmarking
- With concurrency characterization
- With real scaling trade-offs documented

The database bottleneck is solved. The system is ready for either GPU acceleration, horizontal scaling, or model optimization.

This README will continue evolving as the system approaches GPU acceleration and deployment-grade readiness.