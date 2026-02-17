~~# Roadmap & Milestones

This roadmap reflects the **actual implementation state** of the project and the **planned next steps**, aligned with an *internship‑grade, production‑oriented ML backend project*.

---

## Phase 0 – Project Foundation (Completed)

**Goal:** Set up a professional backend foundation.

**Completed:**

* FastAPI application structure
* Environment configuration via `.env`
* Logging system (`app/core/logs.py`, persisted logs)
* Health check endpoint
* Dependency injection pattern (`deps.py`)
* Alembic migrations setup
* SQLAlchemy session & base models
* Pytest configuration and test harness

**Skills demonstrated:**

* Backend project structuring
* Dependency injection
* Database migrations
* Testing fundamentals

---

## Phase 1 – Face Registration Pipeline (Completed)

**Goal:** Register users and persist face embeddings safely.

**Implemented:**

* `/register` endpoint
* UUID‑based user identity (no client‑side IDs)
* `User` and `Face` relational models
* Image validation and preprocessing
* Face detection (single face enforced)
* InsightFace embedding extraction
* Transaction‑safe DB writes

**Design decisions:**

* UUIDs generated server‑side
* Registration schema separate from DB models
* One‑to‑many (`User → Face`) support for extensibility

**Skills demonstrated:**

* ML pipeline integration
* Transaction safety
* Schema vs model separation

---

## Phase 2 – Face Recognition Pipeline (Completed)

**Goal:** Identify users from an uploaded image.

**Implemented:**

* `/recognize` endpoint
* Shared preprocessing & validation logic
* InsightFace embedder loaded once at startup
* Matcher abstraction (cosine similarity + threshold)
* Full DB scan for similarity comparison
* Best‑match selection logic
* Typed response schema with similarity score

**Current limitations (intentional):**

* Linear scan over embeddings
* In‑memory similarity computation

**Skills demonstrated:**

* ML inference services
* Similarity search logic
* Clean error handling for ML edge cases

---

## Phase 3 – API Hardening & Security (Next)

**~~Goal:** Make the API production‑grade.

**Planned:**

* Rate limiting (recognition abuse prevention)
* Request size & content enforcement
* Structured error responses
* Authentication layer (JWT or OAuth2)
* Protected endpoints (`/delete_user`, future admin ops)

**Skills targeted:**

* API security
* AuthN / AuthZ
* Production FastAPI patterns

---

## Phase 4 – Vector Storage & Scaling (Next)

**Goal:** Prepare for real‑world scale.

**Planned:**

* PostgreSQL + pgvector integration
* Vector index (IVFFLAT / HNSW)
* DB‑side similarity search
* Threshold tuning via validation data

**Outcome:**

* Replace O(N) scan with indexed search
* ML‑aware database design

**Skills targeted:**

* Vector databases
* ML system scaling
* Performance engineering

---

## Phase 5 – User Lifecycle Management (Planned)

**Goal:** Complete identity lifecycle.

**Planned:**

* `/delete_user` endpoint
* Auth‑protected destructive actions
* Cascading face deletion
* Audit‑safe operations

**Skills targeted:**

* Secure CRUD design
* Data integrity

---

## Phase 6 – Deployment & DevOps (Planned)

**Goal:** Make it deployable anywhere.

**Planned:**

* Dockerfile (API + model runtime)
* Docker Compose (API + Postgres)
* Environment‑based configs
* Optional CI pipeline

**Skills targeted:**

* Containerization
* DevOps fundamentals

---

## Phase 7 – Optional UI & Webcam Support (Optional)

**Goal:** Showcase end‑to‑end system.

**Planned:**

* Webcam capture (browser)
* Simple frontend (React / HTML)
* Live recognition demo

**Note:** Optional, value is *demonstration*, not core ML skill.

---

## Final Outcome

By the end of this roadmap, the project demonstrates:

* End‑to‑end ML backend ownership
* Production‑grade API design
* Vector similarity search at scale
* Real‑world face recognition constraints

This positions the project as **strong internship / junior ML engineer level**, with clear signals for backend + ML systems roles.
