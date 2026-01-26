# What to Learn, Why, and When (Function-Level Guide)

This document tells you **exactly what to learn**, **why you need it**, and **when** in the project timeline,
mapped directly to **functions and modules** you will implement.

This does **not** change the project structure, logic, or roadmap.

---

## Phase 1 — API Skeleton & File Uploads

### Learn
- FastAPI core concepts (`FastAPI`, `APIRouter`)
- Request/response lifecycle
- File uploads (`UploadFile`, `File`)
- HTTP status codes
- Pydantic schemas

### Why
You must validate the system architecture before adding ML complexity.

### Used for
- `/register` endpoint
- `/recognize` endpoint
- File validation logic

---

## Phase 2 — Image Validation & Decoding

### Learn
- Image formats (`.jpg`, `.png`)
- Byte streams vs decoded images
- Image decoding with PIL or OpenCV
- NumPy arrays

### Why
ML models operate on arrays, not raw bytes.

### Used for
- `validate_image_format(file)`
- `decode_image(bytes)`

---

## Phase 3 — Face Detection

### Learn
- Face detection vs recognition
- Pretrained detection models
- Bounding boxes
- Edge cases (no face, multiple faces)

### Why
Recognition requires isolating the face region.

### Used for
- `detect_face(image)`

---

## Phase 4 — Face Cropping & Normalization

### Learn
- Cropping using bounding boxes
- Image resizing
- Pixel normalization

### Why
Embedding models expect standardized input.

### Used for
- `crop_face(image, box)`
- `resize_face(face)`

---

## Phase 5 — Embeddings (Core ML Step)

### Learn
- What embeddings are
- Vector dimensionality
- L2 normalization
- Cosine similarity intuition

### Why
Embeddings represent identity and enable comparison.

### Used for
- `extract_embedding(face_image)`

---

## Phase 6 — Database & Persistence (PostgreSQL)

### Learn
- PostgreSQL fundamentals
- Table design
- SQLAlchemy ORM
- Storing vectors as arrays

### Why
You are building a production-style system.

### Used for
- `faces` table
- `insert_embedding()`
- `fetch_embeddings()`

---

## Phase 7 — Recognition Logic

### Learn
- Similarity metrics
- Threshold-based matching
- False positives vs false negatives

### Why
Recognition must be confident, not just closest.

### Used for
- `find_best_match(embedding)`
- Match threshold logic

---

## Phase 8 — Error Handling & Logging

### Learn
- Custom exceptions
- Mapping errors to HTTP responses
- Logging levels

### Why
Production systems must fail safely.

### Used for
- Custom error classes
- API error responses

---

## Phase 9 — pgvector Optimization

### Learn
- Vector databases
- pgvector extension
- Vector indexing (IVFFLAT, HNSW)
- SQL similarity queries

### Why
This enables scalable, industry-grade similarity search.

### Used for
- Vector columns
- DB-side similarity search

---

## Phase 10 — Testing

### Learn
- Pytest basics
- Mocking file uploads
- Testing ML pipelines

### Why
Tests protect your reasoning and refactors.

### Used for
- API tests
- Pipeline tests

---

## Phase 11 — Optional: Docker

### Learn
- Dockerfiles
- docker-compose
- Environment variables

### Why
Deployment and reproducibility.

### Used for
- Dockerized API and database

---

## Final Note

If you can explain:
- why each step exists
- where it fits in the pipeline
- what breaks if it fails

Then you truly understand the system.
