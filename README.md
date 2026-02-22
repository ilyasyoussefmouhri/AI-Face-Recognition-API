# Face Recognition API (Production-Oriented ML Backend)

> **Status:** Performance-benchmarked & production-hardened\
> **Focus:** ML systems engineering, scalability, and backend
> architecture\
> **Audience:** ML / Backend internship recruiters and engineers

------------------------------------------------------------------------

## 🚀 Executive Summary

This project is a **production-oriented facial recognition backend**
built with FastAPI, PostgreSQL 18, and pgvector.

It demonstrates:

-   End-to-end ML API ownership
-   Vector-indexed similarity search (HNSW)
-   Structured performance benchmarking (3 layers)
-   Concurrency and worker scaling analysis
-   Secure authentication (JWT + RBAC)
-   Clear architectural boundaries

The system evolved from a naive O(N) similarity scan to a
vector-indexed, benchmarked ML backend with documented bottlenecks and
scaling characteristics.

------------------------------------------------------------------------

## 🧠 What Problem Does It Solve?

The API can:

-   Register users by extracting facial embeddings
-   Recognize users via similarity search
-   Enforce authentication and rate limiting
-   Scale beyond in-memory similarity matching
-   Provide measurable performance characteristics

All using pretrained models (InsightFace ArcFace).

------------------------------------------------------------------------

## 🏗️ Architecture Overview

Client → FastAPI → Services → ML (InsightFace) → PostgreSQL + pgvector

Key properties:

-   ThreadPoolExecutor for CPU-bound inference
-   HNSW index for similarity search
-   Clean separation of API / services / ML / DB
-   JWT-based security model
-   Rate limiting on expensive endpoints

------------------------------------------------------------------------

## 📊 Performance Highlights

### Database Scaling

Before pgvector: - O(N) similarity scan - 4,638ms at 10,000 rows - 17MB
transferred per request

After pgvector (HNSW): - Flat 2--8ms DB latency - 241× speedup at 5k
rows - Zero embedding transfer to Python

Database bottleneck eliminated.

For full results, data tables, and analysis see **[PERFORMANCE.md](./PERFORMANCE.md)**.

------------------------------------------------------------------------

### End-to-End Latency (CPU)

-   \~1,300ms total per request
-   \~99.8% inference time
-   \~0.2% database time

The remaining bottleneck is model inference.

------------------------------------------------------------------------

### Concurrency

1 Worker: - Throughput capped at \~0.75 rps - Linear latency scaling
(GIL-bound)

4 Workers: - \~2× throughput increase - Memory trade-off (\~500MB per
worker) - Sub-linear scaling due to shared CPU cores

------------------------------------------------------------------------

## 🔐 Production Hardening

-   JWT authentication (HS256)
-   bcrypt password hashing
-   Role-based access control
-   Per-route rate limiting
-   Cascade-safe deletion
-   Structured logging
-   Environment-based configuration

------------------------------------------------------------------------

## 🧠 Engineering Takeaways

-   Vector databases eliminate O(N) similarity cost.
-   Async does not provide CPU parallelism.
-   CPU-bound ML workloads require multiprocessing or GPU.
-   Benchmarking must precede optimization.
-   Production ML systems are primarily systems engineering.

---

## ⚠️ Ethics & Privacy

- Explicit consent required
- No real faces committed to repo
- Embeddings are deletable
- Educational / evaluative use only

---

------------------------------------------------------------------------

## 🛣️ Next Steps

-   GPU inference benchmarking
-   Threshold tuning with validation data
-   Full end-to-end integration tests
-   Observability improvements

------------------------------------------------------------------------

## 🎯 Why This Project Matters

This project demonstrates:

-   Real performance measurement
-   Scaling trade-offs
-   Bottleneck isolation
-   Production-ready backend structure
-   ML integration under realistic constraints

It is not just an ML demo --- it is a systems-engineered backend with
measurable behavior.
