# Face Recognition API (Production-Oriented ML Backend)

> **Status:** Performance-benchmarked & production-hardened\
> **Focus:** ML systems engineering, scalability, and backend
> architecture\
> **Audience:** ML / Backend internship recruiters and engineers

------------------------------------------------------------------------

## 🚀 Executive Summary

This project is a **production-oriented Face-Based Identity Verification Microservice backend**
built with FastAPI, PostgreSQL 18, and pgvector.

It demonstrates:

-   End-to-end ML API ownership
-   Vector-indexed similarity search (HNSW)
-   Structured performance benchmarking (3 layers)
-   Concurrency and worker scaling analysis
-   Secure authentication (JWT + RBAC)
-   Clear architectural boundaries
---
- The system evolved from a naive O(N) similarity scan to a
vector-indexed, benchmarked ML backend with documented bottlenecks and
scaling characteristics.
- This is not an end-user product. It’s a biometric authentication microservice designed to be embedded inside larger systems like access control, KYC onboarding, or passwordless login systems. The goal of this project was systems engineering and performance characterization, not product-market fit.

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

# 🛣️ Next Steps: Future Improvements

While the core system is functionally complete and performance-characterized, the following extensions have been identified to transition the project from a "production-ready prototype" toward a **production deployment candidate**.

---

### 1. Observability & Monitoring
* **Prometheus Metrics:** Add an endpoint to track request count, latency, inference time, and DB query duration.
* **Structured Tracing:** Implement request IDs for end-to-end tracing.
* **Alerting:** Set thresholds for abnormal latency or error rates.

### 2. Biometric Threshold Calibration
Build a validation harness to empirically tune the similarity threshold rather than relying on fixed defaults. This includes computing:
* **False Accept Rate (FAR)**
* **False Reject Rate (FRR)**
* **ROC Curve Analysis:** Documenting decision boundary trade-offs for real-world reliability.



### 3. Sustained Load Testing
* **Tooling:** Replace ad-hoc concurrency testing with **Locust** or **k6**.
* **Ramp-up Testing:** Shift to ramp-up RPS testing instead of burst-only benchmarking.
* **Latency Percentiles:** Observe system behavior and graceful degradation under sustained saturation.

### 4. Hardware-Aware Scaling
* **Device Switching:** Add configurable CPU/GPU switching.
* **Model Benchmarking:** Compare performance and accuracy trade-offs between `buffalo_l` and `buffalo_sc` models.

### 5. Backpressure & Graceful Degradation
* **Concurrency Semaphore:** Implement a semaphore for inference to prevent resource exhaustion.
* **Saturation Handling:** Return `429 Too Many Requests` when the system is at capacity.
* **Queue Management:** Prevent unbounded request queue growth under high load.

### 6. CI/CD Improvements
* **Test Coverage:** Expand automated unit and integration tests.
* **Static Analysis:** Add **mypy** for type checks and enforce linting/formatting in the pipeline.

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
