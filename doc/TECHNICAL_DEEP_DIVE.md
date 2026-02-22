# TECHNICAL DEEP DIVE --- Face Recognition API

This document is intended for technical interviews and deeper
architectural discussions.

------------------------------------------------------------------------

# 1. System Evolution

## Phase 1 --- Baseline

-   Embeddings stored as ARRAY(Float)
-   Python-side cosine similarity loop
-   Full table scan
-   O(N) scaling

Observed behavior: - 10,000 rows → \~4.6s per request - Linear growth
confirmed

Root cause: - Embeddings fetched into Python - Similarity computed
outside DB

------------------------------------------------------------------------

## Phase 2 --- pgvector Migration

Changes: - Migrated column to Vector(512) - Added HNSW index
(cosine_ops) - Similarity computed in PostgreSQL

Impact: - Flat 2--8ms DB latency - 241× speedup at 5k rows - No
embedding transfer

Design considerations: - Chose HNSW for better recall/latency
trade-off - Avoided IVFFLAT due to dataset growth expectations -
Preserved cosine similarity to align with L2-normalized embeddings

------------------------------------------------------------------------

# 2. Concurrency Architecture

## Problem

CPU-bound inference (\~1,300ms per request) blocks event loop.

## Solution 1 --- ThreadPoolExecutor

-   Used asyncio.run_in_executor
-   Prevented event loop starvation
-   Allowed concurrent request acceptance

Limitation: - Still GIL-bound - No true parallelism in single worker

------------------------------------------------------------------------

## Solution 2 --- Multi-Worker Deployment

uvicorn --workers 4

Each worker: - Separate process - Separate GIL - Separate model instance

Results: - \~2× throughput increase - \~500MB RAM per worker -
Sub-linear scaling (CPU core contention)

------------------------------------------------------------------------

# 3. Bottleneck Analysis

Layer 1 --- DB: Solved with pgvector.

Layer 2 --- Inference: Dominates 99.8% of latency.

Layer 3 --- Concurrency: Limited by CPU inference + GIL.

Conclusion: Database is no longer the constraint. Model inference
hardware is.

------------------------------------------------------------------------

# 4. Trade-Offs Considered

Why pgvector over external vector DB? - Simpler deployment - ACID
consistency - No additional infrastructure

Why HNSW? - Better performance at moderate dataset sizes - Lower tuning
burden

Why JWT instead of OAuth server? - Single API context - Reduced
infrastructure complexity

Why buffalo_l model? - Accuracy priority over CPU speed

------------------------------------------------------------------------

# 5. Scaling Path Forward

Options ranked by impact:

1.  GPU inference (\~10--50ms expected)
2.  Lighter model (buffalo_sc)
3.  Horizontal scaling with load balancer
4.  Further index tuning (ef_search)

------------------------------------------------------------------------

# 6. System Maturity Signals

-   Clean architectural boundaries
-   Performance measured before optimization
-   Scaling behavior documented
-   Security integrated early
-   No hidden coupling between ML and API layers

------------------------------------------------------------------------

# Interview Discussion Topics

Be ready to discuss:

-   Why async does not equal parallelism
-   GIL implications for ML inference
-   Vector indexing strategies (HNSW vs IVFFLAT)
-   Threshold tuning methodology
-   Memory vs throughput trade-offs
-   CPU vs GPU scaling economics
