# Performance Analysis

**Date:** 2026-02-22  
**System:** CPU only, no GPU  
**Model:** InsightFace buffalo_l (ArcFace, 512-dim embeddings)  
**DB:** PostgreSQL 18 + pgvector, HNSW index (cosine ops, m=16, ef_construction=64)  
**Stack:** FastAPI, SQLAlchemy 2.0, psycopg v3, Docker  

---

## Benchmark Architecture

Three complementary layers were run to fully characterize system performance. Each layer isolates a different part of the stack.

**Layer 1 — Direct DB:** bypasses HTTP and ML entirely. Measures pure DB fetch + similarity loop. Isolates exactly what pgvector replaces.

**Layer 2 — Sequential HTTP:** full pipeline over HTTP with a real face image. Measures total end-to-end latency including image decode, InsightFace inference, and DB query.

**Layer 3 — Concurrent HTTP:** multiple simultaneous users via asyncio. Measures queuing behaviour and throughput under load, across 1 and 4 Uvicorn workers.

---

## Layer 1: Direct DB Benchmark

Method: SQLAlchemy query against live PostgreSQL, 30 iterations per dataset size, no HTTP overhead, no ML inference.

### Pre-pgvector — ARRAY(Float) + Python similarity loop

| n | avg_ms | p95_ms | p99_ms | db_ms | sim_ms | memory_MB |
|---|---|---|---|---|---|---|
| 100 | 34.44 | 42.13 | 42.98 | 33.46 | 0.98 | 1.229 |
| 1,000 | 401.33 | 626.78 | 689.62 | 389.13 | 12.19 | 5.300 |
| 5,000 | 2,139.00 | 2,380.52 | 2,381.49 | 2,077.29 | 61.71 | 14.343 |
| 10,000 | 4,638.49 | 5,024.23 | 5,199.32 | 4,500.61 | 137.89 | 17.108 |

O(N) confirmed. Latency ratio (ms/n) is constant across all sizes at ~0.45ms per row. At 10,000 rows, 17MB of embedding data is transferred to Python on every single request.

### Post-pgvector — Vector(512) + HNSW index

| n | avg_ms | p95_ms | p99_ms | db_ms | sim_ms | memory_MB | speedup |
|---|---|---|---|---|---|---|---|
| 100 | 8.33 | 10.93 | 11.97 | 8.33 | 0.0 | 0.000 | 4× |
| 1,000 | 8.16 | 10.84 | 11.07 | 8.16 | 0.0 | 0.000 | 49× |
| 5,000 | 8.85 | 10.66 | 14.42 | 8.85 | 0.0 | 0.000 | 241× |
| 10,000 | 43.95 | 51.42 | 51.46 | 43.95 | 0.0 | 0.000 | 105× |

Latency is flat from 100 to 5,000 rows. Similarity time is 0ms — computed entirely inside PostgreSQL via the HNSW index. Memory per request dropped to ~0MB. The slight growth at 10,000 rows is normal HNSW graph traversal at higher N and remains O(log N).

**Key implementation note:** the HNSW index only fires when the query vector is passed as a literal string parameter. Passing via ORM (`cosine_distance()`) or subquery causes PostgreSQL to fall back to a sequential scan. The production query uses `CAST(:vec AS vector)` via SQLAlchemy `text()` — the `::vector` PostgreSQL cast syntax conflicts with SQLAlchemy's `:param` notation.

---

## Layer 2: Sequential HTTP Benchmark

Method: sequential requests with a real face image (1280×1280 JPEG), 30 iterations per dataset size, BENCHMARK_MODE=true for header-based timing breakdown.

| n | avg_ms | p95_ms | p99_ms | db_ms | inference_ms |
|---|---|---|---|---|---|
| 100 | 1,325 | 1,480 | 1,609 | 2.38 | 1,323 |
| 1,000 | 1,578 | 1,811 | 2,467 | 2.44 | 1,575 |
| 5,000 | 1,446 | 1,725 | 1,767 | 4.54 | 1,442 |
| 10,000 | 1,370 | 1,657 | 1,710 | 2.73 | 1,368 |

Total latency is flat across all dataset sizes. The ~200ms variance between runs exceeds the difference between n=100 and n=10,000 — dataset size has no measurable effect on end-to-end latency.

**Breakdown:**

| component | time | share |
|---|---|---|
| InsightFace inference (CPU) | ~1,300ms | 99.8% |
| pgvector HNSW query | 2–5ms | 0.2% |
| image decode + preprocessing | included above | — |
| JWT decode + validation | <1ms | negligible |

The database bottleneck is fully solved. Inference is the only remaining constraint.

---

## Layer 3: Concurrent HTTP Benchmark

Method: `asyncio.gather` fires N requests simultaneously, 5 batches per combination, real face image, BENCHMARK_MODE=true.

### 1 Worker

| n | c=1 avg_ms | c=5 avg_ms | c=20 avg_ms | c=1 rps | c=5 rps | c=20 rps | c=1 db_ms | c=20 db_ms |
|---|---|---|---|---|---|---|---|---|
| 100 | 830 | 5,836 | 21,679 | 1.20 | 0.84 | 0.75 | 2.04 | 2.49 |
| 1,000 | 1,387 | 6,410 | 21,002 | 0.72 | 0.75 | 0.77 | 6.72 | 2.98 |
| 5,000 | 1,370 | 6,505 | 21,113 | 0.73 | 0.77 | 0.76 | 3.05 | 3.74 |
| 10,000 | 1,544 | 6,761 | 22,118 | 0.64 | 0.74 | 0.73 | 9.57 | 4.36 |

### 4 Workers

| n | c=1 avg_ms | c=5 avg_ms | c=20 avg_ms | c=1 rps | c=5 rps | c=20 rps | c=1 db_ms | c=20 db_ms |
|---|---|---|---|---|---|---|---|---|
| 100 | 1,800 | 5,111 | 11,414 | 0.55 | 0.72 | 1.34 | 2.20 | 6.34 |
| 1,000 | 1,490 | 4,121 | 11,164 | 0.67 | 0.98 | 1.32 | 7.05 | 5.99 |
| 5,000 | 1,516 | 4,546 | 10,268 | 0.66 | 0.88 | 1.44 | 2.91 | 7.02 |
| 10,000 | 1,722 | 4,452 | 10,305 | 0.58 | 0.94 | 1.50 | 7.66 | 7.18 |

### Workers=1 vs Workers=4 at n=1,000

| concurrency | w=1 avg_ms | w=4 avg_ms | latency improvement | w=1 rps | w=4 rps | throughput improvement |
|---|---|---|---|---|---|---|
| 1 | 1,387ms | 1,490ms | −7% (regression) | 0.72 | 0.67 | −7% |
| 5 | 6,410ms | 4,121ms | 36% faster | 0.75 | 0.98 | 31% |
| 20 | 21,002ms | 11,164ms | 47% faster | 0.77 | 1.32 | 71% |

### Findings

**Requests serialize on CPU with 1 worker.** c=5 latency ≈ 5× single-user, c=20 ≈ 20× single-user. Throughput is flat at ~0.75 rps regardless of concurrency level — adding more users increases wait time, not throughput. This is GIL-bound CPU queuing on InsightFace inference.

**4 workers roughly doubles throughput under concurrent load.** At c=20, throughput goes from 0.75 to ~1.5 rps. Latency under load roughly halves. Scaling is sub-linear because 4 workers still share the same physical CPU cores.

**Single-user latency slightly regresses with 4 workers** due to CPU contention between 4 model instances competing for the same cores.

**DB time is unaffected by concurrency or worker count** — stays flat at 2–10ms across all combinations tested. pgvector is not a bottleneck under any load condition tested.

---

## Path to Further Improvement

| option | expected inference_ms | trade-off |
|---|---|---|
| Current (CPU, buffalo_l, 1280px) | ~1,300ms | baseline |
| Input resize to 640×640 | ~800–1,000ms | minimal |
| buffalo_sc model (CPU) | ~400–600ms | lower accuracy |
| N Uvicorn workers | ~1,300ms / N throughput | N × ~500MB RAM |
| GPU inference | ~10–50ms | hardware cost |

The only path to sub-100ms inference is GPU. Everything else is incremental.