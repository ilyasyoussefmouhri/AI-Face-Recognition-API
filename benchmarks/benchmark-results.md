# Benchmark Results — Face Recognition API

**Date:** 2026-02-22  
**System:** CPU only (no GPU)  
**Model:** InsightFace buffalo_l (ArcFace, 512-dim embeddings)  
**DB:** PostgreSQL 18 + pgvector, HNSW index (cosine ops, m=16, ef_construction=64)  
**Stack:** FastAPI, SQLAlchemy 2.0, psycopg v3, Docker  

---

## Overview

Three benchmark layers were run to fully characterize system performance:

1. **Direct DB benchmark** — bypasses HTTP and ML, measures pure DB fetch + similarity loop. Isolates what pgvector replaces.
2. **Sequential HTTP benchmark** — full pipeline over HTTP with a real face image. Measures total end-to-end latency including InsightFace inference.
3. **Concurrent HTTP benchmark** — multiple simultaneous users. Measures queuing behaviour and throughput under load, with 1 worker and 4 workers.

---

## Layer 1: Direct DB Benchmark (no HTTP, no ML)

Measures: DB fetch time + Python cosine similarity loop  
Method: SQLAlchemy query against live PostgreSQL, 30 iterations per size  

### Pre-pgvector (ARRAY(Float) + Python similarity loop)

| dataset_size | avg_ms | p95_ms | db_ms | sim_ms | memory_MB |
|---|---|---|---|---|---|
| 100 | 34.44 | 42.13 | 33.46 | 0.98 | 1.229 |
| 1,000 | 401.33 | 626.78 | 389.13 | 12.19 | 5.300 |
| 5,000 | 2,139.00 | 2,380.52 | 2,077.29 | 61.71 | 14.343 |
| 10,000 | 4,638.49 | 5,024.23 | 4,500.61 | 137.89 | 17.108 |

**Finding:** Confirmed O(N) scaling. Both DB fetch and similarity loop grow linearly with dataset size. At 10,000 rows, 17MB of embedding data is transferred to Python on every single request.

### Post-pgvector (Vector(512) + HNSW index)

| dataset_size | avg_ms | p95_ms | db_ms | sim_ms | memory_MB | speedup |
|---|---|---|---|---|---|---|
| 100 | 8.33 | 10.93 | 8.33 | 0.0 | 0.000 | 4× |
| 1,000 | 8.16 | 10.84 | 8.16 | 0.0 | 0.000 | 49× |
| 5,000 | 8.85 | 10.66 | 8.85 | 0.0 | 0.000 | 241× |
| 10,000 | 43.95 | 51.42 | 43.95 | 0.0 | 0.000 | 105× |

**Finding:** Latency is flat from 100 to 5,000 rows. Similarity time is 0ms — computed entirely inside PostgreSQL. Memory per request dropped to ~0MB. The HNSW index eliminates the full table scan entirely.

---

## Layer 2: Sequential HTTP Benchmark (full pipeline, 1 user)

Measures: total end-to-end latency over HTTP including image decode, InsightFace inference, and DB query  
Method: sequential requests with real face image (1280×1280 JPEG), 30 iterations per size  

### Post-pgvector + image resizing (BENCHMARK_MODE=true)

| dataset_size | avg_ms | p95_ms | db_ms | inference_ms |
|---|---|---|---|---|
| 100 | 1,325.36 | 1,479.65 | 2.38 | 1,322.98 |
| 1,000 | 1,577.66 | 1,810.66 | 2.44 | 1,575.21 |
| 5,000 | 1,446.32 | 1,725.12 | 4.54 | 1,441.78 |
| 10,000 | 1,370.29 | 1,656.68 | 2.73 | 1,367.56 |

**Finding:** Total latency is flat across all dataset sizes. DB contributes 2–4ms (~0.2% of total). InsightFace CPU inference is ~1,300ms per request (99.8% of total latency). Dataset size is irrelevant to end-to-end performance — the bottleneck is the model, not the database.

---

## Layer 3: Concurrent HTTP Benchmark

Measures: per-request latency and throughput under simultaneous load  
Method: asyncio.gather fires N requests simultaneously, 5 batches per combination  

### 1 Worker (default)

| dataset_size | c=1 avg_ms | c=5 avg_ms | c=20 avg_ms | c=1 rps | c=5 rps | c=20 rps |
|---|---|---|---|---|---|---|
| 100 | 830 | 5,836 | 21,679 | 1.20 | 0.84 | 0.75 |
| 1,000 | 1,387 | 6,410 | 21,002 | 0.72 | 0.75 | 0.77 |
| 5,000 | 1,370 | 6,505 | 21,113 | 0.73 | 0.77 | 0.76 |
| 10,000 | 1,544 | 6,761 | 22,118 | 0.64 | 0.74 | 0.73 |

DB time across all: 2–10ms regardless of concurrency level.

**Finding:** Requests serialize on the CPU. c=5 latency ≈ 5× single-user, c=20 ≈ 20× single-user. Throughput caps at ~0.75 rps regardless of concurrency — adding more users increases wait time, not throughput. This is GIL-bound CPU queuing on InsightFace inference.

### 4 Workers (UVICORN_WORKERS=4)

| dataset_size | c=1 avg_ms | c=5 avg_ms | c=20 avg_ms | c=1 rps | c=5 rps | c=20 rps |
|---|---|---|---|---|---|---|
| 100 | 1,800 | 5,111 | 11,414 | 0.55 | 0.72 | 1.34 |
| 1,000 | 1,490 | 4,121 | 11,164 | 0.67 | 0.98 | 1.32 |
| 5,000 | 1,516 | 4,546 | 10,268 | 0.66 | 0.88 | 1.44 |
| 10,000 | 1,722 | 4,452 | 10,305 | 0.58 | 0.94 | 1.50 |

### Workers=1 vs Workers=4 Comparison

| concurrency | w=1 avg_ms | w=4 avg_ms | w=1 rps | w=4 rps |
|---|---|---|---|---|
| 1 | 1,387ms | 1,490ms | 0.72 | 0.67 |
| 5 | 6,410ms | 4,121ms | 0.75 | 0.98 |
| 20 | 21,002ms | 11,164ms | 0.77 | 1.32 |

(n=1,000 rows, representative of all dataset sizes)

**Finding:** Single-user latency is slightly worse with 4 workers due to CPU contention between model instances. Under concurrent load, 4 workers roughly halves latency and doubles throughput at c=20 (0.75 → 1.5 rps). Scaling is sub-linear because 4 workers still share the same physical CPU cores.

---

## Summary: What Each Component Costs

| component | time | % of total |
|---|---|---|
| InsightFace inference (CPU) | ~1,300ms | ~99.8% |
| pgvector HNSW query | ~2–8ms | ~0.2% |
| image decode + preprocessing | included in inference_ms | — |
| JWT decode + validation | <1ms | negligible |

---

## Architectural Conclusions

**pgvector completely solved the DB bottleneck.** Pre-migration, latency grew O(N) from 34ms to 4,638ms. Post-migration, DB time is flat at 2–8ms regardless of dataset size, concurrency, or worker count. 241× faster at 5,000 rows. This is a solved problem.

**InsightFace CPU inference is the only remaining bottleneck.** At ~1,300ms per request it dominates everything else. No amount of DB optimization, worker scaling, or architectural work changes this number — only the hardware or model does.

**ThreadPoolExecutor correctly unblocks the event loop.** Inference runs in a thread pool so FastAPI can accept new connections while inference is in progress. Without this, async concurrency provides no benefit.

**4 workers doubles throughput under load.** The cost is ~2GB additional RAM (one buffalo_l model instance per worker at ~500MB each) and slight single-user latency regression from CPU contention.

---

## Path to Further Improvement

| option | expected inference time | cost |
|---|---|---|
| Current (CPU, buffalo_l) | ~1,300ms | baseline |
| Resize input to 640×640 | ~800–1,000ms | minimal |
| buffalo_sc model (CPU) | ~400–600ms | lower accuracy |
| GPU inference | ~10–50ms | hardware cost |
| Multiple workers (N) | ~1,300ms / N throughput | N × 500MB RAM |