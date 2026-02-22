# benchmarks/run_benchmark_direct.py
#
# Alternative benchmark that bypasses the HTTP layer entirely and calls
# your service functions directly with a synthetic numpy embedding.
#
# Why this exists:
#   run_benchmark.py measures end-to-end HTTP latency, which includes:
#     - Network stack overhead
#     - FastAPI routing + middleware
#     - JWT decode + DB lookup for auth
#     - Image validation + PIL decode
#     - InsightFace ML inference (~100-500ms on CPU — dominates everything else)
#
#   If you want to isolate DB query time + similarity loop time specifically,
#   this script is cleaner: it imports SessionLocal and calls the DB directly,
#   then runs your matcher.similarity() loop.
#
# This is NOT a replacement for run_benchmark.py — use both:
#   run_benchmark.py       → honest end-to-end numbers (what users experience)
#   run_benchmark_direct.py → isolated DB+similarity numbers (what pgvector replaces)
#
# Usage (from project root):
#   PYTHONPATH=/src python benchmarks/run_benchmark_direct.py

import sys
import os
import time
import statistics
import tracemalloc
import psutil

sys.path.insert(0, os.path.dirname(__file__))
# Also ensure the app package is importable (PYTHONPATH=/src should handle this)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import DATASET_SIZES, ITERATIONS_PER_SIZE, RESULTS_FILE
from db_seeder import seed_embeddings, count_benchmark_rows
from data_gen import generate_query_embedding
from csv_writer import append_latency_result

import numpy as np
from app.db.session import SessionLocal
from app.db.models import Face, User
from app.models.matcher import InsightFaceMatcher


def _run_db_and_similarity(query_embedding: np.ndarray, matcher: InsightFaceMatcher) -> dict:
    """
    Open a DB session, fetch all faces, run the similarity loop, close session.
    Returns timing for each phase in milliseconds.
    """
    db = SessionLocal()
    try:
        # ── DB fetch phase ─────────────────────────────────────────────────
        t0 = time.perf_counter()
        faces = db.query(Face).join(User).all()
        t1 = time.perf_counter()
        db_ms = (t1 - t0) * 1000

        if not faces:
            return {"db_time_ms": db_ms, "similarity_time_ms": 0.0, "n_faces": 0}

        # ── Similarity loop phase ──────────────────────────────────────────
        # This mirrors recognition.py exactly:
        #   for face in faces:
        #       stored = np.array(face.embedding, dtype=np.float32)
        #       similarity = matcher.similarity(query_embedding, stored)
        t2 = time.perf_counter()
        best_similarity = -1.0
        for face in faces:
            stored = np.array(face.embedding, dtype=np.float32)
            sim = matcher.similarity(query_embedding, stored)
            if sim > best_similarity:
                best_similarity = sim
        t3 = time.perf_counter()
        sim_ms = (t3 - t2) * 1000

        return {
            "db_time_ms": db_ms,
            "similarity_time_ms": sim_ms,
            "total_ms": db_ms + sim_ms,
            "n_faces": len(faces),
        }
    finally:
        db.close()


def _measure_with_resources(
    query_embedding: np.ndarray, matcher: InsightFaceMatcher
) -> dict:
    proc = psutil.Process(os.getpid())
    mem_before = proc.memory_info().rss / (1024 * 1024)
    _ = proc.cpu_percent(interval=None)

    tracemalloc.start()
    result = _run_db_and_similarity(query_embedding, matcher)
    _cur, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    mem_after = proc.memory_info().rss / (1024 * 1024)
    result["memory_delta_mb"] = mem_after - mem_before
    result["peak_alloc_mb"] = peak_bytes / (1024 * 1024)
    result["cpu_percent"] = proc.cpu_percent(interval=None)
    return result


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(int(p / 100 * len(sorted_vals)), len(sorted_vals) - 1)
    return sorted_vals[idx]


def run_for_size(n: int, query: np.ndarray, matcher: InsightFaceMatcher) -> dict:
    print(f"\n{'=' * 55}")
    print(f"  Direct benchmark — dataset size: {n:,}")
    print(f"{'=' * 55}")

    insert_time = seed_embeddings(n)
    counts = count_benchmark_rows()
    print(f"DB sanity check: {counts}")

    # Warmup
    print("Warming up (3 calls)...")
    for _ in range(3):
        _run_db_and_similarity(query, matcher)

    print(f"Running {ITERATIONS_PER_SIZE} measured iterations...")
    db_times, sim_times, total_times, mem_deltas, cpus = [], [], [], [], []

    for i in range(ITERATIONS_PER_SIZE):
        r = _measure_with_resources(query, matcher)
        db_times.append(r["db_time_ms"])
        sim_times.append(r["similarity_time_ms"])
        total_times.append(r["total_ms"])
        mem_deltas.append(r["memory_delta_mb"])
        cpus.append(r["cpu_percent"])

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{ITERATIONS_PER_SIZE}  "
                  f"db={r['db_time_ms']:.1f}ms  "
                  f"sim={r['similarity_time_ms']:.1f}ms")

    total_times.sort()
    summary = {
        "dataset_size": n,
        "iterations": ITERATIONS_PER_SIZE,
        "avg_latency_ms": round(statistics.mean(total_times), 2),
        "p50_latency_ms": round(_percentile(total_times, 50), 2),
        "p95_latency_ms": round(_percentile(total_times, 95), 2),
        "p99_latency_ms": round(_percentile(total_times, 99), 2),
        "min_latency_ms": round(min(total_times), 2),
        "max_latency_ms": round(max(total_times), 2),
        "std_dev_ms": round(statistics.stdev(total_times) if len(total_times) > 1 else 0, 2),
        "avg_db_time_ms": round(statistics.mean(db_times), 2),
        "avg_similarity_time_ms": round(statistics.mean(sim_times), 2),
        "avg_memory_delta_mb": round(statistics.mean(mem_deltas), 3),
        "avg_cpu_percent": round(statistics.mean(cpus), 1),
        "insert_time_s": round(insert_time, 2),
    }

    print(f"\n  avg_total={summary['avg_latency_ms']}ms  "
          f"avg_db={summary['avg_db_time_ms']}ms  "
          f"avg_sim={summary['avg_similarity_time_ms']}ms")
    print(f"  p95={summary['p95_latency_ms']}ms  "
          f"mem_delta={summary['avg_memory_delta_mb']}MB")

    return summary


def main():
    print("=" * 55)
    print("  Direct DB+Similarity Benchmark (no HTTP)")
    print("  This measures what pgvector will replace.")
    print("=" * 55)

    query_list = generate_query_embedding()
    query = np.array(query_list, dtype=np.float32)
    matcher = InsightFaceMatcher(threshold=0.7)

    # Save to a separate file so it doesn't mix with HTTP benchmark results
    direct_results_file = RESULTS_FILE.replace("results.csv", "results_direct.csv")

    for n in DATASET_SIZES:
        result = run_for_size(n, query, matcher)
        from csv_writer import append_result, LATENCY_COLUMNS
        append_result(direct_results_file, result, LATENCY_COLUMNS)
        print(f"  Result written to {direct_results_file}")
        time.sleep(2)

    print(f"\nAll done. Results: {direct_results_file}")


if __name__ == "__main__":
    main()
