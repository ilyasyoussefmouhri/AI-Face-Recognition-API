# benchmarks/run_benchmark.py
#
# Main benchmark runner.  For each dataset size it:
#   1. Seeds the DB with synthetic rows (via db_seeder.py)
#   2. Warms up the endpoint (3 requests, unmeasured)
#   3. Runs ITERATIONS_PER_SIZE measured recognition requests
#   4. Collects latency, optional per-phase times, memory delta, CPU
#   5. Writes one summary row to results/results.csv
#
# Prerequisites (run once before first benchmark):
#   python benchmarks/auth_helper.py   — registers + verifies the bench user
#
# Usage:
#   # From project root:
#   PYTHONPATH=/src python benchmarks/run_benchmark.py
#
#   # Or with Docker Compose (see docker-compose.benchmark.yml):
#   docker-compose -f docker-compose.yaml -f docker-compose.benchmark.yml run benchmark
#
# Important: /recognize accepts a file upload (multipart/form-data), NOT a JSON body.
# We send a tiny valid JPEG so the endpoint doesn't 422 on image validation,
# but the image contains no real face — the embedder will raise NoFaceDetectedError.
#
# ── A note on what we're actually measuring ──────────────────────────────────
# The recognise endpoint runs:
#   validate_image → decode_image → load_image → embedder.embed → DB query → similarity loop
#
# For a pure DB+similarity benchmark we want to bypass the ML steps.
# The cleanest approach without modifying your production code is to measure
# TOTAL end-to-end latency and use the X-DB-Time-Ms / X-Similarity-Time-Ms
# headers (set by benchmark_timing.py + the patched recognition.py) to
# decompose the time budget.
#
# If you want to benchmark ONLY the DB+similarity phase, you can call the
# service layer directly (see run_benchmark_direct.py) — but end-to-end
# latency is the most honest number for comparing against pgvector.

import sys
import os
import io
import time
import statistics
import tracemalloc
import psutil
import requests

# Make sure config and helpers are importable when running from project root
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    BASE_URL, RECOGNIZE_ENDPOINT,
    DATASET_SIZES, ITERATIONS_PER_SIZE, REQUEST_GAP_SECONDS,
    RESULTS_FILE,
)
from db_seeder import seed_embeddings, count_benchmark_rows
from auth_helper import register_benchmark_user, get_benchmark_token
from csv_writer import append_latency_result, LATENCY_COLUMNS


# ── Minimal valid JPEG bytes ──────────────────────────────────────────────────
# 1×1 pixel white JPEG.  Passes your signature check (starts with FF D8 FF)
# and PIL can decode it.  InsightFace will raise NoFaceDetectedError because
# there's no face — recognition.py returns 422.
# For a TRUE end-to-end latency test you need a real face image; see comments below.
#
# For the purpose of benchmarking DB query time + similarity loop, you have two options:
#
#   OPTION A (recommended for beginners): supply a real face image file and the
#   benchmark will hit the full pipeline.  Set BENCHMARK_FACE_IMAGE env var to
#   the path of a real face JPEG.
#
#   OPTION B: benchmark DB+similarity directly via run_benchmark_direct.py
#   which bypasses the HTTP layer entirely and calls the service functions
#   with a synthetic numpy embedding.
MINIMAL_JPEG = bytes([
    0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,
    0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43,
    0x00, 0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08, 0x07, 0x07, 0x07, 0x09,
    0x09, 0x08, 0x0A, 0x0C, 0x14, 0x0D, 0x0C, 0x0B, 0x0B, 0x0C, 0x19, 0x12,
    0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E, 0x1D, 0x1A, 0x1C, 0x1C, 0x20,
    0x24, 0x2E, 0x27, 0x20, 0x22, 0x2C, 0x23, 0x1C, 0x1C, 0x28, 0x37, 0x29,
    0x2C, 0x30, 0x31, 0x34, 0x34, 0x34, 0x1F, 0x27, 0x39, 0x3D, 0x38, 0x32,
    0x3C, 0x2E, 0x33, 0x34, 0x32, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01,
    0x00, 0x01, 0x01, 0x01, 0x11, 0x00, 0xFF, 0xC4, 0x00, 0x1F, 0x00, 0x00,
    0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
    0x09, 0x0A, 0x0B, 0xFF, 0xC4, 0x00, 0xB5, 0x10, 0x00, 0x02, 0x01, 0x03,
    0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00, 0x01, 0x7D,
    0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3F, 0x00, 0xFB, 0x3F,
    0xFF, 0xD9,
])


def _load_face_image() -> bytes:
    """
    Load the image bytes to send with each recognition request.

    If BENCHMARK_FACE_IMAGE env var points to a real face JPEG, we use that
    (recommended — gives you true end-to-end timing including ML inference).
    Otherwise falls back to the minimal JPEG which will cause a 422 from
    InsightFace but still exercises the auth + file-size check + DB layers.
    """
    path = os.getenv("BENCHMARK_FACE_IMAGE")
    if path and os.path.isfile(path):
        with open(path, "rb") as f:
            data = f.read()
        print(f"Using real face image: {path} ({len(data):,} bytes)")
        return data
    else:
        print(
            "WARNING: No BENCHMARK_FACE_IMAGE set.  Using minimal JPEG.\n"
            "         InsightFace will 422 (no face detected).\n"
            "         Set BENCHMARK_FACE_IMAGE=/path/to/face.jpg for full pipeline timing."
        )
        return MINIMAL_JPEG


def _make_request(session: requests.Session, image_bytes: bytes, token: str) -> dict:
    """
    Send one /recognize request and return timing + header info.

    Uses a persistent requests.Session for connection reuse (more realistic
    than opening a new TCP connection per request).
    """
    headers = {"Authorization": f"Bearer {token}"}
    files = {"file": ("bench.jpg", io.BytesIO(image_bytes), "image/jpeg")}

    t0 = time.perf_counter()
    resp = session.post(
        f"{BASE_URL}{RECOGNIZE_ENDPOINT}",
        headers=headers,
        files=files,
        timeout=30,
    )
    t1 = time.perf_counter()

    # We accept both 200 (match/no-match) and 422 (no face detected with fake image).
    # 429 = rate limited — we sleep and retry once.
    if resp.status_code == 429:
        print("  [rate limited] sleeping 15s then retrying...")
        time.sleep(15)
        return _make_request(session, image_bytes, token)

    total_ms = (t1 - t0) * 1000

    return {
        "total_latency_ms": total_ms,
        "status_code": resp.status_code,
        "db_time_ms": _parse_header(resp, "X-DB-Time-Ms"),
        "similarity_time_ms": _parse_header(resp, "X-Similarity-Time-Ms"),
    }


def _parse_header(resp: requests.Response, name: str) -> float | None:
    val = resp.headers.get(name)
    try:
        return float(val) if val else None
    except ValueError:
        return None


def _measure_with_resources(
    session: requests.Session, image_bytes: bytes, token: str
) -> dict:
    """
    Wrap _make_request with memory and CPU measurements.

    tracemalloc: measures Python-heap allocations during the call.
    psutil: measures process RSS delta and CPU (rough but useful for trends).
    """
    proc = psutil.Process(os.getpid())
    mem_before = proc.memory_info().rss / (1024 * 1024)
    _ = proc.cpu_percent(interval=None)  # prime the counter

    tracemalloc.start()
    result = _make_request(session, image_bytes, token)
    _current, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    mem_after = proc.memory_info().rss / (1024 * 1024)
    cpu = proc.cpu_percent(interval=None)

    result["memory_delta_mb"] = mem_after - mem_before
    result["peak_alloc_mb"] = peak_bytes / (1024 * 1024)
    result["cpu_percent"] = cpu
    return result


def _percentile(sorted_values: list[float], p: float) -> float:
    """Return the p-th percentile (0-100) of a pre-sorted list."""
    if not sorted_values:
        return 0.0
    idx = int(p / 100 * len(sorted_values))
    idx = min(idx, len(sorted_values) - 1)
    return sorted_values[idx]


def run_for_size(n: int, token: str, image_bytes: bytes) -> dict:
    """
    Full benchmark run for one dataset size.
    Returns a result dict ready for csv_writer.
    """
    print(f"\n{'=' * 55}")
    print(f"  Dataset size: {n:,} face rows")
    print(f"{'=' * 55}")

    insert_time = seed_embeddings(n)

    counts = count_benchmark_rows()
    print(f"DB sanity check: {counts}")

    # Warmup — 3 unmeasured requests to fill connection pools
    print("Warming up (3 requests)...")
    with requests.Session() as session:
        for _ in range(3):
            _make_request(session, image_bytes, token)
            time.sleep(0.5)

    print(f"Running {ITERATIONS_PER_SIZE} measured iterations...")

    latencies, db_times, sim_times, mem_deltas, cpus = [], [], [], [], []

    with requests.Session() as session:
        for i in range(ITERATIONS_PER_SIZE):
            result = _measure_with_resources(session, image_bytes, token)

            latencies.append(result["total_latency_ms"])
            mem_deltas.append(result["memory_delta_mb"])
            cpus.append(result["cpu_percent"])

            if result["db_time_ms"] is not None:
                db_times.append(result["db_time_ms"])
            if result["similarity_time_ms"] is not None:
                sim_times.append(result["similarity_time_ms"])

            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{ITERATIONS_PER_SIZE}  "
                      f"last={result['total_latency_ms']:.1f}ms  "
                      f"status={result['status_code']}")

            time.sleep(REQUEST_GAP_SECONDS)

    latencies.sort()
    summary = {
        "dataset_size": n,
        "iterations": ITERATIONS_PER_SIZE,
        "avg_latency_ms": round(statistics.mean(latencies), 2),
        "p50_latency_ms": round(_percentile(latencies, 50), 2),
        "p95_latency_ms": round(_percentile(latencies, 95), 2),
        "p99_latency_ms": round(_percentile(latencies, 99), 2),
        "min_latency_ms": round(min(latencies), 2),
        "max_latency_ms": round(max(latencies), 2),
        "std_dev_ms": round(statistics.stdev(latencies) if len(latencies) > 1 else 0, 2),
        "avg_db_time_ms": round(statistics.mean(db_times), 2) if db_times else None,
        "avg_similarity_time_ms": round(statistics.mean(sim_times), 2) if sim_times else None,
        "avg_memory_delta_mb": round(statistics.mean(mem_deltas), 3),
        "avg_cpu_percent": round(statistics.mean(cpus), 1),
        "insert_time_s": round(insert_time, 2),
    }

    print(f"\n  avg={summary['avg_latency_ms']}ms  "
          f"p95={summary['p95_latency_ms']}ms  "
          f"p99={summary['p99_latency_ms']}ms  "
          f"mem_delta={summary['avg_memory_delta_mb']}MB")
    if db_times:
        print(f"  avg_db={summary['avg_db_time_ms']}ms  "
              f"avg_sim={summary['avg_similarity_time_ms']}ms")
    else:
        print("  (No X-DB-Time-Ms headers — is BENCHMARK_MODE=true?)")

    return summary


def main():
    print("=" * 55)
    print("  Face Recognition Baseline Benchmark")
    print("  (pre-pgvector)")
    print("=" * 55)

    # Step 1: ensure the benchmark user exists and get a token
    register_benchmark_user()
    token = get_benchmark_token()

    # Step 2: load the image to send
    image_bytes = _load_face_image()

    # Step 3: run for each dataset size
    for n in DATASET_SIZES:
        result = run_for_size(n, token, image_bytes)
        append_latency_result(result)
        print(f"  Result written to {RESULTS_FILE}")
        time.sleep(3)  # brief pause to let DB settle between sizes

    print(f"\nAll done. Results: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
