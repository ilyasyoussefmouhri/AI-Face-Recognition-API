# benchmarks/run_benchmark.py
#
# Full-stack HTTP benchmark for the /recognize endpoint.
# Measures total latency, inference time, and DB time via response headers.
# Requires: BENCHMARK_MODE=true in docker-compose, real face image.
#
# Usage:
#   BENCHMARK_FACE_IMAGE=/path/to/face.jpg \
#   BENCHMARK_LABEL=post_pgvector \
#   PYTHONPATH=$(pwd) python benchmarks/run_benchmark.py

import sys
import os
import time
import statistics
import csv
from datetime import datetime, timezone

import httpx

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    BASE_URL, RECOGNIZE_ENDPOINT,
    DATASET_SIZES, RESULTS_DIR,
)
from auth_helper import register_benchmark_user, get_benchmark_token
from db_seeder import seed_embeddings

RESULTS_FILE = os.path.join(RESULTS_DIR, "results_http.csv")
ITERATIONS   = 30
TIMEOUT      = 60.0  # InsightFace on CPU can be slow

COLUMNS = [
    "timestamp", "run_label", "dataset_size",
    "requests",
    "avg_latency_ms", "p50_latency_ms", "p95_latency_ms", "p99_latency_ms",
    "min_latency_ms", "max_latency_ms", "std_dev_ms",
    "avg_db_ms", "avg_inference_ms",
    "insert_time_s",
]


def _load_image() -> bytes:
    path = os.getenv("BENCHMARK_FACE_IMAGE")
    if not path or not os.path.isfile(path):
        raise RuntimeError(
            "Set BENCHMARK_FACE_IMAGE=/path/to/face.jpg\n"
            "A real face image is required for HTTP benchmarking — "
            "the fake JPEG will 422 on InsightFace."
        )
    with open(path, "rb") as f:
        return f.read()


def _parse_header(resp: httpx.Response, name: str) -> float | None:
    val = resp.headers.get(name)
    try:
        return float(val) if val else None
    except ValueError:
        return None


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(int(p / 100 * len(sorted_vals)), len(sorted_vals) - 1)
    return sorted_vals[idx]


def run_sequential(token: str, image_bytes: bytes, n_requests: int) -> list[dict]:
    """Single-threaded sequential requests. Measures per-request cost with no contention."""
    results = []
    headers = {"Authorization": f"Bearer {token}"}

    with httpx.Client(timeout=TIMEOUT) as client:
        print("  Warming up (3 requests)...")
        for _ in range(3):
            client.post(
                f"{BASE_URL}{RECOGNIZE_ENDPOINT}",
                headers=headers,
                files={"file": ("face.jpg", image_bytes, "image/jpeg")},
            )

        print(f"  Running {n_requests} measured requests...")
        for i in range(n_requests):
            t0 = time.perf_counter()
            resp = client.post(
                f"{BASE_URL}{RECOGNIZE_ENDPOINT}",
                headers=headers,
                files={"file": ("face.jpg", image_bytes, "image/jpeg")},
            )
            t1 = time.perf_counter()

            if resp.status_code == 429:
                print("  [rate limited] sleeping 15s...")
                time.sleep(15)
                continue

            latency_ms = (t1 - t0) * 1000
            results.append({
                "latency_ms":    latency_ms,
                "db_ms":         _parse_header(resp, "X-DB-Time-Ms"),
                "total_time_ms": _parse_header(resp, "X-Total-Time-Ms"),
                "status":        resp.status_code,
            })

            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{n_requests}  "
                      f"latency={latency_ms:.0f}ms  "
                      f"status={resp.status_code}")

    return results


def summarise(results: list[dict], dataset_size: int, insert_time: float) -> dict:
    latencies = sorted(r["latency_ms"] for r in results)
    db_times  = [r["db_ms"] for r in results if r["db_ms"] is not None]

    # Inference time = total client latency minus DB time
    # This isolates ML inference + image decode + everything else
    inference_times = []
    for r in results:
        if r["db_ms"] is not None:
            inference_times.append(r["latency_ms"] - r["db_ms"])

    summary = {
        "dataset_size":   dataset_size,
        "requests":       len(results),
        "avg_latency_ms": round(statistics.mean(latencies), 2),
        "p50_latency_ms": round(_percentile(latencies, 50), 2),
        "p95_latency_ms": round(_percentile(latencies, 95), 2),
        "p99_latency_ms": round(_percentile(latencies, 99), 2),
        "min_latency_ms": round(min(latencies), 2),
        "max_latency_ms": round(max(latencies), 2),
        "std_dev_ms":     round(statistics.stdev(latencies) if len(latencies) > 1 else 0, 2),
        "avg_db_ms":      round(statistics.mean(db_times), 2) if db_times else None,
        "avg_inference_ms": round(statistics.mean(inference_times), 2) if inference_times else None,
        "insert_time_s":  round(insert_time, 2),
    }

    print(f"\n  avg={summary['avg_latency_ms']}ms  "
          f"p95={summary['p95_latency_ms']}ms  "
          f"p99={summary['p99_latency_ms']}ms")
    if db_times:
        print(f"  avg_db={summary['avg_db_ms']}ms  "
              f"avg_inference={summary['avg_inference_ms']}ms")
    else:
        print("  (no X-DB-Time-Ms headers — is BENCHMARK_MODE=true?)")

    return summary


def write_result(result: dict) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    write_header = not os.path.exists(RESULTS_FILE)
    result.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    result.setdefault("run_label", os.getenv("BENCHMARK_LABEL", "post_pgvector"))
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({k: ("" if result.get(k) is None else result.get(k, "")) for k in COLUMNS})
    print(f"  Result written to {RESULTS_FILE}")


def main():
    print("=" * 55)
    print("  HTTP End-to-End Benchmark")
    print("  /recognize — full pipeline including ML inference")
    print("=" * 55)

    image_bytes = _load_image()
    print(f"Face image loaded: {len(image_bytes):,} bytes")

    register_benchmark_user()
    token = get_benchmark_token()

    for n in DATASET_SIZES:
        print(f"\n{'=' * 55}")
        print(f"  Dataset size: {n:,}")
        print(f"{'=' * 55}")

        insert_time = seed_embeddings(n)
        results = run_sequential(token, image_bytes, ITERATIONS)

        if not results:
            print("  No results collected — all requests may have been rate limited")
            continue

        summary = summarise(results, n, insert_time)
        write_result(summary)
        time.sleep(2)

    print(f"\nAll done. Results: {RESULTS_FILE}")


if __name__ == "__main__":
    main()