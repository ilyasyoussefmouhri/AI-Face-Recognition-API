# benchmarks/run_benchmark_concurrent.py
#
# Concurrent HTTP benchmark for the /recognize endpoint.
# Fires N simultaneous requests and measures per-request latency under load.
#
# What this measures:
#   How latency degrades as concurrent users increase.
#   With CPU-bound InsightFace inference, requests queue behind each other —
#   this makes that queuing visible and quantified.
#
# Usage:
#   BENCHMARK_FACE_IMAGE=/path/to/face.jpg \
#   BENCHMARK_LABEL=concurrency \
#   PYTHONPATH=$(pwd) python benchmarks/run_benchmark_concurrent.py

import sys
import os
import asyncio
import time
import statistics
import csv
from datetime import datetime, timezone

import httpx

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    BASE_URL, RECOGNIZE_ENDPOINT,
    DATASET_SIZES, RESULTS_DIR,
    BENCHMARK_HTTP_TIMEOUT,
)
from auth_helper import register_benchmark_user, get_benchmark_token
from db_seeder import seed_embeddings

RESULTS_FILE = os.path.join(RESULTS_DIR, "results_concurrent.csv")

# Concurrency levels to test
CONCURRENCY_LEVELS = [1, 5, 20]

# How many batches to run per (dataset_size, concurrency) combination.
# Each batch fires `concurrency` simultaneous requests.
# Total requests = BATCHES × concurrency.
# 5 batches gives enough data points without taking forever at high concurrency.
BATCHES = 5

COLUMNS = [
    "timestamp", "run_label",
    "dataset_size", "concurrency", "batches", "total_requests",
    "avg_latency_ms", "p50_latency_ms", "p95_latency_ms", "p99_latency_ms",
    "min_latency_ms", "max_latency_ms", "std_dev_ms",
    "avg_db_ms", "avg_inference_ms",
    "throughput_rps",
    "timeout_count", "error_count",
]


def _load_image() -> bytes:
    path = os.getenv("BENCHMARK_FACE_IMAGE")
    if not path or not os.path.isfile(path):
        raise RuntimeError(
            "Set BENCHMARK_FACE_IMAGE=/path/to/face.jpg\n"
            "A real face image is required — fake JPEG will 422 on InsightFace."
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


async def _single_request(
    client: httpx.AsyncClient,
    token: str,
    image_bytes: bytes,
) -> dict:
    """
    Fire one recognition request and return timing + header data.
    Never raises — errors are captured in the result dict so a single
    failed request doesn't abort the whole batch.
    """
    headers = {"Authorization": f"Bearer {token}"}
    t0 = time.perf_counter()
    try:
        resp = await client.post(
            f"{BASE_URL}{RECOGNIZE_ENDPOINT}",
            headers=headers,
            files={"file": ("face.jpg", image_bytes, "image/jpeg")},
        )
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000

        return {
            "latency_ms":   latency_ms,
            "db_ms":        _parse_header(resp, "X-DB-Time-Ms"),
            "status":       resp.status_code,
            "timeout":      False,
            "error":        False,
        }

    except httpx.TimeoutException:
        t1 = time.perf_counter()
        return {
            "latency_ms":   (t1 - t0) * 1000,
            "db_ms":        None,
            "status":       None,
            "timeout":      True,
            "error":        False,
        }

    except Exception as e:
        t1 = time.perf_counter()
        return {
            "latency_ms":   (t1 - t0) * 1000,
            "db_ms":        None,
            "status":       None,
            "timeout":      False,
            "error":        True,
        }


async def run_concurrent_batch(
    token: str,
    image_bytes: bytes,
    concurrency: int,
) -> list[dict]:
    """
    Fire `concurrency` requests simultaneously and wait for all to complete.
    Returns one result dict per request.
    """
    async with httpx.AsyncClient(timeout=BENCHMARK_HTTP_TIMEOUT) as client:
        tasks = [
            _single_request(client, token, image_bytes)
            for _ in range(concurrency)
        ]
        # asyncio.gather fires all tasks simultaneously
        results = await asyncio.gather(*tasks)
    return list(results)


async def run_for_combination(
    dataset_size: int,
    concurrency: int,
    token: str,
    image_bytes: bytes,
) -> dict:
    """
    Run BATCHES batches of `concurrency` concurrent requests against
    a DB seeded with `dataset_size` rows.
    """
    print(f"\n  concurrency={concurrency}  dataset_size={dataset_size:,}")
    print(f"  {BATCHES} batches × {concurrency} simultaneous requests "
          f"= {BATCHES * concurrency} total requests")

    # Warmup — 1 sequential request, unmeasured
    async with httpx.AsyncClient(timeout=BENCHMARK_HTTP_TIMEOUT) as client:
        await _single_request(client, token, image_bytes)

    all_results = []
    batch_start = time.perf_counter()

    for b in range(BATCHES):
        batch_results = await run_concurrent_batch(token, image_bytes, concurrency)
        all_results.extend(batch_results)

        successes = sum(1 for r in batch_results if not r["timeout"] and not r["error"])
        timeouts  = sum(1 for r in batch_results if r["timeout"])
        avg_ms    = statistics.mean(r["latency_ms"] for r in batch_results)
        print(f"  batch {b+1}/{BATCHES}  "
              f"avg={avg_ms:.0f}ms  "
              f"ok={successes}  timeouts={timeouts}")

    total_wall_time = time.perf_counter() - batch_start

    # Separate successful results for statistics
    good = [r for r in all_results if not r["timeout"] and not r["error"]]
    timeouts  = sum(1 for r in all_results if r["timeout"])
    errors    = sum(1 for r in all_results if r["error"])

    if not good:
        print("  WARNING: all requests failed or timed out")
        return {
            "dataset_size": dataset_size, "concurrency": concurrency,
            "batches": BATCHES, "total_requests": len(all_results),
            "timeout_count": timeouts, "error_count": errors,
        }

    latencies = sorted(r["latency_ms"] for r in good)
    db_times  = [r["db_ms"] for r in good if r["db_ms"] is not None]
    inference_times = [
        r["latency_ms"] - r["db_ms"]
        for r in good if r["db_ms"] is not None
    ]

    # Throughput = successful requests / total wall time for all batches
    throughput_rps = round(len(good) / total_wall_time, 3)

    summary = {
        "dataset_size":     dataset_size,
        "concurrency":      concurrency,
        "batches":          BATCHES,
        "total_requests":   len(all_results),
        "avg_latency_ms":   round(statistics.mean(latencies), 2),
        "p50_latency_ms":   round(_percentile(latencies, 50), 2),
        "p95_latency_ms":   round(_percentile(latencies, 95), 2),
        "p99_latency_ms":   round(_percentile(latencies, 99), 2),
        "min_latency_ms":   round(min(latencies), 2),
        "max_latency_ms":   round(max(latencies), 2),
        "std_dev_ms":       round(statistics.stdev(latencies) if len(latencies) > 1 else 0, 2),
        "avg_db_ms":        round(statistics.mean(db_times), 2) if db_times else None,
        "avg_inference_ms": round(statistics.mean(inference_times), 2) if inference_times else None,
        "throughput_rps":   throughput_rps,
        "timeout_count":    timeouts,
        "error_count":      errors,
    }

    print(f"  → avg={summary['avg_latency_ms']}ms  "
          f"p95={summary['p95_latency_ms']}ms  "
          f"rps={throughput_rps}  "
          f"timeouts={timeouts}")

    return summary


def write_result(result: dict) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    write_header = not os.path.exists(RESULTS_FILE)
    result.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    result.setdefault("run_label", os.getenv("BENCHMARK_LABEL", "concurrency"))
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({
            k: ("" if result.get(k) is None else result.get(k, ""))
            for k in COLUMNS
        })


async def main():
    print("=" * 55)
    print("  Concurrency Benchmark")
    print(f"  levels={CONCURRENCY_LEVELS}  batches={BATCHES}")
    print("=" * 55)

    image_bytes = _load_image()
    print(f"Face image: {len(image_bytes):,} bytes")

    register_benchmark_user()
    token = get_benchmark_token()

    for dataset_size in DATASET_SIZES:
        print(f"\n{'=' * 55}")
        print(f"  Dataset size: {dataset_size:,}")
        print(f"{'=' * 55}")

        seed_embeddings(dataset_size)

        for concurrency in CONCURRENCY_LEVELS:
            result = await run_for_combination(
                dataset_size, concurrency, token, image_bytes
            )
            write_result(result)
            print(f"  Written to {RESULTS_FILE}")
            # Brief pause between concurrency levels to let the server settle
            await asyncio.sleep(2)

    print(f"\nAll done. Results: {RESULTS_FILE}")


if __name__ == "__main__":
    asyncio.run(main())