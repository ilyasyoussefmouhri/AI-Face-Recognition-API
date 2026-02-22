# benchmarks/csv_writer.py
#
# Writes benchmark results to CSV.  Each call to append_result() adds one row.
# The file is created with headers on first write; subsequent runs append rows,
# so you can compare across different runs (pre-pgvector vs post-pgvector, etc.)

import csv
import os
from datetime import datetime, timezone

# Columns match what run_benchmark.py and load_test.py produce.
# "None" values are written as empty strings in CSV (safe for spreadsheets).
LATENCY_COLUMNS = [
    "timestamp",
    "run_label",           # e.g. "pre_pgvector" — set via env BENCHMARK_LABEL
    "dataset_size",
    "iterations",
    "avg_latency_ms",
    "p50_latency_ms",
    "p95_latency_ms",
    "p99_latency_ms",
    "min_latency_ms",
    "max_latency_ms",
    "std_dev_ms",
    "avg_db_time_ms",          # From X-DB-Time-Ms header (requires BENCHMARK_MODE=true)
    "avg_similarity_time_ms",  # From X-Similarity-Time-Ms header
    "avg_memory_delta_mb",
    "avg_cpu_percent",
    "insert_time_s",           # How long it took to seed this dataset size
]

LOAD_COLUMNS = [
    "timestamp",
    "run_label",
    "dataset_size",
    "num_users",
    "avg_latency_ms",
    "p95_latency_ms",
    "rps",
    "failure_rate",
    "num_requests",
    "num_failures",
]


def _ensure_file(filepath: str, columns: list[str]) -> None:
    """Create the CSV file with headers if it doesn't exist yet."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if not os.path.exists(filepath):
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
        print(f"Created results file: {filepath}")


def append_result(filepath: str, result: dict, columns: list[str]) -> None:
    """
    Append one result row to the CSV.
    Extra keys in `result` are silently ignored (extrasaction='ignore').
    Missing keys are written as empty strings.
    """
    _ensure_file(filepath, columns)

    # Stamp with UTC ISO time and run label from env
    result.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    result.setdefault("run_label", os.getenv("BENCHMARK_LABEL", "pre_pgvector"))

    # Convert None → "" for clean CSV output
    cleaned = {k: ("" if result.get(k) is None else result.get(k, "")) for k in columns}

    with open(filepath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writerow(cleaned)


def append_latency_result(result: dict, filepath: str | None = None) -> None:
    """Convenience wrapper for latency benchmark results."""
    from config import RESULTS_FILE
    append_result(filepath or RESULTS_FILE, result, LATENCY_COLUMNS)


def append_load_result(result: dict, filepath: str | None = None) -> None:
    """Convenience wrapper for load test results."""
    from config import LOAD_RESULTS_FILE
    append_result(filepath or LOAD_RESULTS_FILE, result, LOAD_COLUMNS)
