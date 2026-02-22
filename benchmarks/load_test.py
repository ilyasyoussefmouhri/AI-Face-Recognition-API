# benchmarks/load_test.py
#
# Locust load test for the /recognize endpoint.
#
# ── Rate-limit reality check ─────────────────────────────────────────────────
# Your /recognize route is capped at 20 requests/minute PER IP (SlowAPI).
# Locust spawns all virtual users from the SAME process/IP by default, so
# all traffic shares one rate-limit bucket.
#
# Safe configuration: 1 user, wait_time=between(3, 5)
#   → ~12-20 req/min per IP → stays just under the 20/min limit
#
# To simulate real concurrency (multiple users from different IPs):
#   → Run multiple Locust workers from different machines or Docker containers
#   → Or disable rate limiting in your .env for the benchmark environment
#
# ── Usage ────────────────────────────────────────────────────────────────────
# Single-user safe run:
#   BENCHMARK_DATASET_SIZE=1000 \
#   locust -f benchmarks/load_test.py --headless -u 1 -r 1 --run-time 60s \
#          --host http://localhost:8000
#
# With a real face image (required for full pipeline):
#   BENCHMARK_FACE_IMAGE=/path/to/face.jpg \
#   BENCHMARK_DATASET_SIZE=5000 \
#   locust -f benchmarks/load_test.py --headless -u 1 -r 1 --run-time 60s \
#          --host http://localhost:8000
#
# Results are saved to benchmarks/results/results_load.csv when the test ends.
#
# ── Iterating over dataset sizes ─────────────────────────────────────────────
# Run this shell loop (adjust sizes as needed):
#   for SIZE in 100 1000 5000 10000; do
#       echo "=== Load test n=$SIZE ==="
#       BENCHMARK_DATASET_SIZE=$SIZE \
#       locust -f benchmarks/load_test.py \
#           --headless -u 1 -r 1 --run-time 60s \
#           --host http://localhost:8000
#       sleep 5
#   done

import os
import sys
import io
import csv
import requests as req_lib
from datetime import datetime, timezone

from locust import HttpUser, task, between, events

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    BENCHMARK_USERNAME, BENCHMARK_PASSWORD, LOGIN_ENDPOINT, LOAD_RESULTS_FILE
)
from csv_writer import LOAD_COLUMNS


# ── Load image bytes once at module level (shared across all users) ───────────
def _load_image() -> bytes:
    path = os.getenv("BENCHMARK_FACE_IMAGE")
    if path and os.path.isfile(path):
        with open(path, "rb") as f:
            return f.read()
    # Minimal valid JPEG — will 422 on face detection, but exercises
    # auth + file validation + DB layers
    return bytes([
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


FACE_IMAGE_BYTES: bytes = _load_image()
DATASET_SIZE: str = os.getenv("BENCHMARK_DATASET_SIZE", "unknown")


def _get_token(host: str) -> str:
    """Fetch a JWT from /auth/login before the load test starts."""
    resp = req_lib.post(
        f"{host}{LOGIN_ENDPOINT}",
        data={"username": BENCHMARK_USERNAME, "password": BENCHMARK_PASSWORD},
        timeout=10,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Login failed: {resp.status_code} {resp.text}")
    return resp.json()["access_token"]


# Global token — obtained once, shared by all users
_TOKEN: str | None = None


class RecognitionUser(HttpUser):
    """
    Simulates one concurrent user hitting /recognize repeatedly.

    wait_time: between(3, 5) → average ~10 req/min per user.
    With 1 user this stays safely under the 20/min rate limit.
    Increase users only if running from multiple IPs.
    """
    wait_time = between(3, 5)

    def on_start(self):
        """Called once per virtual user at spawn time — get auth token."""
        global _TOKEN
        if _TOKEN is None:
            _TOKEN = _get_token(self.host)
        self.token = _TOKEN

    @task
    def recognize(self):
        headers = {"Authorization": f"Bearer {self.token}"}
        files = {"file": ("bench.jpg", io.BytesIO(FACE_IMAGE_BYTES), "image/jpeg")}

        with self.client.post(
            "/recognize",
            headers=headers,
            files=files,
            name="/recognize",
            catch_response=True,
        ) as resp:
            # 200 = matched or no-match (both are valid responses)
            # 422 = no face detected (expected with fake image)
            # 429 = rate limited (fail — we're going too fast)
            if resp.status_code in (200, 422):
                resp.success()
            elif resp.status_code == 429:
                resp.failure(f"Rate limited: {resp.text}")
            elif resp.status_code == 401:
                # Token expired — fetch a new one
                global _TOKEN
                _TOKEN = _get_token(self.host)
                self.token = _TOKEN
                resp.failure("Token expired, refreshed")
            else:
                resp.failure(f"Unexpected {resp.status_code}: {resp.text[:100]}")


# ── Save results to CSV when test finishes ────────────────────────────────────
@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    stats = environment.stats.total

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_label": os.getenv("BENCHMARK_LABEL", "pre_pgvector"),
        "dataset_size": DATASET_SIZE,
        "num_users": environment.runner.user_count if environment.runner else 1,
        "avg_latency_ms": round(stats.avg_response_time, 2),
        "p95_latency_ms": round(stats.get_response_time_percentile(0.95) or 0, 2),
        "rps": round(stats.current_rps, 3),
        "failure_rate": round(stats.fail_ratio, 4),
        "num_requests": stats.num_requests,
        "num_failures": stats.num_failures,
    }

    os.makedirs(os.path.dirname(LOAD_RESULTS_FILE), exist_ok=True)
    write_header = not os.path.exists(LOAD_RESULTS_FILE)
    with open(LOAD_RESULTS_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOAD_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"\nLoad test result saved to {LOAD_RESULTS_FILE}")
    print(f"  n={DATASET_SIZE}  avg={row['avg_latency_ms']}ms  "
          f"p95={row['p95_latency_ms']}ms  rps={row['rps']}")
