# benchmarks/config.py
#
# Single source of truth for all benchmark parameters.
# All other benchmark scripts import from here.
# Change values here to affect every script at once.

import os

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED = 42
EMBEDDING_DIM = 512   # ArcFace via buffalo_l outputs 512-dim vectors

# ── Dataset sizes to benchmark ───────────────────────────────────────────────
DATASET_SIZES = [100, 1000, 5000, 10000]

# ── How many recognition requests to make per dataset size ───────────────────
# 30 gives stable statistics (mean, p95, stdev) without taking too long.
# Increase to 50+ for tighter confidence intervals if you have time.
ITERATIONS_PER_SIZE = 30

# ── App connection ───────────────────────────────────────────────────────────
# When running inside Docker Compose the API service is "api" on port 8000.
# When running from your host machine it's localhost:8000.
BASE_URL = os.getenv("BENCHMARK_BASE_URL", "http://localhost:8000")

# Endpoint paths (match your router definitions exactly)
RECOGNIZE_ENDPOINT = "/recognize"
LOGIN_ENDPOINT = "/auth/login"

# ── Benchmark auth account ───────────────────────────────────────────────────
# Create a dedicated account for benchmarking so you can track/rate-limit
# it separately. Credentials are read from env vars so they never appear
# in source control.
#
# Before running benchmarks, register this account once:
#   POST /auth/register  {"username": "bench_user", "password": "bench_pass"}
#
BENCHMARK_USERNAME = os.getenv("BENCHMARK_USERNAME", "bench_user")
BENCHMARK_PASSWORD = os.getenv("BENCHMARK_PASSWORD", "bench_pass_2026")

# ── Database (psycopg v3 — matches your project's driver) ────────────────────
# Reads the same DATABASE_URL your app uses.  Override via env if needed.
# Note: your validator converts postgresql:// → postgresql+psycopg://
# For psycopg v3 direct use we need plain postgresql:// (no SQLAlchemy dialect prefix).
DB_HOST = os.getenv("BENCHMARK_DB_HOST", "localhost")
DB_PORT = os.getenv("BENCHMARK_DB_PORT", "5432")
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
DB_NAME = os.getenv("POSTGRES_DB", "face_recognition")

# ── Output ───────────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
RESULTS_FILE = os.path.join(RESULTS_DIR, "results.csv")
LOAD_RESULTS_FILE = os.path.join(RESULTS_DIR, "results_load.csv")

# ── Rate-limit awareness ─────────────────────────────────────────────────────
# Your /recognize route is capped at 20/minute (SlowAPI, per IP).
# We stay well below that so the benchmark measures recognition latency,
# not rate-limit rejection latency.
# 30 iterations × ~2s gap each ≈ ~1 min → safe margin under 20/min.
REQUEST_GAP_SECONDS = 2.5   # sleep between individual requests

# ── Load test settings ───────────────────────────────────────────────────────
# Keep concurrent users low enough to stay under 20 req/min per IP.
# Locust spawns all users from one IP, so 1 user / 3s = 20/min ceiling.
# Set LOCUST_USERS=1 and LOCUST_WAIT=(3,5) for a safe single-user load test,
# or run from multiple IPs to simulate real concurrency.
LOCUST_USERS = int(os.getenv("LOCUST_USERS", "1"))
LOCUST_SPAWN_RATE = int(os.getenv("LOCUST_SPAWN_RATE", "1"))
LOCUST_RUN_TIME = os.getenv("LOCUST_RUN_TIME", "60s")
