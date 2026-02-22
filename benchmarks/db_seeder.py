# benchmarks/db_seeder.py
#
# Seeds the database with synthetic face rows for benchmarking,
# and cleans them up afterwards.
#
# ── Schema reality check ─────────────────────────────────────────────────────
# Your faces table (from alembic/versions/cfaffc97d402_initial_schema.py):
#   face_id         UUID  PK
#   user_id         UUID  FK → users.user_id
#   embedding       ARRAY(Float)
#   detection_score Float
#
# Your users table:
#   user_id     UUID  PK
#   name        String
#   surname     String
#   auth_user_id UUID FK → auth_users.auth_user_id  (CASCADE on delete)
#
# There is NO "source" column on faces or users.
# Strategy: create a real AuthUser + User + Face row per synthetic identity,
# using a recognisable username prefix so we can delete them cleanly.
#
# ── Driver ───────────────────────────────────────────────────────────────────
# Your project uses psycopg v3 (psycopg, not psycopg2).
# We use psycopg directly here — no SQLAlchemy overhead needed for bulk inserts.

import time
import uuid
import psycopg
from psycopg import sql
from config import (
    DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME, RANDOM_SEED
)
from data_gen import generate_embeddings

BENCH_USERNAME_PREFIX = "bench_seed_"


def _connect() -> psycopg.Connection:
    """Open a psycopg v3 connection to the benchmark database."""
    return psycopg.connect(
        host=DB_HOST,
        port=int(DB_PORT),
        user=DB_USER,
        password=DB_PASSWORD,
        dbname=DB_NAME,
    )


def clear_benchmark_data() -> int:
    """
    Delete all rows inserted by previous benchmark runs.

    Deletes AuthUser rows whose username starts with BENCH_USERNAME_PREFIX.
    The CASCADE on auth_users → users → faces means one delete cleans
    up all three tables automatically.

    Returns the number of AuthUser rows deleted.
    """
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM auth_users WHERE username LIKE %s RETURNING auth_user_id",
                (BENCH_USERNAME_PREFIX + "%",)
            )
            deleted = cur.rowcount
        conn.commit()
    print(f"Cleared {deleted} benchmark AuthUser rows (cascade removes users + faces).")
    return deleted


def seed_embeddings(n: int) -> float:
    """
    Insert n synthetic face rows (+ their owning User and AuthUser rows).

    Steps:
      1. Clear any previous benchmark data.
      2. Generate n normalised embeddings.
      3. Bulk-insert n AuthUser rows.
      4. Bulk-insert n User rows (one per AuthUser).
      5. Bulk-insert n Face rows (one per User).

    Returns: wall-clock insert time in seconds (steps 3-5 only).

    Why separate bulk inserts instead of row-by-row?
    executemany() with psycopg v3 uses a prepared statement pipeline —
    vastly faster than individual round-trips for large n.
    """
    clear_benchmark_data()

    print(f"Generating {n} synthetic embeddings (seed={RANDOM_SEED})...")
    embeddings = generate_embeddings(n)

    # Pre-generate all UUIDs in Python to avoid repeated DB round-trips
    auth_user_ids = [uuid.uuid4() for _ in range(n)]
    user_ids      = [uuid.uuid4() for _ in range(n)]
    fake_hash     = "$2b$12$benchmarkplaceholderhashXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"

    print(f"Inserting {n} rows into auth_users, users, faces...")
    t_start = time.perf_counter()

    with _connect() as conn:
        with conn.cursor() as cur:

            # 1. auth_users ────────────────────────────────────────────────
            cur.executemany(
                """
                INSERT INTO auth_users
                    (auth_user_id, username, password_hash, is_active, is_admin)
                VALUES (%s, %s, %s, true, false)
                """,
                [
                    (auth_user_ids[i], f"{BENCH_USERNAME_PREFIX}{i}", fake_hash)
                    for i in range(n)
                ],
            )

            # 2. users ─────────────────────────────────────────────────────
            cur.executemany(
                """
                INSERT INTO users (user_id, name, surname, auth_user_id)
                VALUES (%s, %s, %s, %s)
                """,
                [
                    (user_ids[i], "Bench", f"User{i}", auth_user_ids[i])
                    for i in range(n)
                ],
            )

            # 3. faces ─────────────────────────────────────────────────────
            # psycopg v3 stores Python lists into ARRAY columns natively.
            # detection_score = 0.99 is a plausible value; it won't affect
            # similarity computation (recognition.py ignores it during matching).
            cur.executemany(
                """
                INSERT INTO faces (face_id, user_id, embedding, detection_score)
                VALUES (%s, %s, %s, %s)
                """,
                [
                    (uuid.uuid4(), user_ids[i], embeddings[i], 0.99)
                    for i in range(n)
                ],
            )

        conn.commit()

    elapsed = time.perf_counter() - t_start
    print(f"Inserted {n} rows across 3 tables in {elapsed:.2f}s")
    return elapsed


def count_benchmark_rows() -> dict:
    """Return current counts of benchmark rows — useful for sanity checks."""
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM auth_users WHERE username LIKE %s",
                (BENCH_USERNAME_PREFIX + "%",)
            )
            auth_count = cur.fetchone()[0]

            cur.execute(
                """
                SELECT COUNT(*) FROM users u
                JOIN auth_users a ON u.auth_user_id = a.auth_user_id
                WHERE a.username LIKE %s
                """,
                (BENCH_USERNAME_PREFIX + "%",)
            )
            user_count = cur.fetchone()[0]

            cur.execute(
                """
                SELECT COUNT(*) FROM faces f
                JOIN users u ON f.user_id = u.user_id
                JOIN auth_users a ON u.auth_user_id = a.auth_user_id
                WHERE a.username LIKE %s
                """,
                (BENCH_USERNAME_PREFIX + "%",)
            )
            face_count = cur.fetchone()[0]

    return {"auth_users": auth_count, "users": user_count, "faces": face_count}


if __name__ == "__main__":
    # Quick smoke test: seed 10 rows and verify counts
    elapsed = seed_embeddings(10)
    counts = count_benchmark_rows()
    print(f"Counts after seeding: {counts}")
    print(f"Insert time: {elapsed:.3f}s")
