# app/middleware/benchmark_timing.py
#
# Non-invasive timing middleware — active ONLY when BENCHMARK_MODE=true.
# When disabled (the default), it is a ~zero-cost passthrough and has
# absolutely no effect on production behaviour.
#
# What it measures:
#   X-Total-Time-Ms     — full request time seen by the middleware
#   X-DB-Time-Ms        — time to fetch all face rows from PostgreSQL
#   X-Similarity-Time-Ms — time for the Python cosine similarity loop
#
# How times are injected:
#   recognition.py sets request.state.db_time_ms and
#   request.state.similarity_time_ms when BENCHMARK_MODE=true.
#   This middleware reads those values and writes them as response headers.
#
# How to activate:
#   Set BENCHMARK_MODE=true in your environment (docker-compose.benchmark.yml).
#
# How to remove when done:
#   Delete this file and remove the two lines in app/main.py that register it.

import os
import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

BENCHMARK_MODE: bool = os.getenv("BENCHMARK_MODE", "false").lower() == "true"


class BenchmarkTimingMiddleware(BaseHTTPMiddleware):
    """
    Adds timing headers to responses on the /recognize endpoint.
    All other endpoints are passed through unchanged.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        # Fast path: skip entirely unless BENCHMARK_MODE is on
        if not BENCHMARK_MODE:
            return await call_next(request)

        # Only instrument the recognition endpoint — it's the one being benchmarked
        if not request.url.path.endswith("/recognize"):
            return await call_next(request)

        # Initialise state slots so recognition.py can write into them
        request.state.db_time_ms = None
        request.state.similarity_time_ms = None

        wall_start = time.perf_counter()
        response = await call_next(request)
        wall_ms = (time.perf_counter() - wall_start) * 1000

        response.headers["X-Total-Time-Ms"] = f"{wall_ms:.2f}"

        if request.state.db_time_ms is not None:
            response.headers["X-DB-Time-Ms"] = f"{request.state.db_time_ms:.2f}"

        if request.state.similarity_time_ms is not None:
            response.headers["X-Similarity-Time-Ms"] = f"{request.state.similarity_time_ms:.2f}"

        return response
