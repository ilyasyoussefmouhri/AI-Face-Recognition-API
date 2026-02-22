# benchmarks/auth_helper.py
#
# Obtains a JWT from your real /auth/login endpoint so benchmark requests
# go through the full auth stack — exactly as a real client would.
#
# Your login route:
#   POST /auth/login
#   Content-Type: application/x-www-form-urlencoded   (OAuth2PasswordRequestForm)
#   Body: username=...&password=...
#   Response: {"access_token": "...", "token_type": "bearer"}
#
# Rate limit on /auth/login: 10/minute (from your limiter decorator).
# We fetch the token once and reuse it for all benchmark requests.

import requests
from config import BASE_URL, LOGIN_ENDPOINT, BENCHMARK_USERNAME, BENCHMARK_PASSWORD


def get_benchmark_token() -> str:
    """
    Authenticate as the benchmark user and return the Bearer token string.

    Raises RuntimeError if login fails (e.g. user not yet registered,
    wrong password, or app not running).
    """
    url = f"{BASE_URL}{LOGIN_ENDPOINT}"

    # OAuth2PasswordRequestForm expects form-encoded data, not JSON
    response = requests.post(
        url,
        data={
            "username": BENCHMARK_USERNAME,
            "password": BENCHMARK_PASSWORD,
        },
        timeout=10,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Login failed ({response.status_code}): {response.text}\n"
            f"Make sure you have registered the benchmark user first:\n"
            f"  POST {BASE_URL}/auth/register\n"
            f'  Body: {{"username": "{BENCHMARK_USERNAME}", "password": "{BENCHMARK_PASSWORD}"}}'
        )

    token = response.json().get("access_token")
    if not token:
        raise RuntimeError(f"No access_token in login response: {response.json()}")

    print(f"Obtained JWT for '{BENCHMARK_USERNAME}' (expires in ~30min by default)")
    return token


def register_benchmark_user() -> bool:
    """
    Register the benchmark account if it doesn't exist yet.
    Returns True if newly created, False if already exists (409).
    Raises RuntimeError for unexpected errors.
    """
    url = f"{BASE_URL}/auth/register"
    response = requests.post(
        url,
        json={
            "username": BENCHMARK_USERNAME,
            "password": BENCHMARK_PASSWORD,
        },
        timeout=10,
    )

    if response.status_code == 201:
        print(f"Registered benchmark user '{BENCHMARK_USERNAME}'")
        return True
    elif response.status_code == 400 and "already exists" in response.text:
        print(f"Benchmark user '{BENCHMARK_USERNAME}' already registered — OK")
        return False
    else:
        raise RuntimeError(
            f"Unexpected registration response ({response.status_code}): {response.text}"
        )


if __name__ == "__main__":
    register_benchmark_user()
    token = get_benchmark_token()
    print(f"Token preview: {token[:40]}...")
