import sys
import os
import uuid
import pytest
import requests

# Make backend/ importable (utils.db, agents.battle_engine, etc.)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_URL = os.environ.get("ARGUS_TEST_BASE_URL", "http://localhost:8000")
DASHBOARD_KEY = os.environ.get("ARGUS_DASHBOARD_KEY", "argus-dashboard-2025")


@pytest.fixture(scope="session")
def base_url():
    return BASE_URL


@pytest.fixture(scope="session")
def dashboard_headers():
    return {"X-Dashboard-Key": DASHBOARD_KEY}


@pytest.fixture
def session_id():
    """Fresh session_id per test — avoids cross-test state pollution
    in the in-memory SessionTracker (1hr TTL, shared across tests)."""
    return f"pytest_{uuid.uuid4().hex[:12]}"


def chat(base_url, message, session_id, api_key=None, timeout=35):
    """POST /api/v1/chat — returns the requests.Response object."""
    body = {"message": message, "session_id": session_id}
    if api_key is not None:
        body["api_key"] = api_key
    return requests.post(f"{base_url}/api/v1/chat", json=body, timeout=timeout)
