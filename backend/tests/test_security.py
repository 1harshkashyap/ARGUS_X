import pytest
import requests
from conftest import chat


def test_health_is_public(base_url):
    r = requests.get(f"{base_url}/health", timeout=10)
    assert r.status_code == 200


def test_analytics_stats_requires_key(base_url):
    r = requests.get(f"{base_url}/api/v1/analytics/stats", timeout=10)
    assert r.status_code == 401


def test_analytics_stats_with_valid_key(base_url, dashboard_headers):
    r = requests.get(f"{base_url}/api/v1/analytics/stats",
                      headers=dashboard_headers, timeout=10)
    assert r.status_code == 200
    assert "total_events" in r.json()


def test_analytics_logs_requires_key(base_url):
    r = requests.get(f"{base_url}/api/v1/analytics/logs", timeout=10)
    assert r.status_code == 401


def test_analytics_xai_requires_key(base_url):
    r = requests.get(f"{base_url}/api/v1/analytics/xai", timeout=10)
    assert r.status_code == 401


def test_wrong_dashboard_key_rejected(base_url):
    r = requests.get(f"{base_url}/api/v1/analytics/stats",
                      headers={"X-Dashboard-Key": "wrong-key"}, timeout=10)
    assert r.status_code == 401


def test_battle_state_requires_key(base_url):
    r = requests.get(f"{base_url}/api/v1/battle/state", timeout=10)
    assert r.status_code == 401


def test_agents_status_requires_key(base_url):
    r = requests.get(f"{base_url}/api/v1/agents/status", timeout=10)
    assert r.status_code == 401


def test_agents_pause_requires_key(base_url):
    r = requests.post(f"{base_url}/api/v1/agents/pause", json={}, timeout=10)
    assert r.status_code == 401


def test_security_headers_present(base_url):
    r = requests.get(f"{base_url}/health", timeout=10)
    headers_lower = {k.lower() for k in r.headers}
    assert (
        "x-content-type-options" in headers_lower
        or "x-frame-options" in headers_lower
    )


def test_oversized_message_rejected(base_url, session_id):
    """Message exceeding schema max_length (10000) must be rejected
    before reaching the pipeline — 400/413/422 depending on where
    the limit is enforced (body-size middleware vs Pydantic)."""
    huge_message = "a" * 20000
    r = requests.post(
        f"{base_url}/api/v1/chat",
        json={"message": huge_message, "session_id": session_id},
        timeout=15,
    )
    assert r.status_code in (400, 413, 422), f"got {r.status_code}"


@pytest.mark.slow
def test_burst_traffic_does_not_crash_server(base_url):
    """40 rapid requests to /health — either all succeed or some
    return 429, but the server must remain healthy afterward."""
    statuses = []
    for _ in range(40):
        r = requests.get(f"{base_url}/health", timeout=10)
        statuses.append(r.status_code)
    assert all(s in (200, 429) for s in statuses), f"unexpected: {set(statuses)}"

    r = requests.get(f"{base_url}/health", timeout=10)
    assert r.status_code == 200, "server unresponsive after burst"
