import pytest
import requests


def test_battle_state_endpoint(base_url, dashboard_headers):
    r = requests.get(f"{base_url}/api/v1/battle/state",
                      headers=dashboard_headers, timeout=10)
    assert r.status_code == 200
    data = r.json()
    for field in ["tick", "red_attacks", "red_bypasses", "blue_blocks",
                   "blue_patches", "red_tier", "red_strategy",
                   "current_attack_preview", "last_attack_result"]:
        assert field in data, f"missing field: {field}"


def test_agent_status_endpoint(base_url, dashboard_headers):
    r = requests.get(f"{base_url}/api/v1/agents/status",
                      headers=dashboard_headers, timeout=10)
    assert r.status_code == 200
    data = r.json()
    assert "is_paused" in data
    assert "uptime_seconds" in data
    assert 1 <= data["red_tier"] <= 5


def test_tier_escalation_formula():
    """tier = min(5, tick // 10 + 1) — matches BattleEngine._run_tick()
    line: self.red_tier = min(5, self.tick // 10 + 1)"""
    for tick, expected in [(1, 1), (9, 1), (10, 2), (19, 2),
                           (20, 3), (30, 4), (40, 5), (100, 5)]:
        assert min(5, tick // 10 + 1) == expected, (
            f"tick={tick}: expected tier {expected}, got {min(5, tick // 10 + 1)}"
        )


@pytest.mark.slow
def test_pause_resume_cycle(base_url, dashboard_headers):
    # Pause
    r = requests.post(f"{base_url}/api/v1/agents/pause",
                       headers=dashboard_headers, timeout=10)
    assert r.status_code == 200 and r.json()["success"] is True

    status = requests.get(f"{base_url}/api/v1/agents/status",
                           headers=dashboard_headers, timeout=10).json()
    assert status["is_paused"] is True

    # Forced cycle works even while paused
    before = status["tick"]
    r = requests.post(f"{base_url}/api/v1/agents/cycle",
                       headers=dashboard_headers, timeout=40)
    assert r.status_code == 200
    result = r.json()
    assert result["tick"] > before
    assert result["last_attack_result"] in ("BLOCKED", "BYPASSED", "SKIPPED")

    # Resume
    r = requests.post(f"{base_url}/api/v1/agents/resume",
                       headers=dashboard_headers, timeout=10)
    assert r.status_code == 200 and r.json()["success"] is True

    status2 = requests.get(f"{base_url}/api/v1/agents/status",
                            headers=dashboard_headers, timeout=10).json()
    assert status2["is_paused"] is False
