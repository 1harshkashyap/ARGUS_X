"""
ARGUS-X TUI — Async API Client
Wraps every backend endpoint. Never raises — returns Result types.
All methods are async (Textual is async-native).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import httpx
from config import API_URL, DASHBOARD_KEY


# ── Result type ───────────────────────────────────────────────────────

@dataclass
class Ok:
    data: Any

@dataclass
class Err:
    message: str
    status: int = 0

Result = Ok | Err


# ── HTTP helpers ──────────────────────────────────────────────────────

_DASHBOARD_HEADERS = {"X-Dashboard-Key": DASHBOARD_KEY}


async def _get(path: str, headers: dict | None = None, timeout: float = 10.0) -> Result:
    if headers is None:
        headers = {}
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(f"{API_URL}{path}", headers=headers)
            r.raise_for_status()
            return Ok(r.json())
    except httpx.HTTPStatusError as e:
        return Err(f"HTTP {e.response.status_code}", e.response.status_code)
    except httpx.ConnectError:
        return Err("Cannot connect to backend", 0)
    except httpx.TimeoutException:
        return Err("Request timed out", 0)
    except Exception as e:
        return Err(str(e), 0)


async def _post(
    path: str,
    body: dict,
    headers: dict | None = None,
    timeout: float = 35.0,
) -> Result:
    if headers is None:
        headers = {}
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(
                f"{API_URL}{path}",
                json=body,
                headers={"Content-Type": "application/json", **headers},
            )
            r.raise_for_status()
            return Ok(r.json())
    except httpx.HTTPStatusError as e:
        try:
            detail = e.response.json().get("error", str(e))
        except Exception:
            detail = str(e)
        return Err(detail, e.response.status_code)
    except httpx.ConnectError:
        return Err("Cannot connect to backend", 0)
    except httpx.TimeoutException:
        return Err("Request timed out", 0)
    except Exception as e:
        return Err(str(e), 0)


# ── Public API ────────────────────────────────────────────────────────

async def health() -> Result:
    return await _get("/health")


async def stats() -> Result:
    return await _get("/api/v1/analytics/stats", headers=_DASHBOARD_HEADERS)


async def logs(limit: int = 30) -> Result:
    return await _get(
        f"/api/v1/analytics/logs?limit={limit}",
        headers=_DASHBOARD_HEADERS,
    )


async def xai_decisions(limit: int = 10) -> Result:
    return await _get(
        f"/api/v1/analytics/xai?limit={limit}",
        headers=_DASHBOARD_HEADERS,
    )


async def battle_state() -> Result:
    return await _get("/api/v1/battle/state", headers=_DASHBOARD_HEADERS)


async def agent_status() -> Result:
    return await _get("/api/v1/agents/status", headers=_DASHBOARD_HEADERS)


async def chat(message: str, session_id: str, api_key: str) -> Result:
    return await _post(
        "/api/v1/chat",
        body={"message": message, "session_id": session_id, "api_key": api_key},
        timeout=35.0,
    )


async def pause() -> Result:
    return await _post(
        "/api/v1/agents/pause",
        body={},
        headers=_DASHBOARD_HEADERS,
        timeout=10.0,
    )


async def resume() -> Result:
    return await _post(
        "/api/v1/agents/resume",
        body={},
        headers=_DASHBOARD_HEADERS,
        timeout=10.0,
    )


async def cycle() -> Result:
    return await _post(
        "/api/v1/agents/cycle",
        body={},
        headers=_DASHBOARD_HEADERS,
        timeout=40.0,
    )
