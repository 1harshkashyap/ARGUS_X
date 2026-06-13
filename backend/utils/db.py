import asyncio
import httpx
from typing import Optional, List, Dict, Any
from utils.logger import logger
from config import settings


# ── Supabase REST client (httpx-based) ────────────────────────────────
# supabase-py 2.3.x is incompatible with Python 3.14 (typing.Union changes).
# This wrapper calls the PostgREST API directly via httpx — stable, zero deps issues.

_http_client: Optional[httpx.AsyncClient] = None
_stats_lock = asyncio.Lock()


def _get_headers() -> Dict[str, str]:
    """Build Supabase auth headers. Key value never logged."""
    return {
        "apikey": settings.SUPABASE_KEY,
        "Authorization": f"Bearer {settings.SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


def _rest_url(table: str) -> str:
    """Build the PostgREST URL for a table."""
    return f"{settings.SUPABASE_URL}/rest/v1/{table}"


async def _get_http_client() -> Optional[httpx.AsyncClient]:
    """
    Lazy initialization of async httpx client.
    Returns None if credentials not configured — system continues without DB.
    """
    global _http_client
    if _http_client is not None and not _http_client.is_closed:
        return _http_client

    if not settings.SUPABASE_URL or not settings.SUPABASE_KEY:
        logger.warning("Supabase credentials not configured — DB operations disabled")
        return None

    try:
        _http_client = httpx.AsyncClient(
            headers=_get_headers(),
            timeout=httpx.Timeout(settings.DB_TIMEOUT),
        )
        logger.info("Supabase HTTP client initialized")
        return _http_client
    except Exception as e:
        logger.error(f"Failed to initialize Supabase HTTP client: {type(e).__name__}: {str(e)[:200]}")
        return None


# ── Write operations ──────────────────────────────────────────────────

async def log_event(event: Dict[str, Any]) -> Optional[str]:
    """Log a security event. Returns inserted event UUID on success, None on failure."""
    client = await _get_http_client()
    if not client:
        return None
    try:
        response = await client.post(_rest_url("events"), json=event)
        if response.status_code in (200, 201):
            data = response.json()
            if data and isinstance(data, list) and len(data) > 0:
                return data[0].get("id")
            return ""  # Inserted but couldn't extract UUID
        return None
    except httpx.TimeoutException:
        logger.warning(f"log_event timed out after {settings.DB_TIMEOUT}s")
        return None
    except Exception as e:
        logger.warning(f"log_event failed: {type(e).__name__}: {str(e)[:100]}")
        return None


async def log_xai_decision(decision: Dict[str, Any]) -> bool:
    """Log an XAI decision. Non-fatal — returns False on failure."""
    client = await _get_http_client()
    if not client:
        return False
    try:
        response = await client.post(_rest_url("xai_decisions"), json=decision)
        return response.status_code in (200, 201)
    except httpx.TimeoutException:
        logger.warning(f"log_xai_decision timed out after {settings.DB_TIMEOUT}s")
        return False
    except Exception as e:
        logger.warning(f"log_xai_decision failed: {type(e).__name__}: {str(e)[:100]}")
        return False


async def update_battle_state(state: Dict[str, Any]) -> bool:
    """Update battle state row (id=1). Non-fatal — returns False on failure."""
    client = await _get_http_client()
    if not client:
        return False
    try:
        response = await client.patch(
            _rest_url("battle_state"),
            params={"id": "eq.1"},
            json=state,
        )
        return response.status_code in (200, 204)
    except httpx.TimeoutException:
        logger.warning(f"update_battle_state timed out after {settings.DB_TIMEOUT}s")
        return False
    except Exception as e:
        logger.warning(f"update_battle_state failed: {type(e).__name__}: {str(e)[:100]}")
        return False


async def update_stats(increments: Dict[str, int]) -> bool:
    """
    Increment stats counters. Non-fatal — returns False on failure.
    increments: {"total_events": 1, "total_blocked": 1}
    Uses asyncio.Lock to prevent TOCTOU race on concurrent read-modify-write.
    """
    async with _stats_lock:
        client = await _get_http_client()
        if not client:
            return False
        try:
            # Read current stats
            response = await client.get(
                _rest_url("stats"),
                params={"id": "eq.1", "select": "*"},
            )
            if response.status_code != 200:
                return False

            data = response.json()
            if not data:
                return False

            current = data[0]
            updates = {}
            for key, inc in increments.items():
                updates[key] = current.get(key, 0) + inc

            response = await client.patch(
                _rest_url("stats"),
                params={"id": "eq.1"},
                json=updates,
            )
            return response.status_code in (200, 204)
        except httpx.TimeoutException:
            logger.warning(f"update_stats timed out after {settings.DB_TIMEOUT}s")
            return False
        except Exception as e:
            logger.warning(f"Stats update failed: {type(e).__name__}")
            return False


async def add_dynamic_rule(pattern: str, threat_type: str, source_attack: str) -> bool:
    """Add a new dynamic firewall rule. Non-fatal — returns False on failure."""
    client = await _get_http_client()
    if not client:
        return False
    try:
        response = await client.post(
            _rest_url("dynamic_rules"),
            json={
                "pattern": pattern,
                "threat_type": threat_type,
                "source_attack": source_attack[:200],
            },
        )
        return response.status_code in (200, 201)
    except httpx.TimeoutException:
        logger.warning(f"add_dynamic_rule timed out after {settings.DB_TIMEOUT}s")
        return False
    except Exception as e:
        logger.warning(f"add_dynamic_rule failed: {type(e).__name__}: {str(e)[:100]}")
        return False


async def add_campaign(campaign: Dict[str, Any]) -> bool:
    """Add or update a threat campaign. Non-fatal."""
    client = await _get_http_client()
    if not client:
        return False
    try:
        # Upsert via PostgREST: Prefer: resolution=merge-duplicates
        headers = {"Prefer": "return=representation,resolution=merge-duplicates"}
        response = await client.post(
            _rest_url("campaigns"),
            json=campaign,
            headers=headers,
        )
        return response.status_code in (200, 201)
    except httpx.TimeoutException:
        logger.warning(f"add_campaign timed out after {settings.DB_TIMEOUT}s")
        return False
    except Exception as e:
        logger.warning(f"add_campaign failed: {type(e).__name__}: {str(e)[:100]}")
        return False


# ── Read operations ───────────────────────────────────────────────────

async def get_recent_events(limit: int = 20) -> List[Dict]:
    """Get recent security events. Returns [] on failure."""
    limit = min(max(limit, 1), 100)  # Cap between 1 and 100
    client = await _get_http_client()
    if not client:
        return []
    try:
        response = await client.get(
            _rest_url("events"),
            params={
                "select": "*",
                "order": "created_at.desc",
                "limit": str(limit),
            },
        )
        if response.status_code == 200:
            return response.json()
        return []
    except Exception:
        return []


async def get_stats() -> Dict[str, Any]:
    """Get running stats. Returns safe defaults on failure."""
    default = {
        "total_events": 0,
        "total_blocked": 0,
        "total_bypasses": 0,
        "total_mutations": 0,
    }
    client = await _get_http_client()
    if not client:
        return default
    try:
        response = await client.get(
            _rest_url("stats"),
            params={"id": "eq.1", "select": "*"},
        )
        if response.status_code == 200:
            data = response.json()
            if data:
                return data[0]
        return default
    except Exception:
        return default


async def get_dynamic_rules() -> List[Dict]:
    """Get all dynamic firewall rules. Returns [] on failure."""
    client = await _get_http_client()
    if not client:
        return []
    try:
        response = await client.get(
            _rest_url("dynamic_rules"),
            params={"select": "*"},
        )
        if response.status_code == 200:
            return response.json()
        return []
    except Exception:
        return []


async def get_battle_state() -> Optional[Dict]:
    """Get current battle state. Returns None on failure."""
    client = await _get_http_client()
    if not client:
        return None
    try:
        response = await client.get(
            _rest_url("battle_state"),
            params={"id": "eq.1", "select": "*"},
        )
        if response.status_code == 200:
            data = response.json()
            if data:
                return data[0]
        return None
    except Exception:
        return None


async def get_campaigns(limit: int = 10) -> List[Dict]:
    """Get active threat campaigns. Returns [] on failure."""
    limit = min(max(limit, 1), 100)  # Cap between 1 and 100
    client = await _get_http_client()
    if not client:
        return []
    try:
        response = await client.get(
            _rest_url("campaigns"),
            params={
                "select": "*",
                "order": "last_seen.desc",
                "limit": str(limit),
            },
        )
        if response.status_code == 200:
            return response.json()
        return []
    except Exception:
        return []


async def get_xai_decisions(limit: int = 10) -> List[Dict]:
    """Get recent XAI decisions. Returns [] on failure."""
    limit = min(max(limit, 1), 100)  # Cap between 1 and 100
    client = await _get_http_client()
    if not client:
        return []
    try:
        response = await client.get(
            _rest_url("xai_decisions"),
            params={
                "select": "*",
                "order": "created_at.desc",
                "limit": str(limit),
            },
        )
        if response.status_code == 200:
            return response.json()
        return []
    except Exception:
        return []


# ── Lifecycle ─────────────────────────────────────────────────────────

async def close_db_client():
    """Close the httpx client on app shutdown. Prevents connection leaks."""
    global _http_client
    if _http_client and not _http_client.is_closed:
        await _http_client.aclose()
        _http_client = None
        logger.info("Supabase HTTP client closed")


# ── Health check ──────────────────────────────────────────────────────

async def check_connection() -> bool:
    """Test DB connection. Returns True if healthy, False if not."""
    client = await _get_http_client()
    if not client:
        return False
    try:
        response = await client.get(
            _rest_url("stats"),
            params={"id": "eq.1", "select": "id"},
        )
        return response.status_code == 200 and bool(response.json())
    except Exception:
        return False
