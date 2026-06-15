"""
ARGUS-X TUI — Configuration
Reads from environment variables or .env file.
API key is NEVER persisted — held in memory only for the session.
"""

from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the cli directory if it exists
_ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(_ENV_PATH)

API_URL:       str = os.getenv("ARGUS_API_URL",       "http://localhost:8000").rstrip("/")
DASHBOARD_KEY: str = os.getenv("ARGUS_DASHBOARD_KEY", "argus-dashboard-2025")

# GEMINI_KEY is loaded from .env as a convenience default only.
# The user can override it at startup via the key gate screen.
# It is NEVER written to disk by the TUI itself.
_ENV_GEMINI_KEY: str = os.getenv("ARGUS_GEMINI_KEY", "")

# Poll intervals (seconds)
STATS_POLL_INTERVAL:  float = 10.0
EVENTS_POLL_INTERVAL: float = 5.0
BATTLE_POLL_INTERVAL: float = 5.0
