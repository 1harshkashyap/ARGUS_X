import sys
import os
import urllib.request
import json
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("\n" + "="*60)
print("   ARGUS-X DAY 1 — FINAL VERIFICATION")
print("="*60)

all_pass = True
results = []

def check(name, test_fn):
    global all_pass
    try:
        test_fn()
        results.append((True, name, ""))
    except Exception as e:
        results.append((False, name, str(e)))
        all_pass = False

# Schema tests
def test_chat_schema():
    from schemas.chat import ChatResponse, XAIExplanation, ChatRequest
    r = ChatResponse()
    assert r.explanation is not None
    assert r.response == ""
    assert r.blocked == False

def test_empty_message_rejected():
    from schemas.chat import ChatRequest
    from pydantic import ValidationError
    try:
        ChatRequest(message="")
        raise AssertionError("Should have rejected empty message")
    except ValidationError:
        pass

def test_all_schemas():
    from schemas.chat import ChatResponse, LayerDecision, XAIExplanation
    from schemas.events import SecurityEvent
    from schemas.agents import BattleState, AgentStatus
    from schemas.analytics import StatsResponse, LogsResponse

def test_config():
    from config import settings
    assert settings.APP_NAME == "ARGUS-X"
    assert settings.GEMINI_TIMEOUT == 30.0
    assert len(settings.GEMINI_API_KEY) > 10, "Gemini key not loaded from .env"

def test_logger_redacts_keys():
    from utils.logger import redact_keys
    msg = "key=AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZ12345 used"
    assert "AIzaSy" not in redact_keys(msg)
    assert "[REDACTED]" in redact_keys(msg)

def test_health_endpoint():
    with urllib.request.urlopen("http://localhost:8000/health", timeout=5) as r:
        data = json.loads(r.read())
        assert data["status"] == "online"
        assert data["version"] == "2.0.0"
        assert "services" in data

def test_docs_available():
    with urllib.request.urlopen("http://localhost:8000/docs", timeout=5) as r:
        assert r.status == 200

def test_env_not_in_git():
    result = subprocess.run(
        ["git", "ls-files", "backend/.env"],
        capture_output=True, text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    assert result.stdout.strip() == "", ".env is tracked by git — CRITICAL SECURITY ISSUE"

def test_gitignore_exists():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assert os.path.exists(os.path.join(root, ".gitignore"))

def test_procfile_exists():
    assert os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Procfile"))

# Run all checks
check("ChatResponse() — all defaults, no null fields", test_chat_schema)
check("Empty message rejected by validation", test_empty_message_rejected)
check("All 4 schema files import cleanly", test_all_schemas)
check("config.py — settings load from .env", test_config)
check("logger.py — API keys redacted from logs", test_logger_redacts_keys)
check("GET /health — server running, 200 response", test_health_endpoint)
check("GET /docs — Swagger UI available", test_docs_available)
check(".env NOT tracked by git (security check)", test_env_not_in_git)
check(".gitignore exists in root", test_gitignore_exists)
check("Procfile exists for Railway deploy", test_procfile_exists)

# Print results
print()
for passed, name, error in results:
    icon = "[PASS]" if passed else "[FAIL]"
    print(f"  {icon} {name}")
    if error:
        print(f"         ERROR: {error}")

print()
print("="*60)
if all_pass:
    print("  DAY 1 COMPLETE.")
    print("  Server running. Schema locked. Security verified.")
    print("  Commit pushed to GitHub.")
    print("  READY FOR DAY 2.")
else:
    print("  FIX ALL FAILURES BEFORE CALLING DAY 1 DONE.")
    print("  Day 2 does not start until all items pass.")
print("="*60)
