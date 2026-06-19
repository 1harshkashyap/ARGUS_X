"""Quick test: verify chat endpoint responds within timeout."""
import requests
import time

start = time.time()
r = requests.post(
    "http://localhost:8000/api/v1/chat",
    json={"message": "hello", "session_id": "test_quick"},
    timeout=30
)
elapsed = time.time() - start
d = r.json()
print(f"Time: {elapsed:.1f}s")
print(f"Status: {r.status_code}")
print(f"Mode: {d.get('llm_mode')}")
print(f"Blocked: {d.get('blocked')}")
print(f"Error: {d.get('error')}")
print(f"Response: {d.get('response', '')[:80]}")
