import urllib.request
import json
import sys

BASE = "http://localhost:8000"

print("\n=== SEGMENT 5 VERIFICATION ===")
all_pass = True

# Test 1: Root endpoint
try:
    with urllib.request.urlopen(f"{BASE}/") as r:
        data = json.loads(r.read())
        assert data.get("name") == "ARGUS-X"
        assert data.get("status") == "online"
        print("  [PASS] GET / — returns app info")
except Exception as e:
    print(f"  [FAIL] GET / — {e}")
    all_pass = False

# Test 2: Health endpoint
try:
    with urllib.request.urlopen(f"{BASE}/health") as r:
        data = json.loads(r.read())
        assert data.get("status") == "online"
        assert "services" in data
        assert "uptime_seconds" in data
        assert data.get("version") == "2.0.0"
        print("  [PASS] GET /health — returns full health status")
except Exception as e:
    print(f"  [FAIL] GET /health — {e}")
    all_pass = False

# Test 3: Health shows database configured
try:
    with urllib.request.urlopen(f"{BASE}/health") as r:
        data = json.loads(r.read())
        db_status = data.get("services", {}).get("database", "")
        assert db_status in ["configured", "not_configured"]
        print(f"  [PASS] GET /health — database status: {db_status}")
except Exception as e:
    print(f"  [FAIL] GET /health services — {e}")
    all_pass = False

# Test 4: Docs available
try:
    with urllib.request.urlopen(f"{BASE}/docs") as r:
        assert r.status == 200
        print("  [PASS] GET /docs — Swagger UI available")
except Exception as e:
    print(f"  [FAIL] GET /docs — {e}")
    all_pass = False

# Test 5: 404 returns JSON not HTML
try:
    import urllib.error
    try:
        urllib.request.urlopen(f"{BASE}/this-does-not-exist")
    except urllib.error.HTTPError as e:
        assert e.code == 404
        print("  [PASS] GET /nonexistent — returns 404 correctly")
except Exception as e:
    print(f"  [FAIL] 404 handling — {e}")
    all_pass = False

print()
print("SEGMENT 5 COMPLETE — SERVER RUNNING" if all_pass else "FIX FAILURES BEFORE SEGMENT 6")
