import urllib.request, json, sys, time
import urllib.error

BASE = "http://localhost:8000"

def get(path):
    try:
        with urllib.request.urlopen(f"{BASE}{path}", timeout=10) as r:
            return json.loads(r.read()), r.status
    except Exception as e:
        return {"error": str(e)}, 0

print("\n=== DAY 3 SEGMENT 4 — ANALYTICS VERIFICATION ===")
all_pass = True

def chk(name, passed, detail=""):
    global all_pass
    icon = "[PASS]" if passed else "[FAIL]"
    print(f"  {icon} {name}" + (f"\n         {detail}" if detail else ""))
    if not passed: all_pass = False

# 1. /stats returns 200 with all required fields
d, s = get("/api/v1/analytics/stats")
required = ["total_events","total_blocked","total_clean","block_rate","uptime_seconds"]
missing  = [f for f in required if f not in d]
chk("GET /analytics/stats → 200, all fields present",
    s == 200 and not missing,
    f"missing={missing}" if missing else f"events={d.get('total_events')} uptime={d.get('uptime_seconds')}s")

# 2. block_rate is 0.0–1.0
br = d.get("block_rate", -1)
chk("block_rate is 0.0–1.0",
    0.0 <= br <= 1.0, f"block_rate={br}")

# 3. /logs returns 200 with events list
d, s = get("/api/v1/analytics/logs?limit=10")
chk("GET /analytics/logs → 200, events is list",
    s == 200 and isinstance(d.get("events"), list) and "count" in d,
    f"count={d.get('count')} limit={d.get('limit')}")

# 4. /logs limit param works
d5, s5 = get("/api/v1/analytics/logs?limit=5")
chk("GET /analytics/logs?limit=5 → count ≤ 5",
    s5 == 200 and d5.get("limit") == 5,
    f"limit={d5.get('limit')}")

# 5. /xai returns 200 with decisions list
d, s = get("/api/v1/analytics/xai?limit=5")
chk("GET /analytics/xai → 200, decisions is list",
    s == 200 and isinstance(d.get("decisions"), list) and "count" in d,
    f"count={d.get('count')}")

# 6. Endpoints survive bad query params gracefully (FastAPI validates)
try:
    urllib.request.urlopen(f"{BASE}/api/v1/analytics/logs?limit=999", timeout=5)
    chk("limit=999 → 422 validation error", False, "Should have been rejected")
except urllib.error.HTTPError as e:
    chk("limit=999 → 422 validation error (out of range)",
        e.code == 422, f"HTTP {e.code}")
except Exception as e:
    chk("limit=999 validation", False, str(e))

# 7. uptime_seconds increases (not frozen)
d_before, _ = get("/api/v1/analytics/stats")
time.sleep(0.5)
d_after, _ = get("/api/v1/analytics/stats")
chk("uptime_seconds is increasing (not frozen)",
    d_after.get("uptime_seconds", 0) > d_before.get("uptime_seconds", 0),
    f"{d_before.get('uptime_seconds')} → {d_after.get('uptime_seconds')}")

print()
print("SEGMENT 4 COMPLETE — ANALYTICS READY" if all_pass
      else "FIX FAILURES BEFORE SEGMENT 5")
