import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("\n=== DAY 2 SEGMENT 3 VERIFICATION ===")
all_pass = True

try:
    from utils.session import session_tracker, SessionTracker
    print("  [PASS] utils/session.py — imports correctly")
except Exception as e:
    print(f"  [FAIL] Import — {e}")
    exit(1)

# Test 1: New session starts at LOW
try:
    level = session_tracker.get_level("brand-new-session")
    assert level == "LOW"
    print("  [PASS] New session — starts at LOW")
except Exception as e:
    print(f"  [FAIL] New session level — {e}")
    all_pass = False

# Test 2: Clean requests stay at LOW
try:
    for i in range(10):
        session_tracker.update("clean-session", was_threat=False)
    level = session_tracker.get_level("clean-session")
    assert level == "LOW"
    print("  [PASS] Clean session — stays at LOW after 10 clean requests")
except Exception as e:
    print(f"  [FAIL] Clean session — {e}")
    all_pass = False

# Test 3: Escalates to MEDIUM at 25% threats
try:
    tracker = SessionTracker()
    for _ in range(3): tracker.update("s1", was_threat=False)
    tracker.update("s1", was_threat=True)  # 1/4 = 25%
    level = tracker.get_level("s1")
    assert level == "MEDIUM"
    print("  [PASS] Session escalates to MEDIUM at 25% threat ratio")
except Exception as e:
    print(f"  [FAIL] MEDIUM escalation — {e}")
    all_pass = False

# Test 4: Escalates to HIGH at 50%
try:
    tracker = SessionTracker()
    tracker.update("s2", was_threat=True)
    tracker.update("s2", was_threat=True)
    tracker.update("s2", was_threat=False)
    tracker.update("s2", was_threat=False)  # 2/4 = 50%
    level = tracker.get_level("s2")
    assert level == "HIGH"
    print("  [PASS] Session escalates to HIGH at 50% threat ratio")
except Exception as e:
    print(f"  [FAIL] HIGH escalation — {e}")
    all_pass = False

# Test 5: Escalates to CRITICAL at 75%
try:
    tracker = SessionTracker()
    for _ in range(3): tracker.update("s3", was_threat=True)
    tracker.update("s3", was_threat=False)  # 3/4 = 75%
    level = tracker.get_level("s3")
    assert level == "CRITICAL"
    print("  [PASS] Session escalates to CRITICAL at 75% threat ratio")
except Exception as e:
    print(f"  [FAIL] CRITICAL escalation — {e}")
    all_pass = False

# Test 6: get_level never returns None
try:
    for sid in [None, "", "valid-session", "unknown-session-xyz"]:
        level = session_tracker.get_level(sid)
        assert level in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
    print("  [PASS] get_level() — always returns valid string, never None")
except Exception as e:
    print(f"  [FAIL] get_level None safety — {e}")
    all_pass = False

print()
print("SEGMENT 3 COMPLETE" if all_pass else "FIX FAILURES BEFORE SEGMENT 4")
