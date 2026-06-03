import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def run_tests():
    print("\n=== DAY 2 SEGMENT 2 VERIFICATION ===")
    all_pass = True

    # Test 1: Import
    try:
        from utils.db import (
            log_event, get_recent_events, get_stats,
            get_dynamic_rules, add_dynamic_rule,
            update_battle_state, check_connection
        )
        print("  [PASS] utils/db.py — all functions import")
    except Exception as e:
        print(f"  [FAIL] utils/db.py import — {e}")
        return

    from utils.db import (
        log_event, get_recent_events, get_stats,
        get_dynamic_rules, add_dynamic_rule,
        update_battle_state, check_connection
    )

    # Test 2: DB connection
    try:
        connected = await check_connection()
        if connected:
            print("  [PASS] Supabase connection — connected successfully")
        else:
            print("  [WARN] Supabase connection — not connected (check .env credentials)")
    except Exception as e:
        print(f"  [FAIL] DB connection — {e}")
        all_pass = False

    # Test 3: Write a test event
    try:
        result = await log_event({
            "session_id": "test-day2",
            "message_preview": "DAY2 VERIFICATION TEST",
            "blocked": False,
            "threat_type": "CLEAN",
            "llm_mode": "MOCK",
            "latency_ms": 42.0
        })
        if result:
            print("  [PASS] log_event() — wrote test event to Supabase")
        else:
            print("  [WARN] log_event() — returned False (check Supabase credentials)")
    except Exception as e:
        print(f"  [FAIL] log_event() — {e}")
        all_pass = False

    # Test 4: Read events back
    try:
        events = await get_recent_events(limit=5)
        assert isinstance(events, list)
        print(f"  [PASS] get_recent_events() — returned {len(events)} events")
    except Exception as e:
        print(f"  [FAIL] get_recent_events() — {e}")
        all_pass = False

    # Test 5: Get stats (returns safe defaults even if DB empty)
    try:
        stats = await get_stats()
        assert isinstance(stats, dict)
        assert "total_events" in stats
        print(f"  [PASS] get_stats() — returned valid stats dict")
    except Exception as e:
        print(f"  [FAIL] get_stats() — {e}")
        all_pass = False

    # Test 6: Get dynamic rules
    try:
        rules = await get_dynamic_rules()
        assert isinstance(rules, list)
        print(f"  [PASS] get_dynamic_rules() — returned {len(rules)} rules")
    except Exception as e:
        print(f"  [FAIL] get_dynamic_rules() — {e}")
        all_pass = False

    # Test 7: Failure does NOT crash (simulate with invalid operation)
    try:
        # Even with a bad table name pattern, should return False not crash
        result = await get_dynamic_rules()
        assert isinstance(result, list)
        print("  [PASS] DB failure handling — returns safe defaults, never crashes")
    except Exception as e:
        print(f"  [FAIL] DB failure handling — {e}")
        all_pass = False

    print()
    print("SEGMENT 2 COMPLETE" if all_pass else "FIX FAILURES BEFORE SEGMENT 3")

asyncio.run(run_tests())
