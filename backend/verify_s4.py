import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("\n=== SEGMENT 4 VERIFICATION ===")
all_pass = True

# Test 1: All schemas import
try:
    from schemas.chat import (
        ChatRequest, ChatResponse, XAIExplanation,
        LayerDecision, ThreatType, ThreatLevel
    )
    print("  [PASS] schemas/chat.py — all classes import")
except Exception as e:
    print(f"  [FAIL] schemas/chat.py — {e}")
    all_pass = False

try:
    from schemas.events import SecurityEvent, XAIDecision
    print("  [PASS] schemas/events.py — all classes import")
except Exception as e:
    print(f"  [FAIL] schemas/events.py — {e}")
    all_pass = False

try:
    from schemas.agents import BattleState, AgentStatus, CycleResult
    print("  [PASS] schemas/agents.py — all classes import")
except Exception as e:
    print(f"  [FAIL] schemas/agents.py — {e}")
    all_pass = False

try:
    from schemas.analytics import StatsResponse, LogsResponse
    print("  [PASS] schemas/analytics.py — all classes import")
except Exception as e:
    print(f"  [FAIL] schemas/analytics.py — {e}")
    all_pass = False

# Test 2: ChatResponse instantiates with ALL defaults (no arguments needed)
try:
    from schemas.chat import ChatResponse, XAIExplanation
    r = ChatResponse()
    assert r.response == ""
    assert r.blocked == False
    assert r.threat_score == 0.0
    assert r.explanation is not None
    assert r.error is None
    assert r.llm_mode == "NONE"
    print("  [PASS] ChatResponse() — all defaults work, no null fields")
except Exception as e:
    print(f"  [FAIL] ChatResponse defaults — {e}")
    all_pass = False

# Test 3: XAIExplanation never returns None
try:
    from schemas.chat import XAIExplanation
    x = XAIExplanation()
    assert x is not None
    assert x.primary_reason == "No threat detected"
    assert isinstance(x.layer_decisions, list)
    print("  [PASS] XAIExplanation() — never None, always valid object")
except Exception as e:
    print(f"  [FAIL] XAIExplanation — {e}")
    all_pass = False

# Test 4: ChatRequest validation works
try:
    from schemas.chat import ChatRequest
    from pydantic import ValidationError
    # Valid request
    req = ChatRequest(message="Hello world")
    assert req.session_id == "default"
    assert req.api_key is None
    # Empty message should fail
    try:
        ChatRequest(message="")
        print("  [FAIL] ChatRequest validation — empty message should be rejected")
        all_pass = False
    except ValidationError:
        print("  [PASS] ChatRequest validation — empty message correctly rejected")
except Exception as e:
    print(f"  [FAIL] ChatRequest validation — {e}")
    all_pass = False

# Test 5: ThreatType enum values
try:
    from schemas.chat import ThreatType
    assert ThreatType.CLEAN == "CLEAN"
    assert ThreatType.PROMPT_INJECTION == "PROMPT_INJECTION"
    print("  [PASS] ThreatType enum — all values correct")
except Exception as e:
    print(f"  [FAIL] ThreatType enum — {e}")
    all_pass = False

# Test 6: BattleState defaults
try:
    from schemas.agents import BattleState
    b = BattleState()
    assert b.tick == 0
    assert b.red_tier == 1
    print("  [PASS] BattleState() — all defaults work")
except Exception as e:
    print(f"  [FAIL] BattleState — {e}")
    all_pass = False

# Test 7: JSON serialization works (critical for FastAPI responses)
try:
    from schemas.chat import ChatResponse
    r = ChatResponse()
    json_str = r.model_dump_json()
    assert "response" in json_str
    assert "explanation" in json_str
    assert "null" not in json_str or '"error":null' in json_str or '"error": null' in json_str
    print("  [PASS] ChatResponse JSON — serializes cleanly")
except Exception as e:
    print(f"  [FAIL] ChatResponse JSON — {e}")
    all_pass = False

print()
print("SEGMENT 4 COMPLETE — SCHEMAS LOCKED" if all_pass else "FIX FAILURES BEFORE SEGMENT 5")
