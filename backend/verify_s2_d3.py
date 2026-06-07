import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from security.xai_engine import xai_engine
from security.firewall import FirewallResult
from security.fingerprinter import FingerprintResult
from schemas.chat import XAIExplanation

print("\n=== DAY 3 SEGMENT 2 — XAI ENGINE VERIFICATION ===")
all_pass = True

def chk(name, passed, detail=""):
    global all_pass
    icon = "[PASS]" if passed else "[FAIL]"
    print(f"  {icon} {name}" + (f"\n         {detail}" if detail else ""))
    if not passed: all_pass = False

# 1. No args → valid object with 3 layers
r = xai_engine.explain()
chk("No args → valid XAIExplanation, 3 layers",
    isinstance(r, XAIExplanation) and len(r.layer_decisions) == 3)

# 2. Blocked → layer1 triggered, correct action
fr = FirewallResult(blocked=True, threat_type="PROMPT_INJECTION",
                    confidence=0.95, matched_rule="IGNORE_INSTRUCTIONS",
                    signals=["IGNORE_INSTRUCTIONS"])
fp_low = FingerprintResult(sophistication_score=2, sophistication_label="NAIVE")
r = xai_engine.explain(fr, None, fp_low, "LOW")
chk("Blocked, low score → action=BLOCK",
    r.layer_decisions[0].triggered and r.recommended_action == "BLOCK",
    f"action={r.recommended_action}")

# 3. Score ≥ 5 → BLOCK_AND_MONITOR
fp5 = FingerprintResult(sophistication_score=5, sophistication_label="INTERMEDIATE")
r = xai_engine.explain(fr, None, fp5, "LOW")
chk("Score 5 → BLOCK_AND_MONITOR",
    r.recommended_action == "BLOCK_AND_MONITOR",
    f"action={r.recommended_action}")

# 4. Score ≥ 8 → ESCALATE_MONITORING
fp8 = FingerprintResult(sophistication_score=8, sophistication_label="ADVANCED")
r = xai_engine.explain(fr, None, fp8, "LOW")
chk("Score 8 → ESCALATE_MONITORING",
    r.recommended_action == "ESCALATE_MONITORING",
    f"action={r.recommended_action}")

# 5. CRITICAL session → TERMINATE_SESSION
r = xai_engine.explain(fr, None, fp8, "CRITICAL")
chk("CRITICAL session → TERMINATE_SESSION",
    r.recommended_action == "TERMINATE_SESSION",
    f"action={r.recommended_action}")

# 6. HIGH session, low score → ESCALATE_MONITORING
fp_low2 = FingerprintResult(sophistication_score=2)
r = xai_engine.explain(fr, None, fp_low2, "HIGH")
chk("HIGH session → ESCALATE_MONITORING",
    r.recommended_action == "ESCALATE_MONITORING",
    f"action={r.recommended_action}")

# 7. Layer 3 confidence correct per session level
for level, expected_conf in [("CRITICAL",0.90),("HIGH",0.60),("MEDIUM",0.25),("LOW",0.03)]:
    r = xai_engine.explain(None, None, None, level)
    l3 = r.layer_decisions[2]
    chk(f"Layer 3 confidence for {level} = {expected_conf}",
        abs(l3.confidence - expected_conf) < 0.001,
        f"got {l3.confidence}")

# 8. Evolution note set for score ≥ 8
fp9 = FingerprintResult(sophistication_score=9, sophistication_label="APEX")
r = xai_engine.explain(fr, None, fp9, "LOW")
chk("Score ≥ 8 → evolution note non-empty",
    len(r.evolution_note) > 0, f"note={r.evolution_note[:60]!r}")

# 9. Invalid session level clamped to LOW
r = xai_engine.explain(None, None, None, "INVALID_LEVEL")
chk("Invalid session level → clamped, no crash",
    isinstance(r, XAIExplanation) and len(r.layer_decisions) == 3)

# 10. None inputs all around → safe default
for args in [
    (None, None, None, None),
    (None, None, None, "LOW"),
    (FirewallResult(), None, None, "LOW"),
]:
    try:
        res = xai_engine.explain(*args)
        assert len(res.layer_decisions) == 3
    except Exception as e:
        chk(f"None inputs crash", False, str(e)); break
else:
    chk("All None-input combinations → no crash, 3 layers", True)

print()
print("SEGMENT 2 COMPLETE — XAI ENGINE READY" if all_pass
      else "FIX FAILURES BEFORE SEGMENT 3")
