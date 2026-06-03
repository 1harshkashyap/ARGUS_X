import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def run_tests():
    print("\n=== DAY 2 SEGMENT 1 VERIFICATION ===")
    all_pass = True

    # Test 1: Import
    try:
        from utils.llm import llm, detect_key_type, LLMResult
        print("  [PASS] utils/llm.py — imports correctly")
    except Exception as e:
        print(f"  [FAIL] utils/llm.py import — {e}")
        return False

    from utils.llm import llm, detect_key_type

    # Test 2: Key detection
    try:
        assert detect_key_type(None) == "NONE"
        assert detect_key_type("") == "NONE"
        assert detect_key_type("AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZ1234567") == "GEMINI"
        assert detect_key_type("sk-abcdefghijklmnopqrstuvwxyz1234567890123456") == "OPENAI"
        assert detect_key_type("invalid-key") == "UNKNOWN"
        assert detect_key_type("short") == "UNKNOWN"
        print("  [PASS] detect_key_type() — all formats detected correctly")
    except Exception as e:
        print(f"  [FAIL] detect_key_type() — {e}")
        all_pass = False

    # Test 3: Mock mode works (no key)
    try:
        result = await llm.generate("What is the leave policy?")
        # In dev with system key: GEMINI_FLASH or MOCK
        # In dev without system key: MOCK
        assert result.content != ""
        assert result.mode in ["GEMINI_FLASH", "MOCK", "ERROR"]
        assert result.latency_ms > 0
        print(f"  [PASS] llm.generate() — mode: {result.mode}, latency: {result.latency_ms}ms")
    except Exception as e:
        print(f"  [FAIL] llm.generate() — {e}")
        all_pass = False

    # Test 4: Invalid key returns error immediately
    try:
        result = await llm.generate("Hello", user_api_key="invalid-key-xyz")
        assert result.error == "INVALID_API_KEY"
        assert result.mode == "ERROR"
        print("  [PASS] Invalid key — returns INVALID_API_KEY error")
    except Exception as e:
        print(f"  [FAIL] Invalid key handling — {e}")
        all_pass = False

    # Test 5: No key in production returns API_KEY_REQUIRED
    try:
        from config import settings
        original_env = settings.ENVIRONMENT
        settings.ENVIRONMENT = "production"
        result = await llm.generate("Hello", user_api_key=None)
        settings.ENVIRONMENT = original_env
        assert result.error == "API_KEY_REQUIRED"
        print("  [PASS] No key in production — returns API_KEY_REQUIRED")
    except Exception as e:
        print(f"  [FAIL] Production key check — {e}")
        all_pass = False

    # Test 6: LLM does not log key values
    try:
        import logging
        import io
        from utils.logger import RedactingFormatter
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(RedactingFormatter())
        logging.getLogger("argus_x").addHandler(handler)
        fake_key = "AIzaSyTESTKEY123456789012345678901234567"
        await llm.generate("test", user_api_key=fake_key)
        output = stream.getvalue()
        logging.getLogger("argus_x").removeHandler(handler)
        assert "AIzaSy" not in output, "KEY FOUND IN LOGS — CRITICAL SECURITY FAILURE"
        print("  [PASS] Key security — API key NOT visible in logs")
    except AssertionError as e:
        print(f"  [FAIL] KEY SECURITY VIOLATION — {e}")
        all_pass = False
    except Exception as e:
        print(f"  [FAIL] Key logging test — {e}")
        all_pass = False

    print()
    print("SEGMENT 1 COMPLETE" if all_pass else "FIX FAILURES BEFORE SEGMENT 2")
    return all_pass

asyncio.run(run_tests())
