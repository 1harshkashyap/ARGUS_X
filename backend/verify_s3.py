import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("\n=== SEGMENT 3 VERIFICATION ===")
all_pass = True

# Test 1: config.py imports
try:
    from config import settings
    assert settings.APP_NAME == "ARGUS-X"
    assert settings.APP_VERSION == "2.0.0"
    assert settings.GEMINI_TIMEOUT == 30.0
    print("  [PASS] config.py — settings loads correctly")
except Exception as e:
    print(f"  [FAIL] config.py — {e}")
    all_pass = False

# Test 2: Settings has Gemini key
try:
    from config import settings
    has_key = len(settings.GEMINI_API_KEY) > 10
    if has_key:
        print("  [PASS] config.py — Gemini API key loaded from .env")
    else:
        print("  [WARN] config.py — Gemini key empty (check .env file)")
except Exception as e:
    print(f"  [FAIL] config.py key check — {e}")
    all_pass = False

# Test 3: logger.py imports
try:
    from utils.logger import logger, redact_keys
    print("  [PASS] utils/logger.py — imports correctly")
except Exception as e:
    print(f"  [FAIL] utils/logger.py — {e}")
    all_pass = False

# Test 4: Key redaction works
try:
    from utils.logger import redact_keys
    test_msg = "Using key AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZ12345 for request"
    redacted = redact_keys(test_msg)
    assert "AIzaSy" not in redacted
    assert "[REDACTED]" in redacted
    print("  [PASS] utils/logger.py — key redaction works correctly")
except Exception as e:
    print(f"  [FAIL] utils/logger.py redaction — {e}")
    all_pass = False

# Test 5: Logger does not expose keys
try:
    from utils.logger import logger
    import logging
    import io
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    from utils.logger import RedactingFormatter
    handler.setFormatter(RedactingFormatter())
    logger.addHandler(handler)
    logger.info("Test with key AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZ12345")
    output = stream.getvalue()
    logger.removeHandler(handler)
    assert "AIzaSy" not in output
    print("  [PASS] utils/logger.py — logger output is safe, no keys visible")
except Exception as e:
    print(f"  [FAIL] utils/logger.py logger safety — {e}")
    all_pass = False

print()
print("SEGMENT 3 COMPLETE" if all_pass else "FIX FAILURES BEFORE SEGMENT 4")
