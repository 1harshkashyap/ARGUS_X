import sys

packages = [
    ("fastapi",          "fastapi"),
    ("uvicorn",          "uvicorn"),
    ("pydantic",         "pydantic"),
    ("pydantic-settings","pydantic_settings"),
    ("python-dotenv",    "dotenv"),
    ("google-generativeai", "google.generativeai"),
    ("openai",           "openai"),
    ("supabase",         "supabase"),
    ("onnxruntime",      "onnxruntime"),
    ("numpy",            "numpy"),
    ("httpx",            "httpx"),
    ("aiofiles",         "aiofiles"),
    ("tokenizers",       "tokenizers"),
    ("python-multipart", "multipart"),
]

print("\n=== SEGMENT 2 VERIFICATION ===")
all_pass = True
for display, import_name in packages:
    try:
        __import__(import_name)
        print(f"  [PASS] {display}")
    except ImportError as e:
        print(f"  [FAIL] {display} — {e}")
        all_pass = False

print()
if all_pass:
    print("SEGMENT 2 COMPLETE — all packages installed")
else:
    print("FIX FAILED PACKAGES before Segment 3")
