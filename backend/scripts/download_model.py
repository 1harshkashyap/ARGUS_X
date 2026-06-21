"""
ARGUS-X — Download ML Model for Production Deployment
Run: python scripts/download_model.py

Downloads the ProtectAI DeBERTa v3 ONNX model + tokenizer from HuggingFace.
Lightweight — no torch, no transformers, no optimum required.

Uses huggingface_hub (pip install huggingface-hub) for reliable downloading
with resume support and hash verification.
"""

import sys
import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BACKEND_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# HuggingFace repo with ONNX export
REPO_ID = "ProtectAI/deberta-v3-base-prompt-injection-v2"

# Files we need — the ONNX model + tokenizer files
FILES_TO_DOWNLOAD = {
    "onnx/model.onnx": "security_classifier.onnx",
    "tokenizer.json": "tokenizer.json",
    "tokenizer_config.json": "tokenizer_config.json",
    "special_tokens_map.json": "special_tokens_map.json",
}


def download_with_hub():
    """Download using huggingface_hub (preferred — has resume + hash check)."""
    from huggingface_hub import hf_hub_download  # type: ignore

    for remote_path, local_name in FILES_TO_DOWNLOAD.items():
        dest = MODELS_DIR / local_name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  [SKIP] {local_name} already exists ({dest.stat().st_size:,} bytes)")
            continue

        print(f"  Downloading {remote_path} → {local_name} ...")
        downloaded = hf_hub_download(
            repo_id=REPO_ID,
            filename=remote_path,
            local_dir=MODELS_DIR / "_hf_tmp",
            local_dir_use_symlinks=False,
        )
        # Move from HF cache structure to flat models/ dir
        import shutil
        shutil.move(str(downloaded), str(dest))
        print(f"  [OK] {local_name} ({dest.stat().st_size:,} bytes)")

    # Cleanup temp dir
    tmp = MODELS_DIR / "_hf_tmp"
    if tmp.exists():
        import shutil
        shutil.rmtree(str(tmp), ignore_errors=True)


def download_with_urllib():
    """Fallback: download via direct HTTP (no resume, no hash check)."""
    import urllib.request
    import json

    # HuggingFace raw file URL pattern
    BASE_URL = f"https://huggingface.co/{REPO_ID}/resolve/main"

    for remote_path, local_name in FILES_TO_DOWNLOAD.items():
        dest = MODELS_DIR / local_name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  [SKIP] {local_name} already exists ({dest.stat().st_size:,} bytes)")
            continue

        url = f"{BASE_URL}/{remote_path}"
        print(f"  Downloading {url} ...")
        try:
            urllib.request.urlretrieve(url, str(dest))
            print(f"  [OK] {local_name} ({dest.stat().st_size:,} bytes)")
        except Exception as e:
            print(f"  [FAIL] {local_name}: {e}")
            # Remove partial file
            if dest.exists():
                dest.unlink()
            raise


def validate():
    """Validate that model + tokenizer files exist and are non-empty."""
    print("\nValidating...")
    model_path = MODELS_DIR / "security_classifier.onnx"
    tokenizer_path = MODELS_DIR / "tokenizer.json"

    if not model_path.exists():
        print("  [FAIL] security_classifier.onnx not found")
        return False
    if model_path.stat().st_size < 1_000_000:  # Model should be >100MB
        print(f"  [FAIL] security_classifier.onnx too small ({model_path.stat().st_size} bytes)")
        return False

    if not tokenizer_path.exists():
        print("  [FAIL] tokenizer.json not found")
        return False

    print(f"  Model:     {model_path.stat().st_size:,} bytes ✓")
    print(f"  Tokenizer: {tokenizer_path.stat().st_size:,} bytes ✓")

    # Optional: test ONNX load
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        inputs = [i.name for i in sess.get_inputs()]
        print(f"  ONNX inputs: {inputs} ✓")
        return True
    except ImportError:
        print("  [SKIP] onnxruntime not installed — skipping load test")
        return True
    except Exception as e:
        print(f"  [FAIL] ONNX load failed: {e}")
        return False


def main():
    print("\n" + "=" * 60)
    print("   ARGUS-X — Download ML Model")
    print("=" * 60)
    print(f"\n  Repo:   {REPO_ID}")
    print(f"  Output: {MODELS_DIR}\n")

    # Try huggingface_hub first, fall back to urllib
    try:
        import huggingface_hub  # type: ignore
        print("Using huggingface_hub for download (resume + hash check)\n")
        download_with_hub()
    except ImportError:
        print("huggingface_hub not installed — using direct HTTP download\n")
        print("  (For better reliability: pip install huggingface-hub)\n")
        download_with_urllib()

    if validate():
        print("\n" + "=" * 60)
        print("  MODEL DOWNLOAD COMPLETE ✓")
        print("=" * 60 + "\n")
    else:
        print("\n  MODEL DOWNLOAD FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
