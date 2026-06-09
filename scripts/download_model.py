"""
ARGUS-X Model Downloader
Downloads quantized DistilBERT ONNX model + tokenizer files.
Run once: python scripts/download_model.py
No PyTorch. No transformers. Pure urllib.
"""

import sys
import os
import urllib.request
import urllib.error
from pathlib import Path

# ── Target directory ──────────────────────────────────────────────────
BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
MODELS_DIR  = BACKEND_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Files to download ─────────────────────────────────────────────────
# Source: Xenova/distilbert-base-uncased-finetuned-sst-2-english
# Quantized model (q8): ~32MB — fast CPU inference
BASE_URL = (
    "https://huggingface.co/"
    "Xenova/distilbert-base-uncased-finetuned-sst-2-english"
    "/resolve/main"
)

DOWNLOADS = [
    {
        "url":      f"{BASE_URL}/onnx/model_quantized.onnx",
        "filename": "security_classifier.onnx",
        "desc":     "DistilBERT quantized ONNX model (~32MB)",
    },
    {
        "url":      f"{BASE_URL}/tokenizer.json",
        "filename": "tokenizer.json",
        "desc":     "Tokenizer vocabulary and rules (~450KB)",
    },
    {
        "url":      f"{BASE_URL}/tokenizer_config.json",
        "filename": "tokenizer_config.json",
        "desc":     "Tokenizer configuration",
    },
]


def download_file(url: str, dest: Path, desc: str) -> bool:
    """
    Download a file with progress reporting.
    Returns True on success, False on failure.
    """
    if dest.exists():
        size_mb = dest.stat().st_size / 1024 / 1024
        print(f"  [SKIP] {dest.name} already exists ({size_mb:.1f}MB)")
        return True

    print(f"  [DOWN] {desc}")
    print(f"         URL: {url}")
    print(f"         Dest: {dest}")

    try:
        headers = {
            "User-Agent": "ARGUS-X/2.0 (model-downloader)"
        }
        req = urllib.request.Request(url, headers=headers)

        with urllib.request.urlopen(req, timeout=120) as response:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 65536  # 64KB chunks

            with open(dest, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total > 0:
                        pct = downloaded / total * 100
                        mb  = downloaded / 1024 / 1024
                        print(
                            f"\r         {pct:5.1f}% ({mb:.1f}MB)",
                            end="",
                            flush=True,
                        )

        print(f"\r         Complete: {downloaded / 1024 / 1024:.1f}MB      ")
        return True

    except urllib.error.URLError as e:
        print(f"\n  [FAIL] Network error: {e}")
        # Clean up partial download to prevent corrupted files
        if dest.exists():
            dest.unlink()
        return False
    except OSError as e:
        print(f"\n  [FAIL] IO error: {type(e).__name__}: {e}")
        if dest.exists():
            dest.unlink()
        return False
    except Exception as e:
        print(f"\n  [FAIL] Download error: {type(e).__name__}: {e}")
        if dest.exists():
            dest.unlink()
        return False


def verify_onnx(model_path: Path) -> bool:
    """Quick sanity check that the ONNX file is valid."""
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        inputs  = [i.name for i in session.get_inputs()]
        outputs = [o.name for o in session.get_outputs()]
        print(f"  [OK]   ONNX model valid — inputs: {inputs} outputs: {outputs}")
        return True
    except ImportError:
        print("  [FAIL] onnxruntime not installed. Run: pip install onnxruntime")
        return False
    except Exception as e:
        print(f"  [FAIL] ONNX validation failed: {e}")
        return False


def verify_tokenizer(tokenizer_path: Path) -> bool:
    """Quick sanity check that the tokenizer works."""
    try:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(tokenizer_path))
        enc = tok.encode("Test message for tokenizer verification")
        print(f"  [OK]   Tokenizer valid — test encode: {len(enc.ids)} tokens")
        return True
    except ImportError:
        print("  [FAIL] tokenizers not installed. Run: pip install tokenizers")
        return False
    except Exception as e:
        print(f"  [FAIL] Tokenizer validation failed: {e}")
        return False


def main() -> None:
    print("\n" + "=" * 62)
    print("   ARGUS-X — Model Downloader")
    print("   DistilBERT quantized ONNX (CPU-only, no PyTorch)")
    print("=" * 62)
    print(f"\n  Target directory: {MODELS_DIR}\n")

    success_count = 0
    for item in DOWNLOADS:
        dest = MODELS_DIR / item["filename"]
        ok = download_file(item["url"], dest, item["desc"])
        if ok:
            success_count += 1
        print()

    print("=" * 62)
    if success_count == len(DOWNLOADS):
        print("  All files downloaded. Running validation...")
        print()

        model_path = MODELS_DIR / "security_classifier.onnx"
        tok_path   = MODELS_DIR / "tokenizer.json"

        onnx_ok = verify_onnx(model_path)
        tok_ok  = verify_tokenizer(tok_path)

        print()
        if onnx_ok and tok_ok:
            print("  [SUCCESS] Model ready.")
            print(f"  Model path: {model_path}")
            print()
            print("  Next step: set in backend/.env:")
            print("    ONNX_MODEL_PATH=models/security_classifier.onnx")
            print("    ML_ENABLED=true")
            print("  Then restart the server.")
        else:
            print("  [WARNING] Some validations failed. Check errors above.")
            sys.exit(1)
    else:
        print(f"  {success_count}/{len(DOWNLOADS)} files downloaded.")
        print("  Fix network issues and re-run: python scripts/download_model.py")
        sys.exit(1)

    print("=" * 62)


if __name__ == "__main__":
    main()
