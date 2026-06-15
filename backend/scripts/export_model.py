"""
ARGUS-X — ProtectAI DeBERTa ONNX Exporter
Run ONCE: python scripts/export_model.py
Requires: pip install optimum[exporters] transformers torch

After this script succeeds, uninstall those packages:
  pip uninstall optimum transformers torch -y

Runtime requires only: onnxruntime + tokenizers
"""

import sys
import shutil
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR  = BACKEND_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID  = "ProtectAI/deberta-v3-base-prompt-injection-v2"
ONNX_NAME = "security_classifier.onnx"


def export():
    print("\n" + "=" * 60)
    print("   ARGUS-X — ProtectAI DeBERTa ONNX Export")
    print("=" * 60)
    print(f"\n  Model:  {MODEL_ID}")
    print(f"  Output: {MODELS_DIR}\n")

    try:
        from optimum.exporters.onnx import main_export
    except ImportError:
        print("[FAIL] optimum not installed.")
        print("Run: pip install optimum[exporters] transformers torch")
        sys.exit(1)

    # Export to a temporary directory first
    tmp_dir = MODELS_DIR / "_export_tmp"
    tmp_dir.mkdir(exist_ok=True)

    try:
        print("Exporting to ONNX (downloads ~180MB on first run)...")
        main_export(
            model_name_or_path=MODEL_ID,
            output=tmp_dir,
            task="text-classification",
            opset=14,
        )
        print("Export complete.")
    except Exception as e:
        print(f"[FAIL] Export error: {e}")
        # Clean up temp dir on failure
        shutil.rmtree(str(tmp_dir), ignore_errors=True)
        sys.exit(1)

    # Rename model.onnx → security_classifier.onnx
    src = tmp_dir / "model.onnx"
    dst = MODELS_DIR / ONNX_NAME
    if src.exists():
        shutil.move(str(src), str(dst))
        print(f"  Model → {dst}")
    else:
        print(f"[FAIL] model.onnx not found in {tmp_dir}")
        shutil.rmtree(str(tmp_dir), ignore_errors=True)
        sys.exit(1)

    # Copy tokenizer files
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]
    for fname in tokenizer_files:
        src_f = tmp_dir / fname
        if src_f.exists():
            shutil.copy2(str(src_f), str(MODELS_DIR / fname))
            print(f"  {fname} → models/")
        else:
            print(f"  [WARN] {fname} not found — tokenizer may need it")

    # SPM model (DeBERTa v3 uses SentencePiece)
    for spm in tmp_dir.glob("*.model"):
        shutil.copy2(str(spm), str(MODELS_DIR / spm.name))
        print(f"  {spm.name} → models/")

    # Cleanup temp dir
    shutil.rmtree(str(tmp_dir), ignore_errors=True)

    # Validate ONNX file
    print("\nValidating exported ONNX model...")
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(
            str(MODELS_DIR / ONNX_NAME),
            providers=["CPUExecutionProvider"],
        )
        inputs = [i.name for i in sess.get_inputs()]
        outputs = [o.name for o in sess.get_outputs()]
        print(f"  Inputs:  {inputs}")
        print(f"  Outputs: {outputs}")

        # Must have exactly 2 output classes
        out_shape = sess.get_outputs()[0].shape
        assert out_shape[-1] == 2, f"Expected 2 classes, got {out_shape}"
        print(f"  Classes: 2 (index 0=SAFE, index 1=INJECTION) ✓")
    except Exception as e:
        print(f"[FAIL] ONNX validation: {e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  EXPORT COMPLETE.")
    print()
    print("  Next: uninstall export packages:")
    print("    pip uninstall optimum transformers torch -y")
    print()
    print("  Then: update backend/security/ml_classifier.py")
    print("  Then: set ML_ENABLED=true in backend/.env")
    print("=" * 60)


if __name__ == "__main__":
    export()
