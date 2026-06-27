"""
ARGUS-X ML Security Classifier

ONNX-based classifier for prompt injection detection.
CPU only. No PyTorch. No transformers. Uses tokenizers library directly.

Model: ProtectAI DeBERTa v3 (purpose-built prompt injection classifier)
Strategy: INJECTION class probability = threat probability.
          Class 0 = SAFE, Class 1 = INJECTION.

Security guarantees:
  - Never raises to caller
  - Fail-closed on any error
  - Input length hard-capped
  - Graceful degradation when model is missing
"""

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from config import settings
from utils.logger import logger


# ── Result dataclass ──────────────────────────────────────────────────


@dataclass
class MLResult:
    """
    Result of ML classifier inference.

    triggered:   True if threat probability exceeds threshold
    confidence:  Threat probability 0.0–1.0
    method:      ONNX | UNAVAILABLE | ERROR
    latency_ms:  Inference time in milliseconds
    """

    triggered: bool = False
    confidence: float = 0.0
    method: str = "UNAVAILABLE"
    latency_ms: float = 0.0


# ── Constants ─────────────────────────────────────────────────────────

# Maximum tokens to pass to model (DeBERTa v3 max = 512)
_MAX_TOKENS = 512

# Maximum input string length before truncation (defense against huge payloads)
_MAX_INPUT_LEN = 2048


# ── Softmax (numerically stable) ─────────────────────────────────────


def _softmax(logits: List[float]) -> List[float]:
    """Numerically stable softmax — subtracts max to prevent overflow."""
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]


# ── ML Classifier ─────────────────────────────────────────────────────


class MLClassifier:
    """
    ONNX-based ML security classifier.

    Uses ProtectAI DeBERTa v3 (purpose-built prompt injection classifier)
    for threat detection.

    Model inputs  (DeBERTa v3 format):
      input_ids:       [1, seq_len] int64
      attention_mask:  [1, seq_len] int64

    Model outputs:
      logits: [1, 2] float32
        logits[0][0] = SAFE score
        logits[0][1] = INJECTION (malicious) score

    Threat probability = softmax(logits)[1]  (INJECTION class probability)

    Security properties:
      - CPU only (CPUExecutionProvider — no CUDA dependency)
      - Input length guarded at _MAX_INPUT_LEN
      - Token count capped at _MAX_TOKENS (model max)
      - If model missing → available=False, all calls return UNAVAILABLE
      - Fail-closed: any inference error → MLResult(method=ERROR)
      - Never raises to caller
    """

    def __init__(self) -> None:
        self.available: bool = False
        self._session: Optional[object] = None
        self._tokenizer: Optional[object] = None
        self._model_path: Optional[Path] = None
        self._load_model()

    def _load_model(self) -> None:
        """
        Attempt to load ONNX session + tokenizer.
        Sets self.available = True only if BOTH load successfully.
        Non-fatal if either fails.
        """
        if not settings.ML_ENABLED:
            logger.info("MLClassifier: disabled (ML_ENABLED=false)")
            return

        # Resolve model path
        model_path_str = settings.ONNX_MODEL_PATH
        if not os.path.isabs(model_path_str):
            # Relative path: resolve from backend directory
            backend_dir = Path(__file__).resolve().parent.parent
            model_path = backend_dir / model_path_str
        else:
            model_path = Path(model_path_str)

        tokenizer_path = model_path.parent / "tokenizer.json"

        # Check files exist
        if not model_path.exists():
            logger.warning(
                f"MLClassifier: model not found at {model_path} — "
                f"run: python scripts/export_model.py"
            )
            return

        if not tokenizer_path.exists():
            logger.warning(
                f"MLClassifier: tokenizer.json not found at {tokenizer_path}"
            )
            return

        # Load ONNX session
        try:
            import onnxruntime as ort  # type: ignore[import-untyped]

            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            session_options.intra_op_num_threads = 2
            session_options.log_severity_level = 3  # Suppress verbose ONNX logs

            self._session = ort.InferenceSession(
                str(model_path),
                sess_options=session_options,
                providers=["CPUExecutionProvider"],
            )
            self._model_path = model_path
            logger.info(
                f"MLClassifier: ProtectAI DeBERTa loaded from {model_path.name}"
            )
        except Exception as e:
            logger.warning(
                f"MLClassifier: ONNX session load failed: "
                f"{type(e).__name__}: {e}"
            )
            return

        # Load tokenizer
        try:
            from tokenizers import Tokenizer  # type: ignore[import-untyped]

            self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
            self._tokenizer.enable_truncation(max_length=_MAX_TOKENS)
            self._tokenizer.enable_padding(
                pad_id=0,
                pad_token="[PAD]",
                length=None,  # Dynamic padding — no fixed length
            )

            # Validate tokenizer produces sane encodings on representative input
            try:
                _test_enc = self._tokenizer.encode(
                    "What is the annual leave policy?"
                )
                assert len(_test_enc.ids) > 0, "Tokenizer produced empty encoding"
                assert len(_test_enc.ids) < _MAX_TOKENS, "Test encoding too long"
            except Exception as e:
                logger.warning(f"MLClassifier: tokenizer sanity check failed: {e}")
                self._session = None
                return

            logger.info("MLClassifier: tokenizer loaded")
        except Exception as e:
            logger.warning(
                f"MLClassifier: tokenizer load failed: "
                f"{type(e).__name__}: {e}"
            )
            # If tokenizer fails, session is useless — clean up
            self._session = None
            return

        self.available = True
        logger.info(
            f"MLClassifier: ProtectAI DeBERTa online — "
            f"INJECTION classifier active (threshold={settings.FIREWALL_ML_THRESHOLD})"
        )

    def classify(self, text: str) -> MLResult:
        """
        Classify text as threat or benign.

        Args:
            text: Input message to classify

        Returns:
            MLResult — always. Never raises.
        """
        start = time.monotonic()

        # Unavailable path
        if (
            not self.available
            or self._session is None
            or self._tokenizer is None
        ):
            return MLResult(
                method="UNAVAILABLE",
                latency_ms=round((time.monotonic() - start) * 1000, 2),
            )

        try:
            import numpy as np  # type: ignore[import-untyped]

            # Guard input length — hard truncation before tokenization
            safe_text = (text or "")[:_MAX_INPUT_LEN]
            if not safe_text.strip():
                return MLResult(
                    method="ONNX",
                    latency_ms=round((time.monotonic() - start) * 1000, 2),
                )

            # Tokenize
            encoding = self._tokenizer.encode(safe_text)

            # Build numpy arrays — model expects int64
            input_ids = np.array([encoding.ids], dtype=np.int64)
            attention_mask = np.array(
                [encoding.attention_mask], dtype=np.int64
            )

            # Clamp to max tokens (safety guard — tokenizer truncation
            # should already handle this, but belt-and-suspenders)
            if input_ids.shape[1] > _MAX_TOKENS:
                input_ids = input_ids[:, :_MAX_TOKENS]
                attention_mask = attention_mask[:, :_MAX_TOKENS]

            # Build feed dict
            feed = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

            # Some models also need token_type_ids — check dynamically
            session_inputs = {
                i.name for i in self._session.get_inputs()
            }
            if "token_type_ids" in session_inputs:
                feed["token_type_ids"] = np.zeros_like(input_ids)

            # Run inference
            outputs = self._session.run(None, feed)
            logits = outputs[0][0].tolist()  # [safe_score, injection_score]

            # Softmax → probabilities
            probs = _softmax(logits)

            # Threat probability = INJECTION class (index 1)
            # ProtectAI DeBERTa v3: label 0 = SAFE, label 1 = INJECTION
            threat_prob = float(probs[1])
            threshold = settings.FIREWALL_ML_THRESHOLD

            elapsed = round((time.monotonic() - start) * 1000, 2)

            result = MLResult(
                triggered=threat_prob >= threshold,
                confidence=round(threat_prob, 4),
                method="ONNX",
                latency_ms=elapsed,
            )

            if result.triggered:
                logger.info(
                    f"MLClassifier: threat detected "
                    f"prob={threat_prob:.3f} "
                    f"threshold={threshold} "
                    f"time={elapsed}ms"
                )

            return result

        except Exception as e:
            logger.warning(
                f"MLClassifier inference error: "
                f"{type(e).__name__}: {str(e)[:100]}"
            )
            return MLResult(
                method="ERROR",
                latency_ms=round((time.monotonic() - start) * 1000, 2),
            )

    def get_status(self) -> dict:
        """Return classifier status for health endpoint."""
        return {
            "available": self.available,
            "method": "ProtectAI-DeBERTa-ONNX" if self.available else "UNAVAILABLE",
            "threshold": settings.FIREWALL_ML_THRESHOLD,
            "model": self._model_path.name if self._model_path else None,
        }


# ── Global singleton ──────────────────────────────────────────────────
# Loaded at import time — model loads once, cached for all requests
ml_classifier = MLClassifier()
