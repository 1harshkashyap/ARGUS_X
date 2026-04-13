"""
ARGUS-X — Model Loader
Downloads ONNX model from HuggingFace Hub or Supabase Storage.
Falls back to rule-only mode if model unavailable.
"""
import os
import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger("argus.model_loader")

# Try imports
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    log.warning("⚠️ onnxruntime not installed — ML classifier disabled")

try:
    from huggingface_hub import hf_hub_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    log.warning("⚠️ huggingface_hub not installed — auto-download disabled")

try:
    from transformers import AutoTokenizer
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False
    log.warning("⚠️ transformers not installed — tokenizer disabled")


class ModelLoader:
    """
    Loads ML models for ARGUS-X.
    Priority: local file → HuggingFace Hub download → rule-only fallback.
    """

    def __init__(self):
        self.model_dir = os.getenv("MODEL_DIR", "./models")
        self.hf_repo = os.getenv("HF_MODEL_REPO", "")
        self.onnx_session: Optional[any] = None
        self.tokenizer: Optional[any] = None
        self.ml_ready = False
        
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)
        self._load_models()

    def _load_models(self):
        """
        Attempt to load ONNX model and tokenizer.
        Priority: pre-baked (Docker build) → local file → HuggingFace download → rule-only.
        """
        model_path = os.path.join(self.model_dir, "argus_classifier.onnx")
        
        # Step 1: Try pre-baked or local file (instant startup — no download)
        if os.path.exists(model_path):
            log.info("📦 Found pre-baked/local model — skipping HuggingFace download")
            self._load_onnx(model_path)
            self._load_tokenizer_local()
            return

        # Step 2: Try HuggingFace Hub download (runtime fallback for dev)
        # WARNING: This adds 2-5 min to cold start. For production, bake the
        # model into the Docker image using: docker build --build-arg HF_MODEL_REPO=...
        if HF_AVAILABLE and self.hf_repo:
            try:
                log.info(f"📥 Downloading model from HuggingFace: {self.hf_repo}")
                log.warning("⚠️ Runtime model download — expect slow cold start. "
                           "For production, use --build-arg HF_MODEL_REPO to bake into image.")
                downloaded_path = hf_hub_download(
                    repo_id=self.hf_repo,
                    filename="argus_classifier.onnx",
                    local_dir=self.model_dir,
                )
                self._load_onnx(downloaded_path)
                self._load_tokenizer_hf()
                return
            except Exception as e:
                log.warning(f"⚠️ HuggingFace download failed: {e}")

        # Step 3: Fallback — rule-only mode
        log.warning("⚠️ No ML model available — running in RULE-ONLY mode")
        self.ml_ready = False

    def _load_onnx(self, path: str):
        """Load ONNX model for inference."""
        if not ONNX_AVAILABLE:
            return
        try:
            self.onnx_session = ort.InferenceSession(
                path,
                providers=["CPUExecutionProvider"]
            )
            self.ml_ready = True
            log.info(f"✅ ONNX model loaded: {path}")
        except Exception as e:
            log.warning(f"⚠️ ONNX load failed: {e}")
            self.ml_ready = False

    def _load_tokenizer_local(self):
        """Load tokenizer from local directory."""
        if not TOKENIZER_AVAILABLE:
            return
        tokenizer_path = os.path.join(self.model_dir, "tokenizer")
        if os.path.exists(tokenizer_path):
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                log.info("✅ Tokenizer loaded from local")
            except Exception as e:
                log.warning(f"⚠️ Local tokenizer failed: {e}")
                self._load_tokenizer_fallback()
        else:
            self._load_tokenizer_fallback()

    def _load_tokenizer_hf(self):
        """Load tokenizer from HuggingFace Hub."""
        if not TOKENIZER_AVAILABLE:
            return
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.hf_repo)
            # Save locally for next time
            tokenizer_path = os.path.join(self.model_dir, "tokenizer")
            self.tokenizer.save_pretrained(tokenizer_path)
            log.info("✅ Tokenizer loaded from HuggingFace")
        except Exception as e:
            log.warning(f"⚠️ HF tokenizer failed: {e}")
            self._load_tokenizer_fallback()

    def _load_tokenizer_fallback(self):
        """Fallback to distilbert-base-uncased tokenizer."""
        if not TOKENIZER_AVAILABLE:
            return
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            log.info("✅ Fallback tokenizer loaded (distilbert-base-uncased)")
        except Exception as e:
            log.warning(f"⚠️ Fallback tokenizer failed: {e}")

    def get_onnx_session(self):
        return self.onnx_session

    def get_tokenizer(self):
        return self.tokenizer

    def status(self) -> dict:
        return {
            "ml_ready": self.ml_ready,
            "onnx_loaded": self.onnx_session is not None,
            "tokenizer_loaded": self.tokenizer is not None,
            "model_dir": self.model_dir,
            "hf_repo": self.hf_repo,
        }
