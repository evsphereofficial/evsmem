"""Embedding client — GGUF BGE-M3 via llama-cpp-python with CUDA."""

import logging, os, threading, time
from pathlib import Path
import numpy as np

logger = logging.getLogger("evsmem.embeddings")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

# Resolve GGUF model path
_MODEL_PATH = os.getenv(
    "EMBEDDING_MODEL",
    str(Path(__file__).resolve().parent / "models" / "bge-m3-Q8_0.gguf"),
)
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "http://localhost:1234")
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_EMBEDDING_MODEL", "text-embedding-bge-m3")

# Add CUDA DLL directory so llama_cpp can find them at import time
_venv_lib = Path(__file__).resolve().parent.parent / ".venv" / "Lib" / "site-packages"
_cuda_dll_dir = str(_venv_lib / "llama_cpp" / "lib")
if os.path.isdir(_cuda_dll_dir):
    try:
        os.add_dll_directory(_cuda_dll_dir)
    except Exception:
        pass

# Global singleton
_model = None
_MODEL_LOCK = threading.Lock()


def _load_model():
    """Load BGE-M3 GGUF once (GPU if available). Stays in memory."""
    global _model
    if _model is not None:
        return _model
    with _MODEL_LOCK:
        if _model is not None:
            return _model
        if not os.path.isfile(_MODEL_PATH):
            logger.warning(f"Model not found at {_MODEL_PATH}")
            return None
        try:
            from llama_cpp import Llama
            logger.info(f"Loading GGUF model from {_MODEL_PATH}")
            t0 = time.time()
            _model = Llama(
                model_path=str(_MODEL_PATH),
                embedding=True,
                n_gpu_layers=-1,
                n_ctx=2048,
                verbose=False,
            )
            logger.info(f"Model ready ({time.time()-t0:.1f}s)")
            return _model
        except Exception as e:
            logger.warning(f"GGUF model load failed: {e}")
            return None


class EmbeddingClient:
    """Embedding client — loads GGUF model once via llama-cpp-python."""

    def __init__(self):
        self._loaded = False

    def _ensure_model(self):
        if not self._loaded:
            _load_model()
            self._loaded = True

    def embed(self, text: str) -> list[float]:
        self._ensure_model()
        logger.info(f"Embedding: {len(text)} chars")
        if _model is not None:
            try:
                result = _model.create_embedding(text[:12000])
                return result["data"][0]["embedding"]
            except Exception as e:
                logger.warning(f"Embedding failed: {e}")
        import requests
        resp = requests.post(
            f"{EMBEDDING_BASE_URL.rstrip('/')}/v1/embeddings",
            json={"model": LM_STUDIO_MODEL, "input": text},
            timeout=(5, 30),
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self._ensure_model()
        if _model is not None:
            try:
                return [
                    _model.create_embedding(t[:12000])["data"][0]["embedding"]
                    for t in texts
                ]
            except Exception as e:
                logger.warning(f"Batch embedding failed: {e}")
        import requests
        resp = requests.post(
            f"{EMBEDDING_BASE_URL.rstrip('/')}/v1/embeddings",
            json={"model": LM_STUDIO_MODEL, "input": texts},
            timeout=(5, 60),
        )
        resp.raise_for_status()
        return [d["embedding"] for d in resp.json()["data"]]

    def cosine_similarity(self, a, b):
        a_np = np.array(a, dtype=np.float32)
        b_np = np.array(b, dtype=np.float32)
        return float(np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np) + 1e-10))

    def is_available(self) -> bool:
        return os.path.isfile(_MODEL_PATH)

    def model_name(self) -> str:
        return "bge-m3-Q8_0.gguf"
