"""BGE-M3 embedding engine using sentence-transformers.

Provides a functional API (generate_embedding, generate_embeddings_batch, embedding_dimension)
and a backward-compatible EmbeddingClient class for existing consumers.
"""

from __future__ import annotations

__all__ = [
    "generate_embedding",
    "generate_embeddings_batch",
    "embedding_dimension",
    "EmbeddingClient",
]

import logging
import os
import threading
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

# ── Lazy-loaded model singleton ───────────────────────────────────────────

_model = None
_device = None
_lock = threading.Lock()


def _get_model():
    """Load BGE-M3 via sentence-transformers on first call. Thread-safe."""
    global _model, _device
    if _model is not None:
        return _model
    with _lock:
        if _model is not None:
            return _model
        try:
            from sentence_transformers import SentenceTransformer
            import torch

            _device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Loading BGE-M3 model on %s ...", _device)
            _model = SentenceTransformer("BAAI/bge-m3", device=_device)
            logger.info("BGE-M3 model loaded successfully.")
        except ImportError:
            logger.error(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            )
            raise
        except Exception as exc:
            logger.error("Failed to load BGE-M3 model: %s", exc)
            raise
    return _model


# ── Public functional API ─────────────────────────────────────────────────


def generate_embedding(text: str) -> list[float]:
    """Generate a single BGE-M3 embedding for *text*.

    Returns a 1024-dimensional float vector normalized to unit length.
    Returns a zero vector for empty / whitespace-only input.
    """
    if not text or not text.strip():
        return [0.0] * 1024
    model = _get_model()
    emb = model.encode(text, normalize_embeddings=True)
    return emb.tolist()


def generate_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Generate BGE-M3 embeddings for a batch of texts.

    Empty strings are returned as zero vectors.  This is more efficient than
    calling ``generate_embedding`` in a loop because the model processes the
    entire batch at once.
    """
    if not texts:
        return []

    cleaned: list[str] = []
    empty_indices: list[int] = []
    for i, t in enumerate(texts):
        if t and t.strip():
            cleaned.append(t)
        else:
            cleaned.append("")
            empty_indices.append(i)

    model = _get_model()
    embs = model.encode(cleaned, normalize_embeddings=True)

    results: list[list[float]] = []
    for i, emb in enumerate(embs):
        if not cleaned[i]:
            results.append([0.0] * 1024)
        else:
            results.append(emb.tolist())
    return results


def embedding_dimension() -> int:
    """Return the embedding dimensionality (1024 for BGE-M3)."""
    return 1024


# ── Backward-compatible class wrapper ─────────────────────────────────────


class EmbeddingClient:
    """Backward-compatible embedding client wrapping sentence-transformers BGE-M3.

    Preserves the same interface as the previous llama-cpp-python / LM Studio
    implementation so that existing consumers (main.py, deriver.py) continue
    to work without modification.
    """

    def __init__(self) -> None:
        self._loaded = False

    def _ensure_model(self) -> None:
        """Trigger lazy model load (idempotent)."""
        if not self._loaded:
            _get_model()
            self._loaded = True

    def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
        self._ensure_model()
        return generate_embedding(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings."""
        self._ensure_model()
        return generate_embeddings_batch(texts)

    def cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        a_np = np.array(a, dtype=np.float32)
        b_np = np.array(b, dtype=np.float32)
        return float(
            np.dot(a_np, b_np)
            / (np.linalg.norm(a_np) * np.linalg.norm(b_np) + 1e-10)
        )

    def is_available(self) -> bool:
        """Return True if the BGE-M3 model is loadable."""
        try:
            _get_model()
            return _model is not None
        except Exception:
            return False

    def model_name(self) -> str:
        """Return the HuggingFace model identifier."""
        return "BAAI/bge-m3"
