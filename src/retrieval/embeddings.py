"""
Embedding service.

Wraps sentence-transformers for encoding queries and documents.
Thread-safe singleton with lazy model loading.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from src.config import settings
from src.utils.logging import get_logger

log = get_logger(__name__)


class EmbeddingService:
    """Thread-safe embedding service with lazy model loading."""

    def __init__(self, model_name: str | None = None, device: str = "cpu"):
        self._model_name = model_name or settings.embedding_model
        self._device = device
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self._model_name, device=self._device)
                log.info(
                    "embedding.loaded", model=self._model_name, device=self._device
                )
            except Exception as exc:
                log.error("embedding.load_failed", error=str(exc))
                raise RuntimeError(f"Failed to load embedding model: {exc}") from exc

    def encode(self, text: str) -> np.ndarray:
        """Encode a single text string. Returns a 1-D numpy array."""
        self._load()
        vec = self._model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.array(vec, dtype=np.float32)

    def encode_batch(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Encode a batch of texts. Returns (N, dim) numpy array."""
        self._load()
        t0 = time.perf_counter()
        vecs = self._model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=False,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        log.info("embedding.batch", count=len(texts), latency_ms=round(elapsed, 1))
        return np.array(vecs, dtype=np.float32)

    @property
    def dim(self) -> int:
        self._load()
        return self._model.get_sentence_embedding_dimension()

    def health_check(self) -> bool:
        try:
            self._load()
            return self._model is not None
        except Exception:
            return False


# Module-level singleton
_embedding_service: EmbeddingService | None = None
_init_lock = threading.Lock()


def get_embedding_service(**kwargs) -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        with _init_lock:
            if _embedding_service is None:
                _embedding_service = EmbeddingService(**kwargs)
    return _embedding_service
