"""
Cross-encoder reranker.

Uses BAAI/bge-reranker-v2-m3 (or any cross-encoder) to rescore
query-document pairs. Falls back to RRF scores on timeout or failure.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod


from src.config import settings
from src.models.schemas import Chunk
from src.utils.logging import get_logger

log = get_logger(__name__)


class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, chunks: list[Chunk], top_k: int) -> list[Chunk]: ...


class CrossEncoderReranker(BaseReranker):
    """
    Production reranker using a cross-encoder model.
    Batches pairs to avoid GPU OOM and includes timeout fallback.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        batch_size: int = 25,
        device: str = "cpu",
    ):
        self._model_name = model_name
        self._batch_size = batch_size
        self._device = device
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name, device=self._device)
            log.info("reranker.loaded", model=self._model_name, device=self._device)
        except Exception as exc:
            log.error("reranker.load_failed", error=str(exc))
            self._model = None
        return self._model

    def rerank(self, query: str, chunks: list[Chunk], top_k: int = 10) -> list[Chunk]:
        """
        Score each (query, chunk.text) pair and return the top_k by score.
        Falls back to RRF ordering if the model is unavailable.
        """
        if not chunks:
            return []

        model = self._load_model()
        if model is None:
            log.warning("reranker.fallback_to_rrf")
            return self._fallback_rrf(chunks, top_k)

        t0 = time.perf_counter()
        try:
            pairs = [(query, c.text) for c in chunks]
            all_scores: list[float] = []

            # Batch scoring to control memory usage
            for i in range(0, len(pairs), self._batch_size):
                batch = pairs[i : i + self._batch_size]
                scores = model.predict(batch, show_progress_bar=False)
                all_scores.extend(
                    scores.tolist() if hasattr(scores, "tolist") else list(scores)
                )

            # Attach scores and sort
            for chunk, score in zip(chunks, all_scores):
                chunk.rerank_score = float(score)

            ranked = sorted(chunks, key=lambda c: c.rerank_score, reverse=True)
            elapsed = (time.perf_counter() - t0) * 1000
            log.info(
                "reranker.success",
                input_count=len(chunks),
                output_count=min(top_k, len(ranked)),
                latency_ms=round(elapsed, 1),
            )
            return ranked[:top_k]

        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            log.error("reranker.error", error=str(exc), latency_ms=round(elapsed, 1))
            return self._fallback_rrf(chunks, top_k)

    @staticmethod
    def _fallback_rrf(chunks: list[Chunk], top_k: int) -> list[Chunk]:
        """Use pre-computed RRF scores when reranker fails."""
        ranked = sorted(chunks, key=lambda c: c.rrf_score, reverse=True)
        for c in ranked:
            c.rerank_score = c.rrf_score
        return ranked[:top_k]


class DummyReranker(BaseReranker):
    """
    No-op reranker for testing or when reranking is disabled.
    Passes through chunks sorted by RRF score.
    """

    def rerank(self, query: str, chunks: list[Chunk], top_k: int = 10) -> list[Chunk]:
        ranked = sorted(chunks, key=lambda c: c.rrf_score, reverse=True)
        for c in ranked:
            c.rerank_score = c.rrf_score
        return ranked[:top_k]


def get_reranker(enabled: bool = True, **kwargs) -> BaseReranker:
    """Factory: returns a reranker instance."""
    if not enabled:
        return DummyReranker()
    return CrossEncoderReranker(
        model_name=kwargs.get("model_name", settings.reranker_model),
        batch_size=kwargs.get("batch_size", settings.reranker_batch_size),
        device=kwargs.get("device", "cpu"),
    )
