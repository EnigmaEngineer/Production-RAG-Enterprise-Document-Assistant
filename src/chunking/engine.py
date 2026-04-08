"""
Document chunking engine.

Supports three strategies selectable at runtime:
  1. recursive  — recursive character splitter (default)
  2. semantic   — embedding-similarity-based boundary detection
  3. fixed_overlap — simple fixed-window with overlap

All strategies produce List[Chunk] with populated metadata.
"""

from __future__ import annotations

import re
import uuid
from abc import ABC, abstractmethod

import tiktoken

from src.models.schemas import Chunk, DocumentMetadata
from src.utils.logging import get_logger

log = get_logger(__name__)

# Shared tokenizer for counting tokens
_enc = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text, disallowed_special=()))


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BaseChunker(ABC):
    """Interface every chunker must implement."""

    @abstractmethod
    def chunk(
        self,
        text: str,
        doc_id: str,
        filename: str = "",
        title: str = "",
        extra_meta: dict | None = None,
    ) -> list[Chunk]: ...


# ---------------------------------------------------------------------------
# 1. Recursive character splitter
# ---------------------------------------------------------------------------


class RecursiveChunker(BaseChunker):
    """
    Splits text by a hierarchy of separators (paragraph → sentence → word),
    respecting a maximum token count per chunk with configurable overlap.
    """

    SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 128):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # -- public API --

    def chunk(
        self,
        text: str,
        doc_id: str,
        filename: str = "",
        title: str = "",
        extra_meta: dict | None = None,
    ) -> list[Chunk]:
        raw_splits = self._split_recursive(text, self.SEPARATORS)
        merged = self._merge_with_overlap(raw_splits)

        chunks: list[Chunk] = []
        for idx, segment in enumerate(merged):
            meta = DocumentMetadata(
                doc_id=doc_id,
                filename=filename,
                title=title,
                chunk_index=idx,
                chunk_strategy="recursive",
                total_chunks=len(merged),
                extra=extra_meta or {},
            )
            chunks.append(
                Chunk(chunk_id=str(uuid.uuid4()), text=segment, metadata=meta)
            )
        return chunks

    # -- internals --

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split using the finest separator that keeps chunks under size."""
        if not text.strip():
            return []

        if _count_tokens(text) <= self.chunk_size:
            return [text.strip()]

        sep = separators[0] if separators else ""
        parts = text.split(sep) if sep else list(text)
        result: list[str] = []
        current = ""

        for part in parts:
            candidate = (current + sep + part) if current else part
            if _count_tokens(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current.strip():
                    result.append(current.strip())
                # If a single part exceeds the limit, recurse with finer separators
                if _count_tokens(part) > self.chunk_size and len(separators) > 1:
                    result.extend(self._split_recursive(part, separators[1:]))
                    current = ""
                else:
                    current = part

        if current.strip():
            result.append(current.strip())
        return result

    def _merge_with_overlap(self, splits: list[str]) -> list[str]:
        """Merge small splits and add overlap between consecutive chunks."""
        if not splits:
            return []
        if len(splits) == 1:
            return splits

        merged: list[str] = []
        for i, segment in enumerate(splits):
            if i == 0:
                merged.append(segment)
                continue
            # Prepend tail of previous segment as overlap context
            prev_tokens = _enc.encode(splits[i - 1], disallowed_special=())
            overlap_tokens = (
                prev_tokens[-self.chunk_overlap :]
                if len(prev_tokens) > self.chunk_overlap
                else prev_tokens
            )
            overlap_text = _enc.decode(overlap_tokens)
            merged.append(overlap_text.strip() + " " + segment)
        return merged


# ---------------------------------------------------------------------------
# 2. Semantic chunker (embedding-similarity boundaries)
# ---------------------------------------------------------------------------


class SemanticChunker(BaseChunker):
    """
    Splits text into sentences, then groups consecutive sentences whose
    embeddings are above a similarity threshold.

    NOTE: Requires a SentenceTransformer model. Falls back to recursive
    chunking if the model is unavailable.
    """

    def __init__(
        self,
        embedding_model_name: str = "BAAI/bge-base-en-v1.5",
        similarity_threshold: float = 0.75,
        max_chunk_tokens: int = 512,
    ):
        self.threshold = similarity_threshold
        self.max_tokens = max_chunk_tokens
        self._model = None
        self._model_name = embedding_model_name

    def _get_model(self):
        """Lazy-load embedding model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self._model_name)
                log.info("semantic_chunker.model_loaded", model=self._model_name)
            except Exception as exc:
                log.warning("semantic_chunker.model_failed", error=str(exc))
                return None
        return self._model

    def chunk(
        self,
        text: str,
        doc_id: str,
        filename: str = "",
        title: str = "",
        extra_meta: dict | None = None,
    ) -> list[Chunk]:
        import numpy as np

        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            meta = DocumentMetadata(
                doc_id=doc_id,
                filename=filename,
                title=title,
                chunk_index=0,
                chunk_strategy="semantic",
                total_chunks=1,
                extra=extra_meta or {},
            )
            return [Chunk(chunk_id=str(uuid.uuid4()), text=text.strip(), metadata=meta)]

        model = self._get_model()
        if model is None:
            log.warning("semantic_chunker.fallback_to_recursive")
            return RecursiveChunker(self.max_tokens, 128).chunk(
                text, doc_id, filename, title, extra_meta
            )

        embeddings = model.encode(
            sentences, normalize_embeddings=True, show_progress_bar=False
        )

        # Compute cosine similarity between consecutive sentences
        groups: list[list[str]] = [[sentences[0]]]
        for i in range(1, len(sentences)):
            sim = float(np.dot(embeddings[i - 1], embeddings[i]))
            current_group_text = " ".join(groups[-1]) + " " + sentences[i]
            if (
                sim >= self.threshold
                and _count_tokens(current_group_text) <= self.max_tokens
            ):
                groups[-1].append(sentences[i])
            else:
                groups.append([sentences[i]])

        chunks: list[Chunk] = []
        for idx, group in enumerate(groups):
            meta = DocumentMetadata(
                doc_id=doc_id,
                filename=filename,
                title=title,
                chunk_index=idx,
                chunk_strategy="semantic",
                total_chunks=len(groups),
                extra=extra_meta or {},
            )
            chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid4()),
                    text=" ".join(group).strip(),
                    metadata=meta,
                )
            )
        return chunks

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Naive sentence splitter (handles ., !, ?)."""
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in parts if s.strip()]


# ---------------------------------------------------------------------------
# 3. Fixed overlap chunker
# ---------------------------------------------------------------------------


class FixedOverlapChunker(BaseChunker):
    """
    Simple sliding window: fixed token count with a fixed overlap.
    Fastest strategy, best for bulk ingestion.
    """

    def __init__(self, chunk_size: int = 256, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(
        self,
        text: str,
        doc_id: str,
        filename: str = "",
        title: str = "",
        extra_meta: dict | None = None,
    ) -> list[Chunk]:
        tokens = _enc.encode(text, disallowed_special=())
        step = max(1, self.chunk_size - self.chunk_overlap)
        segments: list[str] = []

        for start in range(0, len(tokens), step):
            end = min(start + self.chunk_size, len(tokens))
            segment_text = _enc.decode(tokens[start:end]).strip()
            if segment_text:
                segments.append(segment_text)
            if end >= len(tokens):
                break

        chunks: list[Chunk] = []
        for idx, seg in enumerate(segments):
            meta = DocumentMetadata(
                doc_id=doc_id,
                filename=filename,
                title=title,
                chunk_index=idx,
                chunk_strategy="fixed_overlap",
                total_chunks=len(segments),
                extra=extra_meta or {},
            )
            chunks.append(Chunk(chunk_id=str(uuid.uuid4()), text=seg, metadata=meta))
        return chunks


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

CHUNKER_REGISTRY: dict[str, type[BaseChunker]] = {
    "recursive": RecursiveChunker,
    "semantic": SemanticChunker,
    "fixed_overlap": FixedOverlapChunker,
}


def get_chunker(
    strategy: str = "recursive",
    chunk_size: int = 512,
    chunk_overlap: int = 128,
    **kwargs,
) -> BaseChunker:
    """Factory: returns a chunker instance by strategy name."""
    cls = CHUNKER_REGISTRY.get(strategy)
    if cls is None:
        raise ValueError(
            f"Unknown chunk strategy: {strategy!r}. Available: {list(CHUNKER_REGISTRY)}"
        )

    if strategy == "recursive":
        return cls(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    elif strategy == "semantic":
        return cls(
            similarity_threshold=kwargs.get("semantic_threshold", 0.75),
            max_chunk_tokens=chunk_size,
        )
    elif strategy == "fixed_overlap":
        return cls(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    else:
        return cls()
