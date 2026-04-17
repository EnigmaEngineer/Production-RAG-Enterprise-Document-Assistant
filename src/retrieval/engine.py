"""
Hybrid retrieval engine combining dense (FAISS/Qdrant) and sparse (BM25) search.
Produces a fused candidate list using Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

import time
import threading
from abc import ABC, abstractmethod

import numpy as np
from rank_bm25 import BM25Okapi

from src.models.schemas import Chunk
from src.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Vector store interface + implementations
# ---------------------------------------------------------------------------


class VectorStore(ABC):
    """Base class for vector storage backends.

    FAISS and Qdrant both implement this. The retriever doesn't care
    which one is behind it.
    """

    @abstractmethod
    def add(self, chunks: list[Chunk]) -> None:
        """Store chunks with their embeddings. Will raise if embeddings are null."""
        ...

    @abstractmethod
    def search(
        self, query_embedding: np.ndarray, top_k: int
    ) -> list[tuple[str, float]]:
        """Return (chunk_id, score) pairs sorted by descending similarity."""
        ...

    @abstractmethod
    def count(self) -> int:
        """How many vectors are in the index."""
        ...

    @abstractmethod
    def delete_by_doc_id(self, doc_id: str) -> int:
        """Remove all chunks for a given doc. Returns count of removed chunks."""
        ...


class FAISSStore(VectorStore):
    """In-process FAISS vector store with optional persistence."""

    def __init__(self, dim: int = 768, index_path: str | None = None):
        """Set up a FAISS IndexFlatIP store.

        Uses inner product on L2-normalized vectors which gives you cosine
        similarity. Everything lives in memory. No built-in persistence so
        if the process dies the index is gone.
        """
        import faiss

        self._dim = dim
        self._index = faiss.IndexFlatIP(
            dim
        )  # inner product (cosine on normalized vecs)
        self._id_map: list[str] = []  # chunk_id at each position
        self._chunk_map: dict[str, Chunk] = {}
        self._lock = threading.Lock()
        self._index_path = index_path
        log.info("faiss_store.init", dim=dim)

    def add(self, chunks: list[Chunk]) -> None:
        """Add chunks to the index. Normalizes embeddings to unit vectors first
        so inner product gives cosine similarity."""
        if not chunks:
            return
        embeddings = []
        for c in chunks:
            if c.embedding is None:
                raise ValueError(f"Chunk {c.chunk_id} has no embedding")
            embeddings.append(c.embedding)

        matrix = np.array(embeddings, dtype=np.float32)
        # Normalize for cosine similarity via inner product
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        matrix = matrix / norms

        with self._lock:
            self._index.add(matrix)
            for c in chunks:
                self._id_map.append(c.chunk_id)
                self._chunk_map[c.chunk_id] = c
        log.info("faiss_store.added", count=len(chunks), total=self._index.ntotal)

    def search(
        self, query_embedding: np.ndarray, top_k: int = 100
    ) -> list[tuple[str, float]]:
        """Find the closest vectors. Query gets normalized to match the index.
        Returns empty list if nothing is indexed yet."""
        if self._index.ntotal == 0:
            return []
        qvec = query_embedding.reshape(1, -1).astype(np.float32)
        norm = np.linalg.norm(qvec)
        if norm > 0:
            qvec = qvec / norm

        k = min(top_k, self._index.ntotal)
        with self._lock:
            scores, indices = self._index.search(qvec, k)

        results: list[tuple[str, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._id_map):
                continue
            results.append((self._id_map[idx], float(score)))
        return results

    def count(self) -> int:
        """Number of vectors in the index."""
        return self._index.ntotal

    def delete_by_doc_id(self, doc_id: str) -> int:
        """Delete all chunks from a document. FAISS doesn't support deletion,
        so we rebuild the index without those chunks."""
        with self._lock:
            to_keep = [
                cid
                for cid in self._id_map
                if self._chunk_map[cid].metadata.doc_id != doc_id
            ]
            removed = len(self._id_map) - len(to_keep)
            if removed == 0:
                return 0

            import faiss

            new_index = faiss.IndexFlatIP(self._dim)
            new_id_map: list[str] = []
            embeddings = []
            for cid in to_keep:
                chunk = self._chunk_map[cid]
                embeddings.append(chunk.embedding)
                new_id_map.append(cid)

            if embeddings:
                matrix = np.array(embeddings, dtype=np.float32)
                norms = np.linalg.norm(matrix, axis=1, keepdims=True)
                norms[norms == 0] = 1
                matrix = matrix / norms
                new_index.add(matrix)

            # Remove deleted chunks from map
            for cid in list(self._chunk_map.keys()):
                if self._chunk_map[cid].metadata.doc_id == doc_id:
                    del self._chunk_map[cid]

            self._index = new_index
            self._id_map = new_id_map
        log.info("faiss_store.deleted", doc_id=doc_id, removed=removed)
        return removed

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        """Grab a chunk by ID. None if it doesn't exist."""
        return self._chunk_map.get(chunk_id)

    def get_all_chunks(self) -> list[Chunk]:
        """Dump every chunk in the store. Mainly useful for rebuilds."""
        return list(self._chunk_map.values())


# ---------------------------------------------------------------------------
# BM25 sparse index (in-memory)
# ---------------------------------------------------------------------------


class BM25Index:
    """Wraps rank_bm25 with chunk ID tracking and live rebuild."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """Set up a BM25 keyword index.

        Wraps rank_bm25.BM25Okapi. Rebuilds the whole index on every add
        or delete because BM25Okapi doesn't support incremental updates.
        Tokenization is just lowercased whitespace splitting.

        Heads up: IDF produces zeros for terms in 50%+ of documents. With
        only 2 docs almost everything gets zero. Need at least 4 docs for
        BM25 to be useful. Not a bug, just how Okapi BM25 math works.
        """
        self._k1 = k1
        self._b = b
        self._bm25: BM25Okapi | None = None
        self._chunk_ids: list[str] = []
        self._corpus_tokens: list[list[str]] = []
        self._lock = threading.Lock()

    def add(self, chunks: list[Chunk]) -> None:
        """Add chunks and rebuild. Not incremental, rebuilds the whole thing."""
        with self._lock:
            for c in chunks:
                tokens = c.text.lower().split()
                self._corpus_tokens.append(tokens)
                self._chunk_ids.append(c.chunk_id)
            self._rebuild()
        log.info("bm25_index.added", count=len(chunks), total=len(self._chunk_ids))

    def search(self, query: str, top_k: int = 100) -> list[tuple[str, float]]:
        """Run a BM25 keyword search. Only returns chunks with positive scores
        so you might get fewer than top_k results on small corpora."""
        if self._bm25 is None or not self._chunk_ids:
            return []
        tokens = query.lower().split()
        with self._lock:
            scores = self._bm25.get_scores(tokens)

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self._chunk_ids[idx], float(scores[idx])))
        return results

    def delete_by_doc_id(self, doc_id: str, chunk_map: dict[str, Chunk]) -> int:
        """Remove chunks for a doc and rebuild. Needs the chunk_map because
        BM25Index only tracks chunk IDs, not which doc they belong to."""
        with self._lock:
            new_tokens = []
            new_ids = []
            removed = 0
            for cid, tokens in zip(self._chunk_ids, self._corpus_tokens):
                chunk = chunk_map.get(cid)
                if chunk and chunk.metadata.doc_id == doc_id:
                    removed += 1
                    continue
                new_tokens.append(tokens)
                new_ids.append(cid)
            self._corpus_tokens = new_tokens
            self._chunk_ids = new_ids
            self._rebuild()
        return removed

    def _rebuild(self) -> None:
        if self._corpus_tokens:
            self._bm25 = BM25Okapi(self._corpus_tokens, k1=self._k1, b=self._b)
        else:
            self._bm25 = None

    def count(self) -> int:
        """How many chunks are indexed."""
        return len(self._chunk_ids)


# ---------------------------------------------------------------------------
# Hybrid retriever with RRF fusion
# ---------------------------------------------------------------------------


class HybridRetriever:
    """
    Combines vector search and BM25, fuses results with Reciprocal Rank Fusion.

    Usage:
        retriever = HybridRetriever(vector_store, bm25_index, embed_fn)
        candidates = retriever.search(query, top_k=100)
    """

    RRF_K = 60  # standard RRF constant

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        embed_fn,  # callable: str -> np.ndarray
        alpha: float = 0.5,
    ):
        """Wire up the hybrid retriever.

        alpha controls the balance. 1.0 = vector only, 0.0 = BM25 only.
        0.5 gives equal weight to both which works well in practice.
        """
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.embed_fn = embed_fn
        self.alpha = alpha  # weight: 1.0 = vector only, 0.0 = BM25 only

    def search(
        self,
        query: str,
        vector_top_k: int = 100,
        bm25_top_k: int = 100,
    ) -> list[Chunk]:
        """Run both vector and BM25 search, merge with RRF.

        Each result gets vector_score, bm25_score and rrf_score attached
        so the reranker and citation formatter can use them downstream.
        RRF constant is 60 which is the standard value from the paper.
        """
        t0 = time.perf_counter()

        # Dense retrieval
        query_emb = self.embed_fn(query)
        vec_results = self.vector_store.search(query_emb, top_k=vector_top_k)

        # Sparse retrieval
        bm25_results = self.bm25_index.search(query, top_k=bm25_top_k)

        # RRF fusion
        rrf_scores: dict[str, float] = {}

        for rank, (chunk_id, score) in enumerate(vec_results):
            rrf = self.alpha / (self.RRF_K + rank + 1)
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + rrf

        for rank, (chunk_id, score) in enumerate(bm25_results):
            rrf = (1 - self.alpha) / (self.RRF_K + rank + 1)
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + rrf

        # Build result chunks
        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
        chunks: list[Chunk] = []
        for cid in sorted_ids:
            chunk = (
                self.vector_store.get_chunk(cid)
                if isinstance(self.vector_store, FAISSStore)
                else None
            )
            if chunk is None:
                continue
            # Attach scores for downstream use
            vec_score = next((s for i, s in vec_results if i == cid), 0.0)
            bm25_score = next((s for i, s in bm25_results if i == cid), 0.0)
            chunk.vector_score = vec_score
            chunk.bm25_score = bm25_score
            chunk.rrf_score = rrf_scores[cid]
            chunks.append(chunk)

        elapsed = (time.perf_counter() - t0) * 1000
        log.info(
            "hybrid_retriever.search",
            query_len=len(query),
            vec_hits=len(vec_results),
            bm25_hits=len(bm25_results),
            fused_hits=len(chunks),
            latency_ms=round(elapsed, 1),
        )
        return chunks

    def ingest(self, chunks: list[Chunk]) -> None:
        """Add chunks to both vector store and BM25 index."""
        self.vector_store.add(chunks)
        self.bm25_index.add(chunks)

    def delete_doc(self, doc_id: str) -> int:
        """Remove a document from both indexes."""
        chunk_map = {}
        if isinstance(self.vector_store, FAISSStore):
            chunk_map = self.vector_store._chunk_map
        v_removed = self.vector_store.delete_by_doc_id(doc_id)
        b_removed = self.bm25_index.delete_by_doc_id(doc_id, chunk_map)
        return max(v_removed, b_removed)
