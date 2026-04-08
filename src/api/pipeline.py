"""
RAG pipeline orchestrator.

Wires together: chunking → embedding → retrieval → reranking → LLM → citations.
This is the main entry point called by the FastAPI routes.
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any

from src.config import settings
from src.models.schemas import (
    IngestRequest, IngestResponse, QueryRequest,
    QueryResponse,
)
from src.chunking.engine import get_chunker
from src.retrieval.engine import HybridRetriever, FAISSStore, BM25Index
from src.retrieval.embeddings import get_embedding_service
from src.reranker.engine import get_reranker
from src.llm.client import get_llm_client
from src.citation.formatter import citation_formatter
from src.utils.logging import get_logger

log = get_logger(__name__)


class RAGPipeline:
    """
    Full RAG pipeline with configurable components.

    Lifecycle:
      1. startup()  — called once during FastAPI lifespan
      2. ingest()   — add documents
      3. query()    — answer questions
      4. shutdown() — cleanup
    """

    def __init__(self):
        self._embedding_svc = None
        self._retriever: HybridRetriever | None = None
        self._reranker = None
        self._llm_client = None
        self._ready = False

    def startup(self) -> None:
        """Initialize all components. Called during FastAPI lifespan."""
        log.info("pipeline.starting")

        # Determine if we're in test/dummy mode
        use_dummy_llm = os.getenv("RAG_USE_DUMMY_LLM", "true").lower() == "true"

        # 1. Embedding service
        self._embedding_svc = get_embedding_service()

        # 2. Vector store (FAISS for now)
        vector_store = FAISSStore(dim=settings.embedding_dim)

        # 3. BM25 index
        bm25_index = BM25Index(k1=settings.bm25_k1, b=settings.bm25_b)

        # 4. Hybrid retriever
        self._retriever = HybridRetriever(
            vector_store=vector_store,
            bm25_index=bm25_index,
            embed_fn=self._embedding_svc.encode,
            alpha=settings.hybrid_alpha,
        )

        # 5. Reranker
        self._reranker = get_reranker(enabled=settings.reranker_enabled)

        # 6. LLM client
        self._llm_client = get_llm_client(use_dummy=use_dummy_llm)

        self._ready = True
        log.info("pipeline.ready", dummy_llm=use_dummy_llm)

    def shutdown(self) -> None:
        """Cleanup resources."""
        log.info("pipeline.shutdown")
        self._ready = False

    @property
    def is_ready(self) -> bool:
        return self._ready

    # ── Ingest ──────────────────────────────────────────────────────────

    def ingest(self, request: IngestRequest) -> IngestResponse:
        """Chunk a document, embed it, and add to both indexes."""
        t0 = time.perf_counter()
        doc_id = str(uuid.uuid4())

        # 1. Chunk
        chunker = get_chunker(
            strategy=request.chunk_strategy,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )
        chunks = chunker.chunk(
            text=request.text,
            doc_id=doc_id,
            filename=request.filename,
            title=request.title,
            extra_meta=request.metadata,
        )

        if not chunks:
            return IngestResponse(doc_id=doc_id, num_chunks=0, chunk_strategy=request.chunk_strategy)

        # 2. Embed
        texts = [c.text for c in chunks]
        embeddings = self._embedding_svc.encode_batch(texts)
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb.tolist()

        # 3. Add to retriever (both vector + BM25)
        self._retriever.ingest(chunks)

        elapsed = (time.perf_counter() - t0) * 1000
        log.info(
            "pipeline.ingest",
            doc_id=doc_id,
            num_chunks=len(chunks),
            strategy=request.chunk_strategy,
            latency_ms=round(elapsed, 1),
        )
        return IngestResponse(
            doc_id=doc_id,
            num_chunks=len(chunks),
            chunk_strategy=request.chunk_strategy,
        )

    # ── Query ───────────────────────────────────────────────────────────

    def query(self, request: QueryRequest) -> QueryResponse:
        """Full RAG pipeline: retrieve → rerank → generate → cite."""
        t_total = time.perf_counter()

        # 1. Retrieve
        t_ret = time.perf_counter()
        candidates = self._retriever.search(
            query=request.query,
            vector_top_k=settings.vector_top_k,
            bm25_top_k=settings.bm25_top_k,
        )
        retrieval_ms = (time.perf_counter() - t_ret) * 1000

        # 2. Rerank
        t_rr = time.perf_counter()
        if request.rerank and self._reranker:
            reranked = self._reranker.rerank(
                query=request.query,
                chunks=candidates,
                top_k=request.top_k,
            )
        else:
            reranked = sorted(candidates, key=lambda c: c.rrf_score, reverse=True)[: request.top_k]
            for c in reranked:
                c.rerank_score = c.rrf_score
        rerank_ms = (time.perf_counter() - t_rr) * 1000

        # 3. Generate answer
        t_gen = time.perf_counter()
        if reranked:
            answer, _ = self._llm_client.generate(request.query, reranked)
        else:
            answer = "I could not find relevant information in the available documents to answer your question."
        gen_ms = (time.perf_counter() - t_gen) * 1000

        # 4. Build citations
        citations = citation_formatter.build_citations(reranked)

        total_ms = (time.perf_counter() - t_total) * 1000
        log.info(
            "pipeline.query",
            query_len=len(request.query),
            candidates=len(candidates),
            reranked=len(reranked),
            total_ms=round(total_ms, 1),
        )

        return QueryResponse(
            answer=answer,
            citations=citations,
            query=request.query,
            model=getattr(self._llm_client, '_model', 'unknown'),
            latency_ms=round(total_ms, 1),
            retrieval_latency_ms=round(retrieval_ms, 1),
            rerank_latency_ms=round(rerank_ms, 1),
            generation_latency_ms=round(gen_ms, 1),
            num_chunks_retrieved=len(candidates),
            num_chunks_after_rerank=len(reranked),
        )

    # ── Stats ───────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Return current index statistics."""
        return {
            "vector_count": self._retriever.vector_store.count() if self._retriever else 0,
            "bm25_count": self._retriever.bm25_index.count() if self._retriever else 0,
            "ready": self._ready,
        }


# Module-level singleton
pipeline = RAGPipeline()
