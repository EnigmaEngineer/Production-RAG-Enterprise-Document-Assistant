"""
Unit tests for the RAG pipeline components.
Run with: pytest tests/ -v
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np

from src.chunking.engine import (
    RecursiveChunker,
    FixedOverlapChunker,
    get_chunker,
)
from src.retrieval.engine import FAISSStore, BM25Index, HybridRetriever
from src.reranker.engine import DummyReranker
from src.citation.formatter import CitationFormatter
from src.llm.client import DummyLLMClient
from src.models.schemas import Chunk, DocumentMetadata


# ---------------------------------------------------------------------------
# Chunking tests
# ---------------------------------------------------------------------------


class TestRecursiveChunker:
    def test_short_text_single_chunk(self):
        chunker = RecursiveChunker(chunk_size=512, chunk_overlap=128)
        chunks = chunker.chunk("Hello world.", doc_id="d1", filename="test.txt")
        assert len(chunks) == 1
        assert chunks[0].text == "Hello world."
        assert chunks[0].metadata.doc_id == "d1"

    def test_long_text_multiple_chunks(self):
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
        text = " ".join(["word"] * 200)  # ~200 tokens
        chunks = chunker.chunk(text, doc_id="d2", filename="long.txt")
        assert len(chunks) > 1
        for c in chunks:
            assert c.metadata.doc_id == "d2"
            assert c.metadata.chunk_strategy == "recursive"

    def test_metadata_populated(self):
        chunker = RecursiveChunker(chunk_size=512)
        chunks = chunker.chunk(
            "Some text.", doc_id="d3", filename="f.pdf", title="Title"
        )
        assert chunks[0].metadata.filename == "f.pdf"
        assert chunks[0].metadata.title == "Title"
        assert chunks[0].metadata.total_chunks == 1
        assert chunks[0].metadata.chunk_index == 0


class TestFixedOverlapChunker:
    def test_produces_chunks(self):
        chunker = FixedOverlapChunker(chunk_size=20, chunk_overlap=5)
        text = " ".join(["token"] * 100)
        chunks = chunker.chunk(text, doc_id="d4")
        assert len(chunks) > 1
        for c in chunks:
            assert c.metadata.chunk_strategy == "fixed_overlap"


class TestChunkerFactory:
    def test_recursive(self):
        c = get_chunker("recursive", chunk_size=256)
        assert isinstance(c, RecursiveChunker)

    def test_fixed_overlap(self):
        c = get_chunker("fixed_overlap", chunk_size=256)
        assert isinstance(c, FixedOverlapChunker)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown chunk strategy"):
            get_chunker("nonexistent")


# ---------------------------------------------------------------------------
# Retrieval tests
# ---------------------------------------------------------------------------


def _make_chunk(chunk_id: str, text: str, doc_id: str, embedding: list[float]) -> Chunk:
    """Helper to create a Chunk with an embedding."""
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        metadata=DocumentMetadata(doc_id=doc_id, filename="test.txt"),
        embedding=embedding,
    )


class TestFAISSStore:
    def test_add_and_search(self):
        store = FAISSStore(dim=4)
        c1 = _make_chunk("c1", "hello world", "d1", [1.0, 0.0, 0.0, 0.0])
        c2 = _make_chunk("c2", "goodbye world", "d1", [0.0, 1.0, 0.0, 0.0])
        store.add([c1, c2])
        assert store.count() == 2

        results = store.search(np.array([1.0, 0.0, 0.0, 0.0]), top_k=2)
        assert len(results) == 2
        assert results[0][0] == "c1"  # nearest neighbor

    def test_delete_by_doc_id(self):
        store = FAISSStore(dim=4)
        c1 = _make_chunk("c1", "a", "d1", [1.0, 0.0, 0.0, 0.0])
        c2 = _make_chunk("c2", "b", "d2", [0.0, 1.0, 0.0, 0.0])
        store.add([c1, c2])
        removed = store.delete_by_doc_id("d1")
        assert removed == 1
        assert store.count() == 1

    def test_empty_search(self):
        store = FAISSStore(dim=4)
        results = store.search(np.array([1.0, 0.0, 0.0, 0.0]), top_k=5)
        assert results == []


class TestBM25Index:
    def test_add_and_search(self):
        index = BM25Index()
        # BM25 IDF needs >=3 documents to produce nonzero scores
        # (terms in >50% of a 2-doc corpus get IDF≈0)
        chunks = [
            Chunk(
                chunk_id="c1",
                text="kubernetes pod deployment scaling cluster management",
            ),
            Chunk(
                chunk_id="c2",
                text="python machine learning deep neural network training",
            ),
            Chunk(
                chunk_id="c3",
                text="database sql query performance optimization indexing",
            ),
            Chunk(
                chunk_id="c4", text="docker container image registry build push pull"
            ),
        ]
        index.add(chunks)
        assert index.count() == 4

        results = index.search("kubernetes pod cluster", top_k=5)
        assert len(results) >= 1
        assert results[0][0] == "c1"

    def test_empty_search(self):
        index = BM25Index()
        results = index.search("anything", top_k=5)
        assert results == []


class TestHybridRetriever:
    def test_search_returns_fused_results(self):
        store = FAISSStore(dim=4)
        bm25 = BM25Index()

        def embed_fn(text: str) -> np.ndarray:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        retriever = HybridRetriever(store, bm25, embed_fn)

        c1 = _make_chunk("c1", "kubernetes cluster", "d1", [1.0, 0.0, 0.0, 0.0])
        c2 = _make_chunk("c2", "python flask app", "d2", [0.0, 1.0, 0.0, 0.0])
        retriever.ingest([c1, c2])

        results = retriever.search("kubernetes", vector_top_k=10, bm25_top_k=10)
        assert len(results) >= 1
        # c1 should rank higher (keyword + vector match)
        assert results[0].chunk_id == "c1"
        assert results[0].rrf_score > 0


# ---------------------------------------------------------------------------
# Reranker tests
# ---------------------------------------------------------------------------


class TestDummyReranker:
    def test_preserves_rrf_order(self):
        reranker = DummyReranker()
        c1 = Chunk(chunk_id="c1", text="first")
        c1.rrf_score = 0.9
        c2 = Chunk(chunk_id="c2", text="second")
        c2.rrf_score = 0.5
        result = reranker.rerank("query", [c2, c1], top_k=2)
        assert result[0].chunk_id == "c1"
        assert result[1].chunk_id == "c2"

    def test_top_k_truncation(self):
        reranker = DummyReranker()
        chunks = [Chunk(chunk_id=f"c{i}", text=f"text {i}") for i in range(20)]
        for i, c in enumerate(chunks):
            c.rrf_score = float(20 - i)
        result = reranker.rerank("query", chunks, top_k=5)
        assert len(result) == 5


# ---------------------------------------------------------------------------
# Citation tests
# ---------------------------------------------------------------------------


class TestCitationFormatter:
    def test_build_citations(self):
        formatter = CitationFormatter()
        c = Chunk(
            chunk_id="c1",
            text="Kubernetes is an open-source container orchestration platform.",
            metadata=DocumentMetadata(
                doc_id="d1", filename="k8s.pdf", page_number=3, chunk_index=0
            ),
        )
        c.rerank_score = 0.95
        citations = formatter.build_citations([c])
        assert len(citations) == 1
        assert citations[0].doc_id == "d1"
        assert citations[0].page_number == 3
        assert citations[0].relevance_score == 0.95

    def test_format_context_for_llm(self):
        formatter = CitationFormatter()
        c = Chunk(
            chunk_id="c1",
            text="Test passage content.",
            metadata=DocumentMetadata(doc_id="d1", filename="test.pdf", page_number=1),
        )
        context = formatter.format_context_for_llm([c])
        assert "[1]" in context
        assert "test.pdf" in context
        assert "Test passage content." in context

    def test_snippet_truncation(self):
        formatter = CitationFormatter()
        long_text = "x" * 300
        c = Chunk(
            chunk_id="c1",
            text=long_text,
            metadata=DocumentMetadata(doc_id="d1"),
        )
        c.rerank_score = 0.5
        citations = formatter.build_citations([c])
        assert len(citations[0].text_snippet) <= 203  # 200 + "..."


# ---------------------------------------------------------------------------
# LLM client tests
# ---------------------------------------------------------------------------


class TestDummyLLMClient:
    def test_generates_answer_with_references(self):
        llm = DummyLLMClient()
        chunks = [Chunk(chunk_id=f"c{i}", text=f"passage {i}") for i in range(3)]
        answer, latency = llm.generate("What is X?", chunks)
        assert "[1]" in answer
        assert "[2]" in answer
        assert "[3]" in answer
        assert latency >= 0

    def test_health_check(self):
        llm = DummyLLMClient()
        assert llm.health_check() is True
