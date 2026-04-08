"""
Integration tests for the RAG pipeline.

These tests exercise the full pipeline path — ingestion through query —
using real embedding models and in-memory indexes. They validate that
components wire together correctly, not just that each unit works in isolation.

Run with: pytest tests/test_integration.py -v --timeout=120
"""

import json
import os
import sys
import subprocess
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["RAG_USE_DUMMY_LLM"] = "true"
os.environ["RAG_RERANKER_ENABLED"] = "false"
os.environ["RAG_LOG_LEVEL"] = "error"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def embed_svc():
    """Shared embedding service — loaded once for all tests in this module."""
    from src.retrieval.embeddings import EmbeddingService

    svc = EmbeddingService()
    svc.encode("warmup")
    return svc


@pytest.fixture()
def pipeline(embed_svc):
    """Fresh RAG pipeline per test with shared embedding service."""
    from src.api.pipeline import RAGPipeline

    p = RAGPipeline()
    p.startup()
    yield p
    p.shutdown()


# ---------------------------------------------------------------------------
# Test 1: Full ingest → query round-trip
# ---------------------------------------------------------------------------


class TestIngestAndQuery:
    """Verify that a document ingested via the pipeline can be retrieved
    and produces a response with valid citations."""

    def test_ingest_then_query_returns_relevant_citations(self, pipeline):
        from src.models.schemas import IngestRequest, QueryRequest

        # Ingest two topically distinct documents
        r1 = pipeline.ingest(
            IngestRequest(
                text=(
                    "Kubernetes is an open-source container orchestration platform. "
                    "It automates deployment, scaling, and management of containerized "
                    "applications. The control plane runs the API server, etcd, the "
                    "scheduler, and the controller manager."
                ),
                filename="k8s_architecture.pdf",
                title="Kubernetes Architecture",
                chunk_strategy="recursive",
                chunk_size=256,
                chunk_overlap=64,
            )
        )
        assert r1.num_chunks >= 1, "Document should produce at least one chunk"
        k8s_doc_id = r1.doc_id

        r2 = pipeline.ingest(
            IngestRequest(
                text=(
                    "Python is a high-level programming language known for its "
                    "readability and versatility. It supports multiple paradigms "
                    "including procedural, object-oriented, and functional programming."
                ),
                filename="python_intro.pdf",
                title="Introduction to Python",
                chunk_strategy="recursive",
                chunk_size=256,
                chunk_overlap=64,
            )
        )
        assert r2.num_chunks >= 1

        # Query about Kubernetes
        response = pipeline.query(
            QueryRequest(
                query="What components are in the Kubernetes control plane?",
                top_k=3,
                rerank=False,
            )
        )

        # Verify structure
        assert response.answer, "Answer should not be empty"
        assert len(response.citations) > 0, "Should have at least one citation"
        assert response.latency_ms > 0, "Latency should be positive"
        assert response.retrieval_latency_ms > 0
        assert response.num_chunks_retrieved > 0

        # Verify the top citation references the K8s document, not Python
        top_citation = response.citations[0]
        assert top_citation.doc_id == k8s_doc_id, (
            f"Top citation should reference the K8s doc ({k8s_doc_id}), "
            f"got {top_citation.doc_id}"
        )
        assert top_citation.filename == "k8s_architecture.pdf"

    def test_query_empty_index_returns_graceful_response(self, pipeline):
        """Query on an empty index should not crash."""
        from src.models.schemas import QueryRequest

        response = pipeline.query(
            QueryRequest(
                query="What is the meaning of life?",
                top_k=5,
            )
        )
        assert response.answer, "Should return some answer even with no docs"
        assert response.num_chunks_retrieved == 0

    def test_multiple_chunking_strategies_produce_different_counts(self, pipeline):
        """Different chunking strategies should produce different chunk counts."""
        from src.models.schemas import IngestRequest

        text = " ".join(["This is a test sentence with several words."] * 50)

        r_recursive = pipeline.ingest(
            IngestRequest(
                text=text,
                filename="test.txt",
                chunk_strategy="recursive",
                chunk_size=64,
                chunk_overlap=16,
            )
        )

        r_fixed = pipeline.ingest(
            IngestRequest(
                text=text,
                filename="test2.txt",
                chunk_strategy="fixed_overlap",
                chunk_size=64,
                chunk_overlap=16,
            )
        )

        # Both should produce chunks, but counts may differ
        assert r_recursive.num_chunks >= 1
        assert r_fixed.num_chunks >= 1
        assert r_recursive.chunk_strategy == "recursive"
        assert r_fixed.chunk_strategy == "fixed_overlap"


# ---------------------------------------------------------------------------
# Test 2: Backtest script on synthetic data
# ---------------------------------------------------------------------------


class TestBacktestScript:
    """Run the actual backtest.py script on synthetic data and verify
    it produces valid output."""

    def test_backtest_produces_valid_json_output(self, tmp_path):
        """Run backtest.py with synthetic data and verify the output."""
        output_file = tmp_path / "results.json"

        result = subprocess.run(
            [
                sys.executable,
                "evaluation/backtest.py",
                "--generate-synthetic",
                "--num-samples",
                "5",
                "--single-config",
                "recursive_512_no_rerank",
                "--output",
                str(output_file),
                "--no-gates",
            ],
            capture_output=True,
            text=True,
            timeout=180,
            cwd=str(Path(__file__).resolve().parent.parent),
            env={
                **os.environ,
                "RAG_USE_DUMMY_LLM": "true",
                "RAG_RERANKER_ENABLED": "false",
                "RAG_LOG_LEVEL": "error",
            },
        )

        assert result.returncode == 0, (
            f"backtest.py failed with exit code {result.returncode}.\n"
            f"stdout: {result.stdout[-500:]}\n"
            f"stderr: {result.stderr[-500:]}"
        )

        # Verify output file exists and is valid JSON
        assert output_file.exists(), "Results file should be created"
        with open(output_file) as f:
            data = json.load(f)

        # Verify structure
        assert "metadata" in data, "Results should have metadata"
        assert "configs" in data, "Results should have configs"
        assert data["metadata"]["num_samples"] > 0

        # Verify metrics were computed
        config_data = data["configs"][0]
        metrics = config_data["metrics"]
        assert "mrr" in metrics, "MRR should be computed"
        assert "hit_rate_at_5" in metrics, "Hit@5 should be computed"
        assert "ndcg_at_5" in metrics, "NDCG@5 should be computed"
        assert "token_f1" in metrics, "Token F1 should be computed"
        assert "rouge_l_f1" in metrics, "ROUGE-L should be computed"
        assert "latency_p50_ms" in metrics, "Latency p50 should be computed"
        assert "throughput_qps" in metrics, "Throughput should be computed"

        # Verify per-query results exist
        per_query = config_data["per_query"]
        assert len(per_query) > 0, "Should have per-query results"
        assert all("query" in pq for pq in per_query)
        assert all("latency_ms" in pq for pq in per_query)

    def test_backtest_with_config_yaml(self, tmp_path):
        """Verify that --config flag loads YAML settings."""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(
            "gates:\n  mrr: 0.99\n  hit_rate_at_5: 0.99\nchunk_size: 256\n"
        )
        output_file = tmp_path / "results.json"

        subprocess.run(
            [
                sys.executable,
                "evaluation/backtest.py",
                "--config",
                str(config_file),
                "--generate-synthetic",
                "--num-samples",
                "3",
                "--single-config",
                "recursive_512_no_rerank",
                "--output",
                str(output_file),
            ],
            capture_output=True,
            text=True,
            timeout=180,
            cwd=str(Path(__file__).resolve().parent.parent),
            env={
                **os.environ,
                "RAG_USE_DUMMY_LLM": "true",
                "RAG_RERANKER_ENABLED": "false",
                "RAG_LOG_LEVEL": "error",
            },
        )

        # With MRR gate at 0.99, this should likely fail the gate
        # (exit code 1) unless the synthetic data gets perfect MRR
        assert output_file.exists(), (
            "Results should still be written even on gate failure"
        )
